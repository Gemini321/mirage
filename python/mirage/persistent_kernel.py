import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig
import re
import shlex
from pathlib import Path
from typing import Optional

from .core import *
from .kernel import get_key_paths, KNGraph, TBGraph
from .speculative import (
    SpecDecodeConfig,
    PromptLookupConfig,
)

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer;
  std::vector<void*> meta_tensors;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiL", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id);

  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  launch_persistent_kernel();

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""

valid_persistent_kernel_modes = {"offline", "online", "onepass"}

def get_compile_command(
    mpk,
    target_cc,
    cc,
    file_name,
    py_include_dir,
    mirage_home_path,
    mirage_inc_path,
    mirage_deps_path,
    nvshmem_inc_path,
    nvshmem_lib_path,
    mpi_inc_path,
    mpi_lib_path,
    py_so_path,
    profiling,
    use_nvshmem,
    num_workers=None,
    num_local_schedulers=None,
    num_remote_schedulers=None,
    use_cutlass_kernel=True,
):
    max_worker_per_scheduler = 128
    if num_workers != None and num_local_schedulers != None and num_remote_schedulers != None:
        min_schedulers = 0
        if num_remote_schedulers == 0:
            min_schedulers = num_local_schedulers
        else:
            min_schedulers = min(num_local_schedulers, num_remote_schedulers)
        # advance by 1 for the scheduler who are handling the not divisiable num_worker.
        max_worker_per_scheduler = (num_workers // min_schedulers) + 1

    common_cmd = [
        cc,
        file_name,
        "-O3",
        # Use following flags when debugging
        # "-O0",
        # "-g",
        # "-G",
        "--ptxas-options=-v",
        "-Xptxas=-v",
        "-lineinfo",
        f"-I{py_include_dir}",
        f"-I{mirage_inc_path}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
        f"-I{os.path.join(mirage_home_path, 'deps/json/include')}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
        f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
    ]

    flags = [
        "-shared",
        "-std=c++17",
        "-rdc=false" if not use_nvshmem else "-rdc=true",
        "-use_fast_math",
        "-lcuda",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        py_so_path,
    ]
    flags = flags + [f"-DMPK_TARGET_CC={target_cc}", "-DMIRAGE_BACKEND_USE_CUDA"]

    if mpk.mode == "offline":
        flags = flags + ["-DMODE_OFFLINE"]
    elif mpk.mode == "online":
        flags = flags + ["-DMODE_ONLINE"]
    elif mpk.mode == "onepass":
        flags = flags + ["-DMODE_ONEPASS"]
    else:
        raise ValueError(f"Invalid persistent kernel mode: {mpk.mode}")

    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}"]
    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}"]
    flags = flags + [f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}"]
    flags = flags + [f"-DMPK_PAGE_SIZE={mpk.page_size}"]
    flags = flags + [f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}"]
    # Use when debugging
    # flags = flags + [f"-DMPK_ENABLE_VERBOSE"]
    flags = flags + [f"-DMIRAGE_USE_SETMAXNREG"]

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        nvshmem_flags = ["-DUSE_NVSHMEM", "-ccbin=mpic++", "-lnvshmem_host", "-lnvshmem_device", "-lmpi"]
        common_cmd = common_cmd + nvshmem_cmd
        flags = flags + nvshmem_flags

    if target_cc == 90:
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_HOPPER",
            "-DNDEBUG",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 100:
        specific_cmd = [
            "-arch=sm_100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_BLACKWELL",
        ]
    else:
        specific_cmd = [
            "-arch=native",
        ]
    
    if profiling:
        flags = flags + ["-DMPK_ENABLE_PROFILING"]

    return common_cmd + specific_cmd + flags


def _maybe_patch_megakernel_so(
    so_path: str,
    *,
    reg: Optional[int],
    kernel_substr: str = "worker_kernel",
    dump_dir: Optional[str] = None,
) -> str:
    """
    Optionally patch embedded worker_kernel regcount metadata inside a compiled
    megakernel shared library, returning the path to the (possibly) patched .so.
    """
    if reg is None:
        return so_path
    if not (32 <= int(reg) <= 256):
        raise ValueError(f"patch reg must be in 32 ~ 256, got {reg}")

    repo_mirage_root = Path(__file__).resolve().parents[2]
    patch_script = repo_mirage_root / "patch_mpk_worker_kernel_so.py"
    if not patch_script.exists():
        raise FileNotFoundError(f"missing patch script: {patch_script}")

    in_path = Path(so_path)
    out_path = in_path.with_suffix(in_path.suffix + ".patched.so")
    dump_path = Path(dump_dir) if dump_dir else out_path.parent

    cmd = [
        sys.executable,
        str(patch_script),
        "--so",
        str(in_path),
        "--out",
        str(out_path),
        "--reg",
        str(int(reg)),
        "--kernel-substr",
        kernel_substr,
        "--dump-dir",
        str(dump_path),
        # "--no-sass",
        "--no-launch-test",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return str(out_path)


class PersistentKernel:
    def __init__(
        self,
        mode: str,
        world_size: int,
        mpi_rank: int,
        num_workers: int,
        num_local_schedulers: int,
        num_remote_schedulers: int,
        max_seq_length: int,
        max_num_batched_requests: int,
        max_num_batched_tokens: int,
        max_num_pages: int,
        page_size: int,
        eos_token_id: int64,
        meta_tensors: dict,
        profiler_tensor: torch.Tensor,
        trace_name: str,
        spec_decode_config: SpecDecodeConfig,
        use_cutlass_kernel: bool
    ):
        self.__finalized__ = False
        self._is_compiled = False
        if mode not in valid_persistent_kernel_modes:
            raise ValueError(f"Invalid persistent kernel mode: {mode}")
        self.mode = mode
        self.world_size = world_size
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        self.max_seq_length = max_seq_length
        self.max_num_batched_requests = max_num_batched_requests
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.eos_token_id = eos_token_id
        self.kn_graph = KNGraph(CyKNGraph(disable_fingerprint=True))
        self.meta_tensors = meta_tensors
        self.profiler_tensor = profiler_tensor
        self.trace_name = trace_name
        self.use_nvshmem = True if world_size > 1 else False
        self.spec_decode_config = spec_decode_config
        self.use_cutlass_kernel = use_cutlass_kernel
        self._spec_decode_handlers = {
            "promptlookup": self.prompt_lookup_spec_handler,
        }
        self._spec_verify_handlers = {
            "promptlookup": self.prompt_lookup_verify_handler,
        }
        # determine total number of requests for offline serving
        self.total_num_requests = meta_tensors["tokens"].shape[0]
        assert self.max_seq_length == meta_tensors["tokens"].shape[1]
        self.target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        # Check tensor shapes
        qo_indptr_buffer = self.meta_tensors["qo_indptr_buffer"]
        assert qo_indptr_buffer.shape == (self.max_num_batched_requests+1,)
        paged_kv_indptr_buffer = self.meta_tensors["paged_kv_indptr_buffer"]
        assert paged_kv_indptr_buffer.shape == (self.max_num_batched_requests+1,)
        paged_kv_indices_buffer = self.meta_tensors["paged_kv_indices_buffer"]
        assert paged_kv_indices_buffer.shape == (self.max_num_pages,)
        paged_kv_last_page_len_buffer = self.meta_tensors["paged_kv_last_page_len_buffer"]
        assert paged_kv_last_page_len_buffer.shape == (self.max_num_batched_requests,)

    def attach_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        dims = tuple([d for d in torch_tensor.shape])
        strides = tuple([s for s in torch_tensor.stride()])
        # Assert a row-major layout
        for d in range(len(dims) - 1):
            assert strides[d] == strides[d + 1] * dims[d + 1]
        dtype = convert_torch_type_to_dtype(torch_tensor.dtype)
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        self.kn_graph.attach_torch_tensor(t, torch_tensor, name)
        return t

    def new_tensor(
        self,
        dims: tuple,
        strides: tuple = None,
        dtype: dtype = bfloat16,
        name: str = None,
        io_category: str = "cuda_tensor",
    ) -> DTensor:
        # Assert a row-major layout
        # if strides is not None:
        #     for d in range(len(dims) - 1):
        #         assert strides[d] == strides[d + 1] * dims[d + 1]
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        if io_category == "cuda_tensor":
            self.kn_graph.attach_cuda_tensor(t, name)
        elif io_category == "nvshmem_tensor":
            self.kn_graph.attach_nvshmem_tensor(t, name)
        else:
            raise RuntimeError(f"Invalid io_category: {io_category}")
        return t

    def fuse_tensors(
        self, inputs: list[DTensor], fused_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support fusing the 0-th dimension
        assert fused_dim == 0
        t = self.kn_graph.fuse_tensors(inputs, fused_dim, num_groups, name)
        return t

    def shuffle_tensors(
        self, inputs: list[DTensor], shuffled_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support shuffling the 0-th dimension
        assert shuffled_dim == 0
        t = self.kn_graph.shuffle_tensors(inputs, shuffled_dim, num_groups, name)
        return t

    def embed_layer(
        self,
        input: DTensor, # [batch_size, num_spec_tokens]
        weight: DTensor, # [vocab_size, hidden_size]
        output: DTensor, # [batch_size, hidden_size]
        grid_dim: tuple,
        block_dim: tuple,
        input_source: int = 0, # 0: all_tokens, 1: input_token
    ):
        # TODO: Support batch size > 1
        # tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # tb_graph.new_input(input, (-1, -1, -1), -1, True)
        # tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        # tb_graph.new_input(output, (-1, -1, -1), -1, True)
        # self.kn_graph.customized([input, weight, output], tb_graph)
        # self.kn_graph.register_task(tb_graph, "embedding")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(weight, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "embedding" if self.target_cc == 90 else "embedding", [input_source])

    def rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/output are 2D tensors
        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(output, (0, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm")

    def rmsnorm_linear_layer(
        self,
        input: DTensor,
        weight_norm: DTensor,
        weight_linear: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/weight_linear/output are 2D tensors
        assert input.num_dims == 2
        assert weight_linear.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight_norm, (-1, -1, -1), 0, True)
        tb_graph.new_input(weight_linear, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "rmsnorm_linear")

    def attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "attention", params)
        
    def single_batch_extend_attention_layer(
        self,
        input: DTensor, # [6, 6144]
        k_cache: DTensor, 
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple, # (6, 8, 1)
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim # 32
        rotary_embed = 0
        output_stride = output.dim(1)

        extend_num = input.dim(0) - 1
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: extend_num
        # params[5]: output_stride
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, extend_num, output_stride]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "single_batch_extend_attention", params)

    def paged_attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_hopper", params)
        elif self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_sm100", params)
        else:
            self.kn_graph.register_task(tb_graph, "paged_attention", params)

    
    def paged_attention_split_kv_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        lse: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        assert output.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv * head_dim / world_size, num_kv_heads)
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        # params[6]: num_kv_chunks
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size, num_kv_chunks]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse, (-1, 2, 1), -1, True)
        tb_graph.new_input(output, (-1, 2, 1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                lse,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_sm100", params)
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_hopper", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")

    def paged_attention_split_kv_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)
        assert output_tmp.num_dims == 3  # (num_tokens, num_chunks, hidden_size / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_qo_heads_per_kv = num_q_heads / grid_dim[1]
        num_kv_heads = grid_dim[1]
        # params[0]: num_qo_heads_per_kv
        # params[1]: head_dim
        # params[2]: max_seq_len
        # params[3]: page_size
        # params[4]: num_kv_heads
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length, self.page_size, num_kv_heads]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(lse, (-1, 2, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, 2, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                lse,
                output_tmp,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_merge_sm100", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")
            
    # MoE Layers
    def tensor_init_layer(
        self,
        input: DTensor,
        dummy_input: DTensor,
        dummy_output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that output
        assert input.num_dims == 2  # (batch_size, output_size)
        assert dummy_input.num_dims == 2 # (batch_size, hidden_size)
        assert dummy_output.num_dims == 2 # (batch_size, output_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_output, (0, -1, -1), -1, True)
        self.kn_graph.customized([input, dummy_input, dummy_output], tb_graph)

        self.kn_graph.register_task(tb_graph, "tensor_init")
    
    def moe_topk_softmax_routing_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, num_experts)
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (num_experts + 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_masks, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, moe_topk_weight, moe_routing_indices, moe_masks], tb_graph)

        self.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100")
        
    def moe_w13_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 3  # (num_experts, 2*intermediate_size, hidden_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, 2*intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm90")
        else:
            assert False
            
    def moe_silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3 # (batch_size, num_expert_per_tok, 2 * intermediate_size)
        assert output.num_dims == 3 # (batch_size, num_expert_per_tok, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_silu_mul")
            
    def moe_w2_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor, 
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_expert_per_tok, intermediate_size)
        assert weight.num_dims == 3  # (num_experts, hidden_size, intermediate_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm90")
        else:
            assert False
        
    def moe_mul_sum_add_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_experts_per_tok, hidden_size)
        assert weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 2, -1), -1, True)
        tb_graph.new_input(weight, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)

        self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100")

    def splitk_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), 1, True)
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "splitk_linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "splitk_linear_swapAB_hopper")
        else:
            assert False

    def linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "linear_sm100")
        elif self.target_cc == 90:
            if weight.dim(0) // grid_dim[0] <= 64:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
        elif self.target_cc == 80:
            self.kn_graph.register_task(tb_graph, "linear")
        else:
            assert False

    def linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "linear_with_residual_sm100")
        elif self.target_cc == 90:
            if weight.dim(0) // grid_dim[0] <= 64:
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_with_residual_hopper")
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper")
        elif self.target_cc == 80:
            self.kn_graph.register_task(tb_graph, "linear_with_residual")
        else:
            assert False

    def allreduce_layer(
        self,
        input: DTensor,
        buffer: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size)
        assert buffer.num_dims == 3  # (world_size, batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        # params[0]: num_gpus
        # params[1]: my_gpu_id
        params = [self.world_size, self.mpi_rank]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), -1, True)
        tb_graph.new_input(buffer, (2, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, buffer, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "allreduce", params)

    def silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2 # (batch_size, 2 * intermediate_size)
        assert output.num_dims == 2 # (batch_size, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul" if self.target_cc == 90 else "silu_mul")

    def identity_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        dependent_tensor: DTensor = None,
    ):
        # TODO: Add support from kn_graph
        last_dim = 0
        assert input.num_dims == output.num_dims
        for i in range(input.num_dims):
            assert input.dim(i) == output.dim(i)
            last_dim = i
        assert last_dim == 1 or last_dim == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (last_dim, -1, -1), 1, True)
        tb_graph.new_input(output, (last_dim, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "identity")

    def silu_mul_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, 2*intermediate_size)
        assert weight.num_dims == 2  # (hidden_size, intermediate_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul_linear_with_residual")

    def argmax_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "argmax")

    def argmax_partial_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert len(output) == 2
        output_value, output_index = output
        assert output_value.num_dims == 2  # (batch_size, num_tasks)
        assert output_index.num_dims == 2  # (batch_size, num_tasks)
        num_tasks = grid_dim[0]
        self.argmax_partial_output_size = input.dim(1) // num_tasks
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, 0, -1), -1, True)
        tb_graph.new_input(output_value, (1, 0, -1), -1, True)
        tb_graph.new_input(output_index, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, output_value, output_index], tb_graph)
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "argmax_partial_sm100", [num_tasks])
        else:
            self.kn_graph.register_task(tb_graph, "argmax_partial", [num_tasks])

    def argmax_reduce_layer(
        self,
        input: tuple[DTensor, DTensor],
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert len(input) == 2
        input_value, input_index = input
        assert input_value.num_dims == 2  # (batch_size, num_tasks)
        assert input_index.num_dims == 2  # (batch_size, num_tasks)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_value, (1, 0, -1), -1, True)
        tb_graph.new_input(input_index, (1, 0, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True) #TODO: Make sure the output map is expected
        self.kn_graph.customized([input_value, input_index, output], tb_graph)
        if self.target_cc == 100:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce_sm100", [self.argmax_partial_output_size]
            )
        else:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce", [self.argmax_partial_output_size]
            )

    def sampling_sm100_layer(
        self,
        logits: DTensor,      # [batch_size, vocab_size]
        output: DTensor,      # [batch_size, 1]
        grid_dim: tuple,
        block_dim: tuple,
        seed: int = 42,
    ):
        """Sampling from logits using Gumbel-Max trick for stochastic token generation."""
        assert logits.num_dims == 2      # (batch_size, vocab_size)
        assert output.num_dims == 2      # (batch_size, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized([logits, output], tb_graph)

        # Register task with seed parameter
        self.kn_graph.register_task(tb_graph, "sampling_sm100", [seed])

    def find_ngram_partial_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, seq_len)
        assert output.num_dims == 2  # (batch_size, num_tasks)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_partial", [ngram_size])
        
    def find_ngram_global_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3, spec_length: int = 5):
        # Currently assume that input/output
        assert len(input) == 2
        partial_results, tokens = input
        assert partial_results.num_dims == 2  # (batch_size, num_tasks)
        assert tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_results, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([partial_results, tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_global", [ngram_size, spec_length])

    def prompt_lookup_spec_handler(
        self, 
        spec_decode_config: PromptLookupConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        partial_ngram_output = self.new_tensor(
            dims=(tokens.dim(0), 96),
            dtype=int64,
            name="partial_ngram_output",
            io_category="cuda_tensor",
        )
        self.find_ngram_partial_layer(
            input=tokens, 
            output=partial_ngram_output, 
            grid_dim=grid_dim, 
            block_dim=block_dim, 
            ngram_size=spec_decode_config.ngram_size
        )
        spec_tokens = self.new_tensor(
            dims=(tokens.dim(0), spec_decode_config.spec_length + 1),
            dtype=int64,
            name="spec_tokens",
            io_category="cuda_tensor",
        )   
        self.find_ngram_global_layer(
            input=(partial_ngram_output, tokens), 
            output=spec_tokens, 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1), 
            ngram_size=spec_decode_config.ngram_size,
            spec_length=spec_decode_config.spec_length
        )
        return spec_tokens
    
    def draft_forward_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        method = spec_decode_config.method
        handler = self._spec_decode_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, tokens, grid_dim, block_dim)
    
    def target_verify_greedy_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple):
        # Currently assume that input/output
        # This tensor is not realy used
        assert len(input) == 2
        spec_tokens, target_tokens = input
        assert spec_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert target_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(spec_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([spec_tokens, target_tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "target_verify_greedy")
        
    def prompt_lookup_verify_handler(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        # This tensor is not realy used
        verify_out = self.new_tensor(
            dims=(1, 1),
            dtype=int64,
            name="verify_out",
            io_category="cuda_tensor",
        )
        self.target_verify_greedy_layer(
            input=(spec_tokens, target_output), output=verify_out, grid_dim=grid_dim, block_dim=block_dim
        )
        return verify_out
    
    def verify_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int] = (1, 1, 1),
        block_dim: tuple[int, int, int] = (128, 1, 1),
    ):
        method = spec_decode_config.method
        handler = self._spec_verify_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, spec_tokens, target_output, grid_dim, block_dim)

    def compile(
        self,
        **kwargs,
    ):
        assert not self._is_compiled
        
        output_dir = kwargs.get("output_dir", None)
        report_regs_kw = bool(kwargs.get("report_regs", False))
        noinline_wrappers_kw = bool(kwargs.get("noinline_task_wrappers", False))
        keep_artifacts_kw = bool(kwargs.get("keep_cuda_artifacts", False))
        patch_worker_kernel_reg_kw = kwargs.get("patch_worker_kernel_reg", None)
        if patch_worker_kernel_reg_kw is None:
            env_reg = os.environ.get("MIRAGE_PATCH_WORKER_KERNEL_REG", "").strip()
            patch_worker_kernel_reg_kw = int(env_reg) if env_reg else None

        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = tempdir_obj.name
        results = self.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)

        cuda_code_path = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")
        # check json file
        json_file_path = os.path.join(tempdir, "task_graph.json")
        with open(json_file_path, "w") as f:
            f.write(results["json_file"])
        with open(cuda_code_path, "w") as f:
            cuda_code = results["cuda_code"]

            # Optional: help attribute per-task register usage by making each task
            # branch in `_execute_task` call a dedicated `__noinline__` wrapper.
            # Enable by setting MIRAGE_NOINLINE_TASK_WRAPPERS=1 or passing
            # noinline_task_wrappers=True to compile().
            #
            # If MIRAGE_REPORT_REGS/report_regs is enabled, we also auto-enable
            # these wrappers so ptxas can report per-branch register usage.
            report_regs_env = os.environ.get("MIRAGE_REPORT_REGS", "0") == "1"
            noinline_env = os.environ.get("MIRAGE_NOINLINE_TASK_WRAPPERS", "0") == "1"
            enable_noinline_wrappers = noinline_wrappers_kw or report_regs_kw or report_regs_env or noinline_env
            if enable_noinline_wrappers:
                try:
                    cuda_code = _inject_noinline_task_wrappers(cuda_code)
                except Exception as e:
                    print(f"Warning: failed to inject noinline wrappers: {e}")
            f.write(cuda_code + HARD_CODE)
            
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{self.mpi_rank}.cu"))
            shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{self.mpi_rank}.json"))

        cc = shutil.which("nvcc")
        if cc is None:
            raise RuntimeError(
                "nvcc not found. Please make sure you have installed CUDA."
            )
        # This function was renamed and made public in Python 3.10
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

        # find mirage home
        if "MIRAGE_HOME" in os.environ:
            MIRAGE_HOME_PATH = os.environ.get("MIRAGE_HOME")
        else:
            raise RuntimeError(
                "MIRAGE_HOME unspecified; Please set MIRAGE_HOME to be the root of the Mirage folder"
            )

        NVSHMEM_INC_PATH = None
        NVSHMEM_LIB_PATH = None
        MPI_INC_PATH = None
        MPI_LIB_PATH = None
        if self.use_nvshmem:
            # find nvshmem include folder and library folder
            if "NVSHMEM_INC_PATH" in os.environ:
                NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}"
                    )
            else:
                NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Cannot find nvshmem.h, please set environment variable NVSHMEM_INC_PATH"
                    )
            # find nvshmem shared library
            if "NVSHMEM_LIB_PATH" in os.environ:
                NVSHMEM_LIB_PATH = os.environ.get("NVSHMEM_LIB_PATH")
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_LIB_PATH is set but cannot find libnvshmem.a at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/"
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Cannot find libnvshmem.a, please set environment variable NVSHMEM_LIB_PATH"
                    )
            # find mpi include foler
            if "MPI_INC_PATH" in os.environ:
                MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_INC_PATH is set but cannot find mpi.h at {header_file_path}"
                    )
            else:
                MPI_INC_PATH = "/usr/include/"
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Cannot find mpi.h, please set environment variable MPI_INC_PATH"
                    )
            # find mpi shared library
            if "MPI_LIB_PATH" in os.environ:
                MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_LIB_PATH is set but cannot find libmpi.so at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/"
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Cannot find libmpi.so, please set environment variable MPI_LIB_PATH"
                    )

        cc_cmd = get_compile_command(
            mpk=self,
            target_cc=self.target_cc,
            cc=cc,
            file_name=cuda_code_path,
            py_include_dir=py_include_dir,
            mirage_home_path=MIRAGE_HOME_PATH,
            mirage_inc_path=INCLUDE_PATH,
            mirage_deps_path=DEPS_PATH,
            nvshmem_inc_path=NVSHMEM_INC_PATH,
            nvshmem_lib_path=NVSHMEM_LIB_PATH,
            mpi_inc_path=MPI_INC_PATH,
            mpi_lib_path=MPI_LIB_PATH,
            py_so_path=so_path,
            profiling=True if self.profiler_tensor is not None else False,
            use_nvshmem=self.use_nvshmem,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_local_schedulers, 
            num_remote_schedulers=self.num_remote_schedulers,
            use_cutlass_kernel=self.use_cutlass_kernel,
        )
        print("Compiling megakernel using the following command line:")
        print(cc_cmd)

        report_regs = report_regs_kw or os.environ.get("MIRAGE_REPORT_REGS", "0") == "1"
        keep_artifacts = keep_artifacts_kw or os.environ.get("MIRAGE_KEEP_CUDA_ARTIFACTS", "0") == "1"
        if report_regs and "--resource-usage" not in cc_cmd:
            cc_cmd = cc_cmd + ["--resource-usage"]
        if keep_artifacts and "--keep" not in cc_cmd:
            keep_dir = os.path.join(output_dir or tempdir, "nvcc_keep")
            os.makedirs(keep_dir, exist_ok=True)
            cc_cmd = cc_cmd + ["--keep", f"--keep-dir={keep_dir}"]

        proc = subprocess.run(
            cc_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            # Keep stdout/stderr for debugging
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cc_cmd)

        if report_regs:
            # nvcc/ptxas messages are not consistently routed; parse both.
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            report = _parse_ptxas_register_report(combined)
            if report:
                smem_static = _dump_static_smem_bytes_for_functions(
                    so_path=so_path, nvcc_path=cc
                )
                regprobe = {k: v for k, v in report.items() if "__mirage_regprobe_" in k}
                to_print = regprobe if regprobe else report
                title = "Per-task resource usage (ptxas + inferred dynamic smem):" if regprobe else "Per-task wrapper register usage (ptxas):"
                print(title)
                for name, regs in sorted(to_print.items(), key=lambda x: (-x[1], x[0])):
                    static_b = smem_static.get(name, None)
                    m = re.search(r"_dsmem(\d+|U)\b", name)
                    dyn_used = None
                    if m:
                        dyn_used = -1 if m.group(1) == "U" else int(m.group(1))
                    dyn_str = "?" if dyn_used is None or dyn_used < 0 else str(dyn_used)
                    static_str = "?" if static_b is None else str(static_b)
                    total_str = "?"
                    if static_b is not None and dyn_used is not None and dyn_used >= 0:
                        total_str = str(static_b + dyn_used)
                    print(f"  {name}: regs={regs}; smem_dynamic_used={dyn_str}; smem_static={static_str}; smem_total={total_str}")
                if output_dir is not None:
                    try:
                        report_path = os.path.join(output_dir, f"reg_report_rank{self.mpi_rank}.txt")
                        with open(report_path, "w") as rf:
                            for name, regs in sorted(to_print.items(), key=lambda x: (-x[1], x[0])):
                                static_b = smem_static.get(name, -1)
                                m = re.search(r"_dsmem(\d+|U)\b", name)
                                dyn_used = -1
                                if m:
                                    dyn_used = -1 if m.group(1) == "U" else int(m.group(1))
                                total = static_b + dyn_used if static_b >= 0 and dyn_used >= 0 else -1
                                rf.write(f"{name}\tregs={regs}\tsmem_dynamic_used={dyn_used}\tsmem_static={static_b}\tsmem_total={total}\n")
                        print(f"Wrote register report to: {report_path}")
                    except Exception as e:
                        print(f"Warning: failed to write reg report file: {e}")
            else:
                # Provide a helpful hint and a fallback summary.
                print("No per-task wrapper register report found in ptxas output.")
                fallback = _parse_ptxas_register_report(combined, only_wrappers=False)
                if fallback:
                    print("Top functions by register usage (ptxas):")
                    for name, regs in sorted(fallback.items(), key=lambda x: (-x[1], x[0]))[:20]:
                        print(f"  {name}: {regs} regs")

        if output_dir is not None:
            try:
                so_out = os.path.join(output_dir, f"megakernel_rank{self.mpi_rank}.so")
                shutil.copy(so_path, so_out)
                print(f"Wrote megakernel shared library to: {so_out}")
                so_path = so_out
            except Exception as e:
                print(f"Warning: failed to copy megakernel .so to output_dir: {e}")

        import importlib.util

        so_path = _maybe_patch_megakernel_so(
            so_path,
            reg=patch_worker_kernel_reg_kw,
            kernel_substr="worker_kernel",
            dump_dir=output_dir if output_dir is not None else None,
        )

        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.finalize_func = getattr(mod, "finalize_func")
        print("Finished megakernel compilation...")

        #meta_tensors_ptr = [tensor.data_ptr() for tensor in self.meta_tensors]
        meta_tensors = list()
        meta_tensors.append(self.meta_tensors["step"])
        meta_tensors.append(self.meta_tensors["tokens"])
        meta_tensors.append(self.meta_tensors["input_tokens"])
        meta_tensors.append(self.meta_tensors["output_tokens"])
        meta_tensors.append(self.meta_tensors["num_new_tokens"])
        meta_tensors.append(self.meta_tensors["prompt_lengths"])
        meta_tensors.append(self.meta_tensors["qo_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indices_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_last_page_len_buffer"])
        meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        profiler_buffer_ptr = (
            self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        )
        self.init_func(
            meta_tensors_ptr,
            profiler_buffer_ptr,
            self.mpi_rank,
            self.num_workers,
            self.num_local_schedulers,
            self.num_remote_schedulers,
            self.max_seq_length,
            self.total_num_requests,
            self.eos_token_id,
        )

        self._is_compiled = True

        # self.call_func = getattr(mod, "call_func")

    def __call__(self, **kwargs):
        # stream = kwargs.get("stream", None)
        # if stream is None:
        #    stream = torch.cuda.default_stream()
        self.launch_func()
        if self.profiler_tensor is not None:
            from .profiler_persistent import export_to_perfetto_trace

            if self.trace_name:
                trace_name = self.trace_name + ".perfetto-trace"
            else:
                trace_name = f"mirage_{self.mpi_rank}.perfetto-trace"

            export_to_perfetto_trace(self.profiler_tensor, trace_name)

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func()
        self.__finalized__ = True


def _inject_noinline_task_wrappers(cuda_code: str) -> str:
    """
    Transform generated CUDA code by wrapping each `_execute_task` branch into a
    dedicated `__device__ __noinline__` wrapper function, so ptxas can report
    registers per task/variant more reliably.

    Expected generated pattern (example):
      __device__ __forceinline__
      void _execute_task(...) {
        if (task_desc->task_type == TASK_X && task_desc->variant_id == 0) {
          kernel::foo(...);
        }
        else if (...) { ... }
      }
    """
    signature = "__device__ __forceinline__\nvoid _execute_task("
    start = cuda_code.find(signature)
    if start < 0:
        # Try a slightly different whitespace format
        signature = "__device__ __forceinline__\nvoid _execute_task"
        start = cuda_code.find(signature)
        if start < 0:
            raise ValueError("Could not find _execute_task definition in generated CUDA code")

    # Find the opening brace of the function
    brace_open = cuda_code.find("{", start)
    if brace_open < 0:
        raise ValueError("Malformed _execute_task: missing '{'")

    # Find matching closing brace by brace counting
    i = brace_open
    depth = 0
    while i < len(cuda_code):
        ch = cuda_code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_close = i
                break
        i += 1
    else:
        raise ValueError("Malformed _execute_task: missing closing '}'")

    prefix = cuda_code[:start]
    func_text = cuda_code[start : brace_close + 1]
    suffix = cuda_code[brace_close + 1 :]

    # Make `_execute_task` itself noinline to avoid it being merged into callers
    # and to improve ptxas' per-function reporting stability.
    func_text = func_text.replace("__device__ __forceinline__", "__device__ __noinline__", 1)

    # Extract branches in the if/else-if chain
    branch_re = re.compile(
        r"(?:^|\n)\s*(?:else\s+)?if\s*\(\s*task_desc->task_type\s*==\s*(TASK_[A-Z0-9_]+)\s*&&\s*task_desc->variant_id\s*==\s*(\d+)\s*\)\s*\{",
        re.M,
    )
    branches = list(branch_re.finditer(func_text))
    if not branches:
        raise ValueError("No task branches found in _execute_task")

    wrappers: list[str] = []
    probes: list[str] = []
    replacements: list[tuple[int, int, str]] = []

    for m in branches:
        task_type = m.group(1)
        variant_id = int(m.group(2))
        block_start = m.end()  # after '{'
        # Find end of this branch block by brace counting starting at this '{'
        j = block_start
        depth2 = 1
        while j < len(func_text):
            if func_text[j] == "{":
                depth2 += 1
            elif func_text[j] == "}":
                depth2 -= 1
                if depth2 == 0:
                    block_end = j  # position of matching '}'
                    break
            j += 1
        else:
            raise ValueError(f"Unmatched braces in branch {task_type} v{variant_id}")

        original_block = func_text[block_start:block_end]
        dyn_smem = _estimate_dynamic_smem_bytes_from_branch(original_block)
        dyn_smem_suffix = f"_dsmem{dyn_smem}" if dyn_smem >= 0 else "_dsmemU"
        wrapper_name = f"__mirage_task_{task_type}_v{variant_id}{dyn_smem_suffix}"
        probe_name = f"__mirage_regprobe_{task_type}_v{variant_id}{dyn_smem_suffix}"
        wrappers.append(
            "\n".join(
                [
                    f"__device__ __noinline__ void {wrapper_name}(TaskDesc const* task_desc, RuntimeConfig const &runtime_config) {{",
                    "  __shared__ int __mirage_task_done;",
                    original_block.rstrip(),
                    "}",
                    "",
                ]
            )
        )
        probes.append(
            "\n".join(
                [
                    f'extern "C" __global__ __launch_bounds__(256) void {probe_name}(TaskDesc const* task_desc, RuntimeConfig const* runtime_config) {{',
                    f"  {wrapper_name}(task_desc, *runtime_config);",
                    "}",
                    "",
                ]
            )
        )

        # Replace the original block body with a call to wrapper (keep braces)
        replacement_body = f"\n      {wrapper_name}(task_desc, runtime_config);\n"
        replacements.append((block_start, block_end, replacement_body))

    # Apply replacements from back to front
    new_func_text = func_text
    for a, b, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_func_text = new_func_text[:a] + rep + new_func_text[b:]

    injected = prefix + "\n".join(wrappers) + "\n" + "\n".join(probes) + "\n" + new_func_text + suffix
    return injected


def _parse_ptxas_register_report(stderr: str, only_wrappers: bool = True) -> dict[str, int]:
    """
    Parse ptxas output and return {demangled_name: regs} for injected wrappers.
    This relies on ptxas 'Function properties for <mangled>' followed by 'Used N registers'.
    """
    current_fn = None
    fn_to_regs: dict[str, int] = {}
    fn_re = re.compile(r"ptxas info\s*: Function properties for (.+)")
    entry_re = re.compile(r"ptxas info\s*: Compiling entry function '([^']+)'")
    regs_re = re.compile(r"ptxas info\s*: Used\s+(\d+)\s+registers\b")
    for line in stderr.splitlines():
        m = fn_re.search(line)
        if m:
            current_fn = m.group(1).strip().strip("'\"")
            continue
        m = entry_re.search(line)
        if m:
            current_fn = m.group(1).strip().strip("'\"")
            continue
        m = regs_re.search(line)
        if m and current_fn:
            fn_to_regs[current_fn] = int(m.group(1))
            current_fn = None

    if not fn_to_regs:
        return {}

    # Demangle if possible
    try:
        proc = subprocess.run(
            ["c++filt"],
            input="\n".join(fn_to_regs.keys()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        demangled = proc.stdout.splitlines()
        if len(demangled) == len(fn_to_regs):
            mapping = dict(zip(fn_to_regs.keys(), demangled))
        else:
            mapping = {k: k for k in fn_to_regs.keys()}
    except FileNotFoundError:
        mapping = {k: k for k in fn_to_regs.keys()}

    out: dict[str, int] = {}
    for mangled, regs in fn_to_regs.items():
        name = mapping.get(mangled, mangled)
        if only_wrappers:
            # Prefer regprobe kernels since they are entry functions and ptxas
            # reliably prints register usage for them.
            if "__mirage_regprobe_" in name or "__mirage_task_" in name:
                out[name] = regs
        else:
            out[name] = regs
    return out


def _estimate_dynamic_smem_bytes_from_branch(branch_body: str) -> int:
    """
    Best-effort inference of a task's dynamic shared memory usage (bytes) by
    parsing the kernel call in the _execute_task branch and mirroring the
    constexpr offset calculations in the corresponding task implementation.

    Returns:
      - >=0: dynamic shared bytes used (by offsets into extern __shared__)
      - -1: unknown
    """
    call_re = re.compile(r"kernel::([A-Za-z0-9_]+)\s*<([^>]*)>\s*\(")
    m = call_re.search(branch_body)
    if not m:
        return -1
    fn = m.group(1)
    targs = [x.strip() for x in m.group(2).split(",") if x.strip()]

    def type_size(type_tok: str) -> int | None:
        t = type_tok.replace(" ", "")
        if "bfloat16" in t:
            return 2
        if t in ("half", "__half") or "float16" in t:
            return 2
        if "float" == t or "float32" in t:
            return 4
        return None

    ints: list[int] = []
    for tok in targs[1:]:
        if tok.isdigit():
            ints.append(int(tok))
    if not targs:
        return -1
    sz = type_size(targs[0])

    # Kernels with no extern shared
    if fn in (
        "embedding_kernel",
        "silu_mul_task_impl",
        # Hopper variants
        "embedding_kernel_hopper",
        "silu_mul_task_impl_hopper",
    ):
        return 0

    if fn == "linear_kernel":
        # <T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, PIPE_MAX=3>
        if sz is None or len(ints) < 4:
            return -1
        batch, out_size, red_size = ints[0], ints[1], ints[2]
        pipe_max = ints[4] if len(ints) >= 5 else 3
        tile = 128
        forloop = red_size // tile
        adj_pipe = pipe_max if pipe_max < forloop else forloop
        out_atom = out_size if out_size <= 64 else 64
        zero = sz * 64
        shared_input = sz * batch * adj_pipe * tile
        shared_weight = sz * tile * adj_pipe * out_atom
        shared_output = sz * batch * out_atom
        return zero + shared_input + shared_weight + shared_output

    if fn in ("linear_kernel_hopper", "linear_swapAB_kernel_hopper"):
        # Mirror the constexpr shared memory layout in:
        # - include/mirage/persistent_kernel/tasks/hopper/linear_hopper.cuh
        # - include/mirage/persistent_kernel/tasks/hopper/linear_swapAB_hopper.cuh
        #
        # Common template args prefix:
        #   <T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, Kstages, ...>
        if sz is None or len(ints) < 4:
            return -1
        batch, out_size, red_size, kstages = ints[0], ints[1], ints[2], ints[3]

        def align_up(x: int, a: int) -> int:
            return ((x + a - 1) // a) * a

        # TaskRegister uses `void` for the residual TMA type when residual is
        # not present; the kernel checks via std::is_void.
        has_residual = any(tok.replace(" ", "") == "void" for tok in targs)

        # Both Hopper linear kernels are configured with a 128-wide K tile
        # (unless the reduction dim is smaller).
        tile_size = red_size if red_size < 128 else 128

        if fn == "linear_kernel_hopper":
            smem_m_size = batch
            output_atom_size = out_size if out_size <= 256 else 256

            shared_input_off = 0
            shared_weight_off = align_up(
                shared_input_off + sz * kstages * smem_m_size * tile_size, 1024
            )
            shared_residual_off = align_up(
                shared_weight_off + sz * kstages * tile_size * output_atom_size, 1024
            )
            shared_mm_out_off = (
                align_up(
                    shared_residual_off + sz * smem_m_size * output_atom_size * kstages,
                    1024,
                )
                if has_residual
                else shared_residual_off
            )
            shared_input_barrier_off = align_up(
                shared_mm_out_off + sz * smem_m_size * output_atom_size * kstages, 16
            )
            shared_weight_barrier_off = align_up(
                shared_input_barrier_off + 8 * kstages, 8
            )
            shared_residual_barrier_off = align_up(
                shared_weight_barrier_off + 8 * kstages, 8
            )
            shared_compute_done_off = (
                align_up(shared_residual_barrier_off + 8 * kstages, 8)
                if has_residual
                else shared_residual_barrier_off
            )
            shared_residual_done_off = align_up(shared_compute_done_off + 8 * kstages, 8)
            return shared_residual_done_off + 8 * kstages

        # linear_swapAB_kernel_hopper
        smem_m_size = 8 if batch <= 8 else 16
        output_atom_size = 64
        shared_input_off = 0
        shared_weight_off = align_up(
            shared_input_off + sz * kstages * smem_m_size * tile_size, 1024
        )
        shared_residual_off = align_up(
            shared_weight_off + sz * kstages * tile_size * output_atom_size, 1024
        )
        shared_mm_out_off = (
            align_up(
                shared_residual_off + sz * smem_m_size * output_atom_size * kstages,
                1024,
            )
            if has_residual
            else shared_residual_off
        )
        shared_input_barrier_off = align_up(
            shared_mm_out_off + sz * smem_m_size * output_atom_size * kstages, 8
        )
        shared_weight_barrier_off = align_up(shared_input_barrier_off + 8 * kstages, 8)
        shared_residual_barrier_off = align_up(shared_weight_barrier_off + 8 * kstages, 8)
        shared_compute_done_off = (
            align_up(shared_residual_barrier_off + 8 * kstages, 8)
            if has_residual
            else shared_residual_barrier_off
        )
        shared_residual_done_off = align_up(shared_compute_done_off + 8 * kstages, 8)
        return shared_residual_done_off + 8 * kstages

    if fn == "rms_norm_hopper_impl":
        # include/mirage/persistent_kernel/tasks/hopper/rmsnorm_hopper.cuh
        # <T, BATCH_SIZE, HIDDEN_DIM, NUM_THREADS=256>
        if sz is None or len(ints) < 2:
            return -1
        hidden_dim = ints[1]
        num_threads = ints[2] if len(ints) >= 3 else 256
        num_warps = num_threads // 32
        return 3 * sz * hidden_dim + 4 * num_warps

    if fn == "multitoken_paged_attention_hopper_impl":
        # include/mirage/persistent_kernel/tasks/hopper/multitoken_paged_attention_hopper.cuh
        # Template prefix:
        # <T, NUM_QO_HEADS, NUM_KV_HEADS, NUM_QO_GROUPS, KV_CACHE_STRIDE, QKV_STRIDE,
        #  O_STRIDE, HEAD_DIM, SEQ_LEN, MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS=8, ...>
        if sz is None or len(ints) < 10:
            return -1
        num_qo_heads = ints[0]
        num_kv_heads = ints[1]
        head_dim = ints[6]
        max_seq_len = ints[8]
        page_size = ints[9]
        max_tokens = ints[10] if len(ints) >= 11 else 8
        if num_kv_heads == 0 or num_qo_heads % num_kv_heads != 0:
            return -1

        def align_up(x: int, a: int) -> int:
            return ((x + a - 1) // a) * a

        num_qo_per_kv = num_qo_heads // num_kv_heads
        kv_tile = 64
        num_threads = 256  # include/mirage/persistent_kernel/tasks/common/worker_config.h
        kstages = 2
        mma_iters_m = (max_tokens * num_qo_per_kv + 63) // 64

        s_q_off = 0
        s_q_size = sz * max_tokens * num_qo_per_kv * head_dim
        s_k_off = align_up(s_q_off + s_q_size, 1024)
        s_k_size = sz * kv_tile * head_dim
        s_kbuf_off = align_up(s_k_off + s_k_size, 1024)
        s_v_off = align_up(s_kbuf_off + s_k_size, 1024)
        s_vbuf_off = align_up(s_v_off + s_k_size, 1024)

        s_q_norm_off = align_up(s_vbuf_off + s_k_size, 4)
        s_q_norm_size = 4 * 4
        s_k_norm_off = s_q_norm_off + s_q_norm_size
        s_k_norm_size = 4 * 4

        s_m_off = s_k_norm_off + s_k_norm_size
        s_m_size = 4 * mma_iters_m * num_threads * 2
        s_d_off = s_m_off + s_m_size
        s_d_size = 4 * mma_iters_m * num_threads * 2
        s_o_buf_off = s_d_off + s_d_size
        s_o_buf_size = 4 * mma_iters_m * num_threads * 64

        s_q_barrier_off = align_up(s_o_buf_off + s_o_buf_size, 8)
        s_q_barrier_size = 8 * kstages
        s_k_barrier_off = align_up(s_q_barrier_off + s_q_barrier_size, 8)
        s_k_barrier_size = 8 * kstages
        s_v_barrier_off = align_up(s_k_barrier_off + s_k_barrier_size, 8)
        s_v_barrier_size = 8 * kstages
        s_compute_done_off = align_up(s_v_barrier_off + s_v_barrier_size, 8)
        s_compute_done_size = 8 * kstages

        # Total excludes `S_O_OFFSET` since it reuses memory.
        return s_compute_done_off + s_compute_done_size

    if fn == "rms_norm_impl":
        # <T, BATCH_SIZE, HIDDEN_DIM>, BATCH_SIZE is asserted == 1 in code.
        if sz is None or len(ints) < 2:
            return -1
        hidden = ints[1]
        num_warps = 4  # NUM_THREADS=128
        return 3 * sz * hidden + 4 * num_warps

    if fn in (
        "argmax_partial_kernel",
        "argmax_reduce_kernel",
        "argmax_partial_sm100_kernel",
        "argmax_reduce_sm100_kernel",
    ):
        # Ampere: tasks/ampere/argmax.cuh
        # Blackwell: tasks/blackwell/argmax_sm100.cuh
        #
        # Both use a block reduce that stores one (val, idx) pair per warp in
        # dynamic shared memory with the same high-level layout:
        #   extern __shared__ char smem[];
        #   smem_idxs = align_up(smem, 128);
        #   smem_vals = reinterpret_cast<T*>(smem_idxs + 32);
        #
        # So the dynamic shared footprint is:
        #   max_align_slack(127B) + 32 * (8B idx + sizeof(T) val)
        if sz is None:
            return -1
        max_warps = 32
        align_slack = 127
        return align_slack + max_warps * (8 + sz)

    if fn == "multitoken_paged_attention_task_impl":
        # <T, NUM_QO_HEADS, NUM_KV_HEADS, KV_CACHE_STRIDE, QKV_STRIDE, O_STRIDE,
        #  HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE, MAX_TOKENS=8>
        if sz is None or len(ints) < 8:
            return -1
        num_qo, num_kv = ints[0], ints[1]
        head_dim = ints[5]
        max_seq_len = ints[6]
        page_size = ints[7]
        max_tokens = ints[8] if len(ints) >= 9 else 8
        if num_kv == 0 or num_qo % num_kv != 0:
            return -1
        num_qo_per_kv = num_qo // num_kv
        num_q = max_tokens * num_qo_per_kv

        def align4(x: int) -> int:
            return (x + 3) & ~3

        zero = sz * 8
        s_q_size = sz * max_tokens * num_qo_per_kv * head_dim
        if (max_tokens * num_qo) <= 16:
            kv_tile = 64
            s_k_size = sz * kv_tile * head_dim
            s_q_off = zero
            s_k_off = s_q_off + s_q_size
            s_kbuf_off = s_k_off + s_k_size
            s_v_off = s_kbuf_off + s_k_size
            s_vbuf_off = s_v_off + s_k_size
            s_q_norm_off = align4(s_vbuf_off + s_k_size)
            s_q_norm_size = 4 * 4
            s_k_norm_off = s_q_norm_off + s_q_norm_size
            s_k_norm_size = 4 * 4
            mma_iters_m = (num_q + 15) // 16
            s_m_size = 4 * mma_iters_m * 128 * 2
            s_d_size = 4 * mma_iters_m * 128 * 2
            s_o_buf_size = 4 * mma_iters_m * 128 * 64
            s_m_off = zero
            s_d_off = s_m_off + s_m_size
            s_o_buf_off = s_d_off + s_d_size
            s_o_off = s_o_buf_off + s_o_buf_size
            s_o_size = s_q_size
            total1 = s_o_off + s_o_size
            total2 = s_k_norm_off + s_k_norm_size
            return total1 if total1 > total2 else total2
        elif (max_tokens * num_qo) <= 64:
            kv_tile = 128
            s_k_size = sz * kv_tile * head_dim
            s_q_off = zero
            s_k_off = s_q_off + s_q_size
            s_kbuf_off = s_k_off + s_k_size
            s_v_off = s_kbuf_off + s_k_size
            s_vbuf_off = s_v_off + s_k_size
            s_q_norm_off = align4(s_vbuf_off + s_k_size)
            s_q_norm_size = 4 * 4
            s_k_norm_off = s_q_norm_off + s_q_norm_size
            s_k_norm_size = 4 * 4
            global_iters_m = (num_q + 64 - 1) // 64
            s_m_size = 4 * global_iters_m * 128 * 2
            s_d_size = 4 * global_iters_m * 128 * 2
            s_o_buf_size = 4 * global_iters_m * 128 * 64
            s_m_off = zero
            s_d_off = s_m_off + s_m_size
            s_o_buf_off = s_d_off + s_d_size
            total1 = s_o_buf_off + s_o_buf_size
            total2 = s_k_norm_off + s_k_norm_size
            return total1 if total1 > total2 else total2
        return -1

    return -1


def _dump_static_smem_bytes_for_functions(so_path: str, nvcc_path: str) -> dict[str, int]:
    """
    Use cuobjdump --dump-resource-usage to fetch per-entry-function static shared memory usage.
    Returns {function_name: static_shared_bytes}.
    """
    if not so_path or not os.path.exists(so_path):
        return {}
    cuobjdump = None
    if nvcc_path:
        candidate = os.path.join(os.path.dirname(nvcc_path), "cuobjdump")
        if os.path.exists(candidate):
            cuobjdump = candidate
    if cuobjdump is None:
        cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {}

    try:
        out = subprocess.check_output(
            [cuobjdump, "--dump-resource-usage", so_path],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return {}

    # Parse blocks like:
    #  Function __mirage_regprobe_TASK_X_v0:
    #   REG:.. STACK:.. SHARED:0 LOCAL:0 ...
    fn_re = re.compile(r"^\s*Function\s+([^\s:]+)\s*:\s*$")
    shared_re = re.compile(r"SHARED:(\d+)")
    current = None
    fn_to_shared: dict[str, int] = {}
    for line in out.splitlines():
        m = fn_re.match(line)
        if m:
            current = m.group(1)
            continue
        if current:
            m = shared_re.search(line)
            if m:
                fn_to_shared[current] = int(m.group(1))
                current = None

    return fn_to_shared
