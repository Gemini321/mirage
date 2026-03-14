from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
import copy
import hashlib
import itertools
import subprocess
import shutil
import sysconfig
import sys
import time
import fcntl
import ast
import ctypes
import ctypes.util
import math
import re
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 100
IGNORE_EOS_SENTINEL = -(1 << 60)

os.environ["HF_HUB_OFFLINE"] = "1"

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 64 == 0 and not use_cutlass_kernel:
        # TODO(Wenqin): If we set OUTPUT_SIZE too much for PTX linear kernel,
        # there is some regression.
        return size // 64
    if size / 96 > 400:
        # TODO: An add-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpect result (not same out put for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, "FATAL: Linear layer size not support, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def safe_decode_tokens(tokenizer, token_tensor):
    token_ids = token_tensor.tolist()
    vocab_size = tokenizer.vocab_size if tokenizer.vocab_size is not None else (1 << 31)
    if tokenizer.unk_token_id is not None:
        safe_ids = [tid if 0 <= tid < vocab_size else tokenizer.unk_token_id for tid in token_ids]
    else:
        safe_ids = [tid if 0 <= tid < vocab_size else 0 for tid in token_ids]
    try:
        return tokenizer.decode(safe_ids, skip_special_tokens=True)
    except Exception:
        return tokenizer.decode(safe_ids, skip_special_tokens=False)


def print_mirage_task_graph_summary(results_json,
                                    model,
                                    args,
                                    mpk,
                                    layer_summaries,
                                    num_local_q_heads,
                                    num_local_kv_heads,
                                    head_dim):
    from mirage.profiler_persistent import event_name_list

    graph = json.loads(results_json)
    task_counts = {}
    for task in graph.get("all_tasks", []):
        task_type = int(task.get("task_type", -1))
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    top = sorted(task_counts.items(), key=lambda x: (-x[1], x[0]))[:20]

    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    max_tokens = mpk.max_num_batched_tokens
    attention_mode = "split_kv" if args.split_kv_cache else "paged"

    print("=== Mirage Task Graph Summary ===")
    print(f"layers={model.config.num_hidden_layers} "
          f"max_tokens={max_tokens} hidden={hidden} "
          f"intermediate={intermediate} "
          f"q_heads(local)={num_local_q_heads} "
          f"kv_heads(local)={num_local_kv_heads} "
          f"head_dim={head_dim}")
    print(f"attention_mode={attention_mode}")
    print("task_type counts (top 20):")
    for task_type, count in top:
        name = event_name_list.get(task_type, f"TASK_{task_type}")
        print(f"  {name}: {count}")

    for layer in layer_summaries:
        idx = layer["index"]
        if idx > 0:
            break
        print(f"\n[layer {idx}]")
        print(f"  input: ({layer['batch']}, {hidden})")
        print(f"  qkv:   ({layer['batch']}, {layer['qkv_n']})")
        print(f"  attn:  ({layer['batch']}, {layer['attn_n']})")
        print(f"  mlp:   gatedup=({layer['batch']}, {layer['gatedup_n']}) "
              f"silu_out=({layer['batch']}, {layer['silu_n']})")

        print(f"  qkv_proj: M={layer['batch']} K={hidden} N={layer['qkv_n']} "
              f"({layer['batch']}x{hidden} @ {hidden}x{layer['qkv_n']} -> "
              f"{layer['batch']}x{layer['qkv_n']}) "
              f"tasks={layer['qkv_tasks']} (split_N = {layer['qkv_split']})")
        print(f"    tasks: M={layer['batch_tile']} K={hidden} N={layer['qkv_tile_n']} "
              f"({layer['batch_tile']}x{hidden} @ {hidden}x{layer['qkv_tile_n']} -> "
              f"{layer['batch_tile']}x{layer['qkv_tile_n']}) "
              f"x{layer['qkv_tasks']}")

        print(f"  attention: tasks={layer['attn_tasks']} "
              f"tiles={{({layer['batch']}, {layer['attn_tile_n']}): "
              f"{layer['attn_tasks']}}}")
        print(f"    tasks: out=({layer['batch']}, {layer['attn_tile_n']}) "
              f"x{layer['attn_tasks']}")

        print(f"  o_proj(+res): M={layer['batch']} K={layer['attn_n']} "
              f"N={hidden} ({layer['batch']}x{layer['attn_n']} @ "
              f"{layer['attn_n']}x{hidden} -> {layer['batch']}x{hidden}) "
              f"tasks={layer['o_tasks']} (split_N = {layer['o_split']})")
        print(f"    tasks: M={layer['batch_tile']} K={layer['attn_n']} "
              f"N={layer['o_tile_n']} ({layer['batch_tile']}x{layer['attn_n']} @ "
              f"{layer['attn_n']}x{layer['o_tile_n']} -> "
              f"{layer['batch_tile']}x{layer['o_tile_n']}) "
              f"x{layer['o_tasks']}")

        print(f"  post_attn_rmsnorm: out=(1, {hidden}) tasks={layer['rms_tasks']}")
        print(f"    tasks: out=(1, {hidden}) x{layer['rms_tasks']}")

        print(f"  gatedup_proj: M={layer['batch']} K={hidden} "
              f"N={layer['gatedup_n']} "
              f"({layer['batch']}x{hidden} @ {hidden}x{layer['gatedup_n']} -> "
              f"{layer['batch']}x{layer['gatedup_n']}) "
              f"tasks={layer['gatedup_tasks']} (split_N = {layer['gatedup_split']})")
        print(f"    tasks: M={layer['batch_tile']} K={hidden} "
              f"N={layer['gatedup_tile_n']} "
              f"({layer['batch_tile']}x{hidden} @ {hidden}x{layer['gatedup_tile_n']} -> "
              f"{layer['batch_tile']}x{layer['gatedup_tile_n']}) "
              f"x{layer['gatedup_tasks']}")

        print(f"  silu_mul: out=({layer['batch']}, {layer['silu_n']}) "
              f"tasks={layer['silu_tasks']}")
        print(f"    tasks: out=({layer['batch']}, {layer['silu_tile_n']}) "
              f"x{layer['silu_tasks']}")

        print(f"  down_proj(+res): M={layer['batch']} K={layer['silu_n']} "
              f"N={hidden} ({layer['batch']}x{layer['silu_n']} @ "
              f"{layer['silu_n']}x{hidden} -> {layer['batch']}x{hidden}) "
              f"tasks={layer['down_tasks']} (split_N = {layer['down_split']})")
        print(f"    tasks: M={layer['batch_tile']} K={layer['silu_n']} "
              f"N={layer['down_tile_n']} ({layer['batch_tile']}x{layer['silu_n']} @ "
              f"{layer['silu_n']}x{layer['down_tile_n']} -> "
              f"{layer['batch_tile']}x{layer['down_tile_n']}) "
              f"x{layer['down_tasks']}")

        print(f"  input_rmsnorm: out=(1, {hidden}) tasks={layer['rms_tasks']}")
        print(f"    tasks: out=(1, {hidden}) x{layer['rms_tasks']}")


def model_fingerprint(model, args, world_size, batch_size):
    payload = {
        "model": args.model if args.model_path is None else args.model_path,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "vocab_size": model.config.vocab_size,
        "world_size": world_size,
        "batch_size": batch_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_batched_requests": args.max_num_batched_requests,
        "max_seq_length": args.max_seq_length,
        "max_num_pages": args.max_num_pages,
        "page_size": args.page_size,
        "split_kv_cache": args.split_kv_cache,
        "use_cutlass_kernel": args.use_cutlass_kernel,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _to_list(v):
    return v if isinstance(v, list) else [v]


_CUDART_HANDLE = None


def _load_cudart():
    global _CUDART_HANDLE
    if _CUDART_HANDLE is not None:
        return _CUDART_HANDLE
    candidates = []
    name = ctypes.util.find_library("cudart")
    if name:
        candidates.append(name)
    candidates.extend(["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"])
    last_err = None
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
            lib.cudaGetLastError.restype = ctypes.c_int
            lib.cudaGetErrorString.argtypes = [ctypes.c_int]
            lib.cudaGetErrorString.restype = ctypes.c_char_p
            _CUDART_HANDLE = lib
            return _CUDART_HANDLE
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load cudart for cudaGetLastError: {last_err}")


def check_cuda_last_error_or_raise(context: str):
    cudart = _load_cudart()
    err = int(cudart.cudaGetLastError())
    if err != 0:
        try:
            err_str = cudart.cudaGetErrorString(err)
            if isinstance(err_str, bytes):
                err_str = err_str.decode("utf-8", errors="ignore")
        except Exception:
            err_str = "unknown CUDA error"
        raise RuntimeError(f"[{context}] cudaGetLastError={err} ({err_str})")


def _acquire_run_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_run_lock(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def build_synthetic_qwen3_worker_model(config, world_size: int, max_num_pages: int, page_size: int):
    # Worker-only synthetic tensors: preserve shape/dtype without loading model weights.
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    num_local_q_heads = num_q_heads // world_size
    num_local_kv_heads = num_kv_heads // world_size
    inter_local = intermediate_size // world_size

    def t(shape):
        return torch.empty(shape, dtype=torch.bfloat16, device="cuda")

    layer = SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=t((hidden_size,))),
        post_attention_layernorm=SimpleNamespace(weight=t((hidden_size,))),
        self_attn=SimpleNamespace(
            q_proj=SimpleNamespace(weight=t((num_local_q_heads * head_dim, hidden_size))),
            k_proj=SimpleNamespace(weight=t((num_local_kv_heads * head_dim, hidden_size))),
            v_proj=SimpleNamespace(weight=t((num_local_kv_heads * head_dim, hidden_size))),
            o_proj=SimpleNamespace(weight=t((hidden_size, num_local_q_heads * head_dim))),
            q_norm=SimpleNamespace(weight=t((head_dim,))),
            k_norm=SimpleNamespace(weight=t((head_dim,))),
        ),
        mlp=SimpleNamespace(
            gate_proj=SimpleNamespace(weight=t((inter_local, hidden_size))),
            up_proj=SimpleNamespace(weight=t((inter_local, hidden_size))),
            down_proj=SimpleNamespace(weight=t((hidden_size, inter_local))),
        ),
    )

    kv_shape = (1, max_num_pages, page_size, num_local_kv_heads, head_dim)
    model = SimpleNamespace(
        config=config,
        lm_head=SimpleNamespace(weight=t((int(config.vocab_size), hidden_size))),
        model=SimpleNamespace(
            layers=[layer],
            kv_cache=(
                torch.empty(kv_shape, dtype=torch.bfloat16, device="cuda"),
                torch.empty(kv_shape, dtype=torch.bfloat16, device="cuda"),
            ),
            norm=SimpleNamespace(weight=t((hidden_size,))),
            embed_tokens=SimpleNamespace(weight=t((int(config.vocab_size), hidden_size))),
        ),
    )
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=8, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-0.6B', help="Model path on hugging face"
    )
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=False,
        help="Not use the cutlass version kernel.",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")

    # -------- Args for CI tests ----------
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling (default off)")
    parser.add_argument(
        "--save-tokens",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Optionally dump first N generated token_ids, text, and latency to JSON. "
            "If path omitted, saves to outputs/qwen3/{torch_output.json|mpk_output.json}."
        ),
    )
    parser.add_argument("--prompt",
        type=str,
        default="Give me an introduction to large language model. Say as much as you can.",
        help="Custom prompt text to generate from.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="Force prompt token length by truncating/padding to this exact length.",
    )

    parser.add_argument("--split-kv-cache", action="store_true", help="Use split-kv cache")
    parser.add_argument("--autotune", action="store_true", help="Enable MPK autotuning")
    parser.add_argument(
        "--autotune-list",
        type=str,
        default=os.path.join(DEFAULT_SAVE_DIR, "autotune_space.json"),
        help="JSON file with candidates",
    )
    parser.add_argument("--autotune-cache", type=str, default=os.path.join(DEFAULT_SAVE_DIR, "autotune_cache.json"))
    parser.add_argument("--autotune-force", action="store_true", help="Ignore cached tuning results")
    parser.add_argument(
        "--autotune-reference-logs",
        type=str,
        default="",
        help=(
            "Comma-separated autotune log paths used as guidance priors. "
            "Each log can provide [autotune] ... sig=... cfg=... entries."
        ),
    )
    parser.add_argument(
        "--autotune-reference-force",
        action="store_true",
        help="Force selecting reference-log candidate per signature when available.",
    )
    parser.add_argument(
        "--autotune-reference-strict-signatures",
        action="store_true",
        help=(
            "When reference logs are provided, only autotune signatures that appear in logs. "
            "Missing signatures keep default behavior/config."
        ),
    )
    parser.add_argument("--autotune-compile-jobs", type=int, default=max((os.cpu_count() or 1) // 2, 1))
    parser.add_argument("--autotune-profile-gpus", type=str, default="0", help="Comma-separated GPU ids for candidate profiling")
    parser.add_argument("--autotune-profile-jobs", type=int, default=2, help="Max parallel candidate profiling jobs")
    parser.add_argument("--autotune-workers-per-gpu", type=int, default=2, help="Max concurrent autotune workers per GPU")
    parser.add_argument(
        "--autotune-kernel-repeat",
        type=int,
        default=1000,
        help="Repeat the profiled kernel N times serially in worker runtime for stable timing.",
    )
    parser.add_argument(
        "--autotune-debug-sanitizer",
        action="store_true",
        help="On autotune candidate failure, run one compute-sanitizer memcheck per failed candidate (debug only).",
    )
    parser.add_argument(
        "--autotune-debug-sanitizer-limit",
        type=int,
        default=4,
        help="Maximum number of failed candidates to run through compute-sanitizer in one autotune run.",
    )
    parser.add_argument(
        "--autotune-debug-sanitizer-bin",
        type=str,
        default="compute-sanitizer",
        help="Path to compute-sanitizer binary for debug mode.",
    )
    parser.add_argument(
        "--no-autotune-worker-synthetic",
        action="store_false",
        dest="autotune_worker_synthetic",
        default=True,
        help="Disable synthetic tensor mode in autotune worker.",
    )
    parser.add_argument("--autotune-worker-task-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--autotune-worker-result-json", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prompt_tokens is not None and args.prompt_tokens <= 0:
        raise ValueError("--prompt-tokens must be > 0")
    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be >= 0")
    if args.autotune_kernel_repeat <= 0:
        raise ValueError("--autotune-kernel-repeat must be >= 1")
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    worker_mode = args.autotune_worker_task_json is not None
    synthetic_worker_mode = worker_mode and args.autotune_worker_synthetic
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    if synthetic_worker_mode:
        cfg_src = args.model_path if args.model_path is not None else model_name
        config = AutoConfig.from_pretrained(cfg_src)
        model = build_synthetic_qwen3_worker_model(
            config=config,
            world_size=world_size,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
        )
        tokenizer = None
    else:
        with torch.device("cuda"):
            if args.model_path is not None:
                # load model locally (necessary for multi-GPU case)
                print(f"Load model from model path: {args.model_path}")
                config = AutoConfig.from_pretrained(args.model_path)
                model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
                # load_model(
                #     model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
                # )
                model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            else:
                model = Qwen3ForCausalLM.from_pretrained(model_name, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests

    if synthetic_worker_mode:
        prompt_len = max(1, int(args.prompt_tokens) if args.prompt_tokens is not None else 1)
        if args.max_new_tokens is not None:
            args.max_seq_length = max(args.max_seq_length, int(prompt_len) + int(args.max_new_tokens))
        tokens = torch.zeros((total_num_requests, args.max_seq_length), dtype=torch.long, device="cuda")
        prompt_lengths = torch.full((total_num_requests,), prompt_len, dtype=torch.int, device="cuda")
        head_dim = int(model.config.head_dim)
        cos = torch.zeros((1, 32768, head_dim), dtype=torch.bfloat16, device="cuda")
        sin = torch.zeros((1, 32768, head_dim), dtype=torch.bfloat16, device="cuda")
        position_embeddings = (cos, sin)
    else:
        prompt = args.prompt
        # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
        code_text = """import numpy as np
                    import matplotlib.pyplot as plt

                    # Calculate the average
                    average_throughput = np.mean(tokens_per_sec_arr)
                    print(f"Average Throughput: {average_throughput} tokens/sec")

                    # Plotting the histogram
                    plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                    plt.title('Histogram of Throughput Values')
                    plt.xlabel('Tokens per Second')
                    plt.ylabel('Frequency')
                    plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                    plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                    plt.show()
                    """
        #question = "Can you please change x axis to start from 0"
        #prompt = code_text + "\n" + question
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.model.embed_tokens.weight.device)
        if args.prompt_tokens is not None:
            input_ids = model_inputs.input_ids
            target_len = max(1, int(args.prompt_tokens))
            cur_len = int(input_ids.shape[-1])
            if cur_len > target_len:
                model_inputs.input_ids = input_ids[:, :target_len]
            elif cur_len < target_len:
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                pad_cols = target_len - cur_len
                pad_ids = torch.full(
                    (input_ids.shape[0], pad_cols),
                    int(pad_token_id),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                model_inputs.input_ids = torch.cat([input_ids, pad_ids], dim=-1)
        prompt_len = model_inputs.input_ids.shape[-1]
        if args.max_new_tokens is not None:
            args.max_seq_length = max(args.max_seq_length, int(prompt_len) + int(args.max_new_tokens))
        tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")
        for r in range(total_num_requests):
            for i in range(model_inputs.input_ids.shape[-1]):
                tokens[r, i] = model_inputs.input_ids[0, i]
        prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")
        positions = torch.arange(32768).unsqueeze(0).to(model.model.embed_tokens.weight.device)
        position_embeddings = model.model.rotary_emb(positions)

    # get all model weight tensors
    if args.max_num_batched_tokens >= 16:
        assert args.max_num_batched_tokens % 16 == 0, "max_num_batched_tokens should be multiple of 16"
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    decode_tokens = args.max_new_tokens if args.max_new_tokens is not None else (args.max_seq_length - prompt_lengths[0].item())
    decode_tokens = max(1, int(decode_tokens))
    num_new_tokens = torch.full((total_num_requests, ), decode_tokens, dtype=torch.int32, device="cuda")

    if args.use_mirage:
        import mirage as mi

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        lm_head_weight = torch.cat(
            (
                model.lm_head.weight,
                torch.full(
                    (153600 - model.config.vocab_size, hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )
        vocab_size = 153600
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_local_q_heads = num_q_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size
        num_kv_cache_chunks = max(1, args.max_seq_length // 256)
        spec_decode_config = mi.speculative.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
        profiler_tensor = (
            torch.zeros(4000 * 128, dtype=torch.uint64, device="cuda").contiguous()
            if args.profiling
            else None
        )

        def _ceil_div(x, y):
            return (x + y - 1) // y

        def _target_cfg(cfg, key, call_sig=None):
            if cfg is None:
                return {}
            entry = cfg.get(key, {})
            if not isinstance(entry, dict):
                return {}
            by_sig = entry.get("by_signature")
            if call_sig is not None and isinstance(by_sig, dict):
                cand = by_sig.get(call_sig)
                if isinstance(cand, dict):
                    return cand
            if "default" in entry and isinstance(entry["default"], dict):
                return entry["default"]
            return entry

        def _linear_grid(cfg, key, output_size, default, swap_xy=False, call_sig=None):
            tcfg = _target_cfg(cfg, key, call_sig=call_sig)
            if "grid_dim" in tcfg:
                return tuple(tcfg["grid_dim"])
            task_output_size = tcfg.get("task_output_size")
            task_batch_size = int(args.max_num_batched_tokens)
            if task_batch_size > 16:
                raise ValueError(
                    f"{key} requires task_batch_size=max_num_batched_tokens={task_batch_size} <= 16"
                )
            gx, gy, gz = default
            if task_output_size is not None:
                if task_output_size <= 0:
                    raise ValueError(f"{key}.task_output_size must be > 0")
                output_tasks = max(1, _ceil_div(output_size, int(task_output_size)))
                if swap_xy:
                    gy = output_tasks
                else:
                    gx = output_tasks
            batch_tasks = max(1, _ceil_div(args.max_num_batched_tokens, task_batch_size))
            if swap_xy:
                gx = batch_tasks
            else:
                gy = batch_tasks
            return (gx, gy, gz)

        def _paged_attention_grid(cfg, key, default, call_sig=None):
            tcfg = _target_cfg(cfg, key, call_sig=call_sig)
            if "grid_dim" in tcfg:
                return tuple(tcfg["grid_dim"])
            task_num_requests = tcfg.get("task_num_requests")
            task_num_kv_heads = tcfg.get("task_num_kv_heads")
            gx, gy, gz = default
            if task_num_requests is not None:
                if task_num_requests <= 0:
                    raise ValueError(f"{key}.task_num_requests must be > 0")
                gx = max(1, _ceil_div(args.max_num_batched_requests, int(task_num_requests)))
            if task_num_kv_heads is not None:
                if task_num_kv_heads <= 0:
                    raise ValueError(f"{key}.task_num_kv_heads must be > 0")
                task_num_kv_heads = int(task_num_kv_heads)
                if num_local_kv_heads % task_num_kv_heads != 0:
                    raise ValueError(
                        f"{key}.task_num_kv_heads={task_num_kv_heads} must divide "
                        f"num_local_kv_heads={num_local_kv_heads}"
                    )
                gy = max(1, num_local_kv_heads // task_num_kv_heads)
            return (gx, gy, gz)

        def _pipe(cfg, key, call_sig=None):
            tcfg = _target_cfg(cfg, key, call_sig=call_sig)
            return tcfg.get("pipe_stage")

        def _max_tokens(cfg, key, call_sig=None):
            tcfg = _target_cfg(cfg, key, call_sig=call_sig)
            v = tcfg.get("max_tokens")
            if v is None:
                v = tcfg.get("MAX_TOKENS")
            return v

        sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
        max_threadblocks_per_kernel = 4096
        target_cc = (
            torch.cuda.get_device_properties(rank).major * 10
            + torch.cuda.get_device_properties(rank).minor
        )
        max_dynamic_smem = mi.get_shared_memory_capacity(target_cc)

        def _align_to(x, a):
            return ((x + a - 1) // a) * a

        def _calc_linear_smem_bytes(task_batch_size, task_output_size, reduction_size, pipe_stage):
            # Mirrors include/mirage/persistent_kernel/tasks/ampere/linear.cuh
            if reduction_size % 128 != 0:
                return None
            forloop_range = reduction_size // 128
            if forloop_range <= 0:
                return None
            adjusted_pipe = min(int(pipe_stage), int(forloop_range))
            if adjusted_pipe <= 0:
                return None
            output_atom = min(int(task_output_size), 64)
            elem_size = 2  # bfloat16
            zero_buf = elem_size * 64
            shared_input = elem_size * int(task_batch_size) * adjusted_pipe * 128
            shared_weight = elem_size * 128 * adjusted_pipe * output_atom
            shared_output = elem_size * int(task_batch_size) * output_atom
            smem_raw = zero_buf + shared_input + shared_weight + shared_output
            return _align_to(smem_raw, 16)

        def _linear_divisibility_checks(task_batch_size, task_output_size, output_sizes):
            if task_batch_size <= 0 or task_output_size <= 0:
                return False, "task_batch_size/task_output_size must be > 0"
            if args.max_num_batched_tokens % task_batch_size != 0:
                return False, (
                    f"max_num_batched_tokens={args.max_num_batched_tokens} not divisible by "
                    f"task_batch_size={task_batch_size}"
                )
            for out_n in output_sizes:
                if out_n % task_output_size != 0:
                    return False, f"output_size={out_n} not divisible by task_output_size={task_output_size}"
            return True, ""

        def validate_autotune_candidate(target, cand, case=None):
            # Legacy explicit grid mode is still accepted without extra checks.
            if "grid_dim" in cand:
                return True, ""

            if target in ("linear_layer", "linear_with_residual_layer"):
                if "task_output_size" not in cand:
                    return False, "missing task_output_size"
                task_batch_size = int(args.max_num_batched_tokens)
                task_output_size = int(cand["task_output_size"])
                pipe_stage = int(cand.get("pipe_stage", 3))
                if task_batch_size > 16:
                    return (
                        False,
                        "linear kernels require task_batch_size=max_num_batched_tokens <= 16 "
                        "(NUM_ITERS_M == 1)",
                    )

                # Per-case validation: use case-specific sizes if available
                if case and "output_size" in case:
                    output_sizes = [case["output_size"]]
                    reduction_sizes = [case["reduction_size"]]
                elif target == "linear_layer":
                    output_sizes = [
                        fused_outdim_1 // world_size,
                        fused_outdim_2 // world_size,
                        vocab_size,
                    ]
                    reduction_sizes = [hidden_size, hidden_size, hidden_size]
                else:
                    output_sizes = [hidden_size]
                    reduction_sizes = [
                        num_local_q_heads * head_dim,
                        intermediate_size // world_size,
                    ]

                ok, reason = _linear_divisibility_checks(
                    task_batch_size, task_output_size, output_sizes
                )
                if not ok:
                    return False, reason

                for out_n in output_sizes:
                    gx = out_n // task_output_size
                    gy = args.max_num_batched_tokens // task_batch_size
                    if gx * gy > max_threadblocks_per_kernel:
                        return (
                            False,
                            f"linear grid_dim=({gx},{gy},1) exceeds max threadblocks "
                            f"{max_threadblocks_per_kernel}",
                        )

                for red in reduction_sizes:
                    smem_per_task = _calc_linear_smem_bytes(
                        task_batch_size, task_output_size, red, pipe_stage
                    )
                    if smem_per_task is None:
                        return False, f"invalid reduction_size={red} for TILE_SIZE=128"
                    if smem_per_task > max_dynamic_smem:
                        return (
                            False,
                            f"linear smem_per_task={smem_per_task} exceeds max_dynamic_smem={max_dynamic_smem}",
                        )
                    # linear.cuh has compile-time static_assert:
                    # 2 * SMEM_PER_GROUP <= MAX_DYNAMIC_SHARED_MEMORY_SIZE.
                    # Keep this as a hard constraint to avoid compile failures.
                    if smem_per_task * 2 > max_dynamic_smem:
                        return (
                            False,
                            f"linear smem_per_task={smem_per_task} violates compile-time smem constraint "
                            f"(2*smem>{max_dynamic_smem})",
                        )
                return True, ""

            if target == "paged_attention_layer":
                max_tokens = cand.get("max_tokens", cand.get("MAX_TOKENS"))
                if max_tokens is None:
                    max_tokens = args.max_num_batched_tokens
                max_tokens = int(max_tokens)
                if max_tokens != args.max_num_batched_tokens:
                    return (
                        False,
                        f"MAX_TOKENS must equal max_num_batched_tokens={args.max_num_batched_tokens}",
                    )
                if max_tokens <= 0:
                    return False, "MAX_TOKENS must be > 0"
                if max_tokens > args.max_num_batched_tokens:
                    return (
                        False,
                        f"MAX_TOKENS={max_tokens} exceeds max_num_batched_tokens={args.max_num_batched_tokens}",
                    )
                if args.page_size % 64 != 0:
                    return False, f"page_size={args.page_size} must be divisible by 64"

                task_num_requests = int(cand.get("task_num_requests", 1))
                task_num_kv_heads = int(cand.get("task_num_kv_heads", 1))
                if task_num_requests <= 0 or task_num_kv_heads <= 0:
                    return False, "task_num_requests/task_num_kv_heads must be > 0"
                if args.max_num_batched_requests % task_num_requests != 0:
                    return (
                        False,
                        f"max_num_batched_requests={args.max_num_batched_requests} not divisible by "
                        f"task_num_requests={task_num_requests}",
                    )
                if num_local_kv_heads % task_num_kv_heads != 0:
                    return (
                        False,
                        f"num_local_kv_heads={num_local_kv_heads} not divisible by "
                        f"task_num_kv_heads={task_num_kv_heads}",
                    )
                if num_local_q_heads % num_local_kv_heads != 0:
                    return False, (
                        f"num_local_q_heads={num_local_q_heads} not divisible by "
                        f"num_local_kv_heads={num_local_kv_heads}"
                    )
                gx = args.max_num_batched_requests // task_num_requests
                gy = num_local_kv_heads // task_num_kv_heads
                if gx * gy > max_threadblocks_per_kernel:
                    return (
                        False,
                        f"attention grid_dim=({gx},{gy},1) exceeds max threadblocks "
                        f"{max_threadblocks_per_kernel}",
                    )
                num_qo_per_kv = num_local_q_heads // num_local_kv_heads
                num_q_heads_per_task = num_qo_per_kv * task_num_kv_heads
                block_tokens = min(max_tokens, 16)
                if target_cc == 80 and block_tokens * num_q_heads_per_task > 64:
                    return (
                        False,
                        f"ampere attention requires min(MAX_TOKENS,16)*num_q_heads_per_task<=64, got "
                        f"{block_tokens}*{num_q_heads_per_task}",
                    )
                return True, ""

            return True, ""

        def _divisors(n):
            out = []
            i = 1
            while i * i <= n:
                if n % i == 0:
                    out.append(i)
                    if i * i != n:
                        out.append(n // i)
                i += 1
            return sorted(out)

        def build_pruned_candidates(target, entry, case=None):
            if isinstance(entry, list):
                pre = len(entry)
                filtered = []
                for cand in entry:
                    ok, _ = validate_autotune_candidate(target, cand, case=case)
                    if ok:
                        filtered.append(cand)
                kind = case["kind"] if case else ""
                print(f"[autotune][prune] target={target} kind={kind} mode=list kept={len(filtered)}/{pre}")
                return filtered

            if not isinstance(entry, dict):
                raise ValueError(f"autotune target '{target}' must be list or dict")

            values = {k: _to_list(v) for k, v in entry.items()}

            if target in ("linear_layer", "linear_with_residual_layer"):
                # Linear kernels bind per-task batch to the runtime batch size;
                # this is not an autotune dimension.
                if "task_batch_size" in values:
                    values.pop("task_batch_size", None)
                # Per-case pruning: use case-specific output_size if available
                if case and "output_size" in case:
                    out_sizes = [case["output_size"]]
                    red_sizes = [case["reduction_size"]]
                else:
                    if target == "linear_layer":
                        out_sizes = [fused_outdim_1 // world_size, fused_outdim_2 // world_size, vocab_size]
                        red_sizes = [hidden_size, hidden_size, hidden_size]
                    else:
                        out_sizes = [hidden_size]
                        red_sizes = [num_local_q_heads * head_dim, intermediate_size // world_size]
                common_div = set(_divisors(out_sizes[0]))
                for n in out_sizes[1:]:
                    common_div &= set(_divisors(n))
                if "task_output_size" in values:
                    values["task_output_size"] = [
                        int(x) for x in values["task_output_size"]
                        if int(x) > 0 and int(x) in common_div
                    ]
                if "pipe_stage" in values:
                    max_effective_pipe = max(max(1, r // 128) for r in red_sizes if r % 128 == 0)
                    # pipe_stage beyond max_effective_pipe is equivalent after ADJUSTED_PIPE_MAX.
                    values["pipe_stage"] = sorted({
                        min(int(x), max_effective_pipe)
                        for x in values["pipe_stage"]
                        if int(x) > 0
                    })

            elif target == "paged_attention_layer":
                if "max_tokens" in values:
                    values["max_tokens"] = [int(args.max_num_batched_tokens)]
                if "MAX_TOKENS" in values:
                    values["MAX_TOKENS"] = [int(args.max_num_batched_tokens)]
                valid_req = set(_divisors(args.max_num_batched_requests))
                if "task_num_requests" in values:
                    values["task_num_requests"] = [
                        int(x) for x in values["task_num_requests"]
                        if int(x) > 0 and int(x) in valid_req
                    ]
                valid_kv = set(_divisors(num_local_kv_heads))
                if "task_num_kv_heads" in values:
                    values["task_num_kv_heads"] = [
                        int(x) for x in values["task_num_kv_heads"]
                        if int(x) > 0 and int(x) in valid_kv
                    ]

            keys = sorted(values.keys())
            pre_grid = 1
            for k in keys:
                pre_grid *= max(1, len(values[k]))

            if any(len(values[k]) == 0 for k in keys):
                print(f"[autotune][prune] target={target} mode=grid kept=0/{pre_grid}")
                return []

            candidates = []
            seen = set()
            for combo in itertools.product(*[values[k] for k in keys]):
                cand = dict(zip(keys, combo))
                sig = json.dumps(cand, sort_keys=True)
                if sig in seen:
                    continue
                seen.add(sig)
                ok, _ = validate_autotune_candidate(target, cand, case=case)
                if ok:
                    candidates.append(cand)

            kind = case["kind"] if case else ""
            print(f"[autotune][prune] target={target} kind={kind} mode=grid kept={len(candidates)}/{pre_grid}")
            return candidates

        def _linear_sig(kind, n, k):
            return f"{kind}:N{int(n)}:K{int(k)}:BT{int(args.max_num_batched_tokens)}"

        def _attn_sig(kind):
            return (
                f"{kind}:REQ{int(args.max_num_batched_requests)}:TOK{int(args.max_num_batched_tokens)}:"
                f"QH{int(num_local_q_heads)}:KVH{int(num_local_kv_heads)}:HD{int(head_dim)}"
            )

        def build_autotune_cases():
            return [
                {
                    "target": "linear_layer",
                    "kind": "qkv_proj",
                    "signature": _linear_sig("qkv_proj", fused_outdim_1 // world_size, hidden_size),
                    "output_size": fused_outdim_1 // world_size,
                    "reduction_size": hidden_size,
                },
                {
                    "target": "linear_layer",
                    "kind": "gatedup_proj",
                    "signature": _linear_sig("gatedup_proj", fused_outdim_2 // world_size, hidden_size),
                    "output_size": fused_outdim_2 // world_size,
                    "reduction_size": hidden_size,
                },
                {
                    "target": "linear_layer",
                    "kind": "lm_head",
                    "signature": _linear_sig("lm_head", vocab_size, hidden_size),
                    "output_size": vocab_size,
                    "reduction_size": hidden_size,
                },
                {
                    "target": "linear_with_residual_layer",
                    "kind": "o_proj",
                    "signature": _linear_sig("o_proj", hidden_size, num_local_q_heads * head_dim),
                    "output_size": hidden_size,
                    "reduction_size": num_local_q_heads * head_dim,
                },
                {
                    "target": "linear_with_residual_layer",
                    "kind": "down_proj",
                    "signature": _linear_sig("down_proj", hidden_size, intermediate_size // world_size),
                    "output_size": hidden_size,
                    "reduction_size": intermediate_size // world_size,
                },
                {
                    "target": "paged_attention_layer",
                    "kind": "self_attention",
                    "signature": _attn_sig("self_attention"),
                },
            ]

        def _ensure_target_dict(cfg, target):
            if target not in cfg or not isinstance(cfg[target], dict):
                cfg[target] = {}
            if "by_signature" not in cfg[target] or not isinstance(cfg[target]["by_signature"], dict):
                cfg[target]["by_signature"] = {}

        def build_mpk(tune_cfg=None, autotune_case=None):
            num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
            qo_indptr_buffer = torch.empty(args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
            paged_kv_indptr_buffer = torch.empty(args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
            paged_kv_indices_buffer = torch.empty(args.max_num_pages, dtype=torch.int32, device="cuda")
            paged_kv_last_page_len_buffer = torch.empty(args.max_num_batched_requests, dtype=torch.int32, device="cuda")
            mpk = mi.PersistentKernel(
                mode="offline",
                world_size=world_size,
                mpi_rank=rank,
                num_workers=num_workers,
                num_local_schedulers=num_schedulers,
                num_remote_schedulers=0,
                max_seq_length=args.max_seq_length,
                max_num_batched_requests=args.max_num_batched_requests,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_pages=args.max_num_pages,
                page_size=args.page_size,
                eos_token_id=model.config.eos_token_id if not args.ignore_eos else IGNORE_EOS_SENTINEL,
                meta_tensors={
                    "step": step,
                    "tokens": tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "num_new_tokens": num_new_tokens,
                    "prompt_lengths": prompt_lengths,
                    "qo_indptr_buffer": qo_indptr_buffer,
                    "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                    "paged_kv_indices_buffer": paged_kv_indices_buffer,
                    "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
                },
                profiler_tensor=profiler_tensor,
                trace_name=args.trace_name,
                spec_decode_config=spec_decode_config,
                use_cutlass_kernel=args.use_cutlass_kernel,
                enable_launch_timing=(autotune_case is not None),
            )
            split_batch_size = max(args.max_num_batched_tokens // 16, 1)
            # Candidate profiling micro-graph: only build the selected callsite.
            if autotune_case is not None:
                layer = model.model.layers[0]
                call_sig = autotune_case["signature"]
                kind = autotune_case["kind"]
                # Keep autotune micro-graph as a single-kernel graph. Repetition
                # is performed in worker runtime loop to avoid disconnected
                # repeated subgraphs that can trip runtime graph assertions.
                kernel_repeat = 1
                if kind == "qkv_proj":
                    rmsnorm_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_rmsnorm_out",
                        io_category="cuda_tensor",
                    )
                    attn_in = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size),
                        dtype=mi.bfloat16,
                        name="at_attn_in",
                        io_category="cuda_tensor",
                    )
                    w_q = mpk.attach_input(torch_tensor=layer.self_attn.q_proj.weight, name="at_q_proj")
                    w_k = mpk.attach_input(torch_tensor=layer.self_attn.k_proj.weight, name="at_k_proj")
                    w_v = mpk.attach_input(torch_tensor=layer.self_attn.v_proj.weight, name="at_v_proj")
                    w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, num_groups=model.config.num_key_value_heads // world_size, name="at_qkv_proj")
                    for _ in range(kernel_repeat):
                        mpk.linear_layer(
                            input=rmsnorm_out,
                            weight=w_qkv,
                            output=attn_in,
                            grid_dim=_linear_grid(
                                tune_cfg,
                                "linear_layer",
                                output_size=w_qkv.dim(0),
                                default=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), args.use_cutlass_kernel), split_batch_size, 1),
                                swap_xy=False,
                                call_sig=call_sig,
                            ),
                            block_dim=(128, 1, 1),
                            pipe_stage=_pipe(tune_cfg, "linear_layer", call_sig=call_sig),
                        )
                    return mpk
                if kind == "gatedup_proj":
                    rmsnorm_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_rmsnorm_out",
                        io_category="cuda_tensor",
                    )
                    mlp_mid = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size),
                        dtype=mi.bfloat16,
                        name="at_mlp_mid",
                        io_category="cuda_tensor",
                    )
                    w_gate_proj = mpk.attach_input(torch_tensor=layer.mlp.gate_proj.weight, name="at_gate_proj")
                    w_up_proj = mpk.attach_input(torch_tensor=layer.mlp.up_proj.weight, name="at_up_proj")
                    rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0), args.use_cutlass_kernel)
                    w_gatedup = mpk.shuffle_tensors(
                        inputs=[w_gate_proj, w_up_proj],
                        shuffled_dim=0,
                        num_groups=rmsnorm_num_tasks // 2,
                        name="at_gatedup_proj",
                    )
                    for _ in range(kernel_repeat):
                        mpk.linear_layer(
                            input=rmsnorm_out,
                            weight=w_gatedup,
                            output=mlp_mid,
                            grid_dim=_linear_grid(
                                tune_cfg,
                                "linear_layer",
                                output_size=w_gatedup.dim(0),
                                default=(rmsnorm_num_tasks, split_batch_size, 1),
                                swap_xy=False,
                                call_sig=call_sig,
                            ),
                            block_dim=(128, 1, 1),
                            pipe_stage=_pipe(tune_cfg, "linear_layer", call_sig=call_sig),
                        )
                    return mpk
                if kind == "lm_head":
                    rmsnorm_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_rmsnorm_out",
                        io_category="cuda_tensor",
                    )
                    argmax_in = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, vocab_size),
                        dtype=mi.bfloat16,
                        name="at_argmax_in",
                        io_category="cuda_tensor",
                    )
                    w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="at_lm_head")
                    for _ in range(kernel_repeat):
                        mpk.linear_layer(
                            input=rmsnorm_out,
                            weight=w_proj,
                            output=argmax_in,
                            grid_dim=_linear_grid(
                                tune_cfg,
                                "linear_layer",
                                output_size=w_proj.dim(0),
                                default=(w_proj.dim(0) // 256, split_batch_size, 1),
                                swap_xy=False,
                                call_sig=call_sig,
                            ),
                            block_dim=(128, 1, 1),
                            pipe_stage=_pipe(tune_cfg, "linear_layer", call_sig=call_sig),
                        )
                    return mpk
                if kind == "o_proj":
                    y = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_residual",
                        io_category="cuda_tensor",
                    )
                    attn_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
                        dtype=mi.bfloat16,
                        name="at_attn_out",
                        io_category="cuda_tensor",
                    )
                    attn_proj_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_o_proj_out",
                        io_category="cuda_tensor",
                    )
                    w = mpk.attach_input(torch_tensor=layer.self_attn.o_proj.weight, name="at_o_proj")
                    for _ in range(kernel_repeat):
                        mpk.linear_with_residual_layer(
                            input=attn_out,
                            weight=w,
                            residual=y,
                            output=attn_proj_out,
                            grid_dim=_linear_grid(
                                tune_cfg,
                                "linear_with_residual_layer",
                                output_size=hidden_size,
                                default=(split_batch_size, hidden_size // 64, 1),
                                swap_xy=True,
                                call_sig=call_sig,
                            ),
                            block_dim=(128, 1, 1),
                            pipe_stage=_pipe(tune_cfg, "linear_with_residual_layer", call_sig=call_sig),
                        )
                    return mpk
                if kind == "down_proj":
                    y = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_residual",
                        io_category="cuda_tensor",
                    )
                    silu_mul_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, intermediate_size // world_size),
                        dtype=mi.bfloat16,
                        name="at_silu_mul_out",
                        io_category="cuda_tensor",
                    )
                    mlp_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16,
                        name="at_down_proj_out",
                        io_category="cuda_tensor",
                    )
                    w = mpk.attach_input(torch_tensor=layer.mlp.down_proj.weight, name="at_down_proj")
                    for _ in range(kernel_repeat):
                        mpk.linear_with_residual_layer(
                            input=silu_mul_out,
                            weight=w,
                            residual=y,
                            output=mlp_out,
                            grid_dim=_linear_grid(
                                tune_cfg,
                                "linear_with_residual_layer",
                                output_size=hidden_size,
                                default=(split_batch_size, hidden_size // 64, 1),
                                swap_xy=True,
                                call_sig=call_sig,
                            ),
                            block_dim=(128, 1, 1),
                            pipe_stage=_pipe(tune_cfg, "linear_with_residual_layer", call_sig=call_sig),
                        )
                    return mpk
                if kind == "self_attention":
                    attn_in = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size),
                        dtype=mi.bfloat16,
                        name="at_attn_in",
                        io_category="cuda_tensor",
                    )
                    attn_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
                        dtype=mi.bfloat16,
                        name="at_attn_out",
                        io_category="cuda_tensor",
                    )
                    cos_pos_embed = mpk.attach_input(
                        torch_tensor=position_embeddings[0][0, :4096, :],
                        name="cos_position_embedding",
                    )
                    sin_pos_embed = mpk.attach_input(
                        torch_tensor=position_embeddings[1][0, :4096, :],
                        name="sin_position_embedding",
                    )
                    w_q_norm = mpk.attach_input(torch_tensor=layer.self_attn.q_norm.weight, name="at_q_norm")
                    w_k_norm = mpk.attach_input(torch_tensor=layer.self_attn.k_norm.weight, name="at_k_norm")
                    k_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[0][0], name="at_k_cache")
                    v_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[1][0], name="at_v_cache")
                    for _ in range(kernel_repeat):
                        mpk.paged_attention_layer(
                            input=attn_in,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            q_norm=w_q_norm,
                            k_norm=w_k_norm,
                            cos_pos_embed=cos_pos_embed,
                            sin_pos_embed=sin_pos_embed,
                            output=attn_out,
                            grid_dim=_paged_attention_grid(
                                tune_cfg,
                                "paged_attention_layer",
                                default=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                                call_sig=call_sig,
                            ),
                            block_dim=(128, 1, 1),
                            max_tokens=_max_tokens(tune_cfg, "paged_attention_layer", call_sig=call_sig),
                        )
                    return mpk
                raise ValueError(f"Unsupported autotune case: {autotune_case}")

            x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
            cos_pos_embed = mpk.attach_input(torch_tensor=position_embeddings[0][0, :4096, :], name="cos_position_embedding")
            sin_pos_embed = mpk.attach_input(torch_tensor=position_embeddings[1][0, :4096, :], name="sin_position_embedding")
            y = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="embed_out", io_category="cuda_tensor")
            rmsnorm_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
            attn_in = mpk.new_tensor(dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size), dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor")
            lse = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads, num_local_kv_heads),
                strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads),
                dtype=mi.float32,
                name="lse",
                io_category="cuda_tensor",
            )
            attn_out_tmp = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim, num_local_kv_heads),
                strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
                dtype=mi.bfloat16,
                name="attn_out_tmp",
                io_category="cuda_tensor",
            )
            attn_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim), dtype=mi.bfloat16, name="attn_out", io_category="cuda_tensor")
            attn_proj_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="attn_proj_out", io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor")
            allreduce_buf = mpk.new_tensor(dims=(world_size, args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="all_reduce_buf", io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor")
            attn_allreduce_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="attn_allreduce_out", io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor")
            mlp_mid = mpk.new_tensor(dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
            silu_mul_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, intermediate_size // world_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
            mlp_out = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="mlp_out", io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor")
            mlp_final = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size), dtype=mi.bfloat16, name="mlp_final", io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor")
            argmax_in = mpk.new_tensor(dims=(args.max_num_batched_tokens, vocab_size), dtype=mi.bfloat16, name="argmax_in", io_category="cuda_tensor")
            argmax_part_value = mpk.new_tensor(dims=(args.max_num_batched_tokens, mpk.num_workers), dtype=mi.bfloat16, name="argmax_part_value", io_category="cuda_tensor")
            argmax_part_index = mpk.new_tensor(dims=(args.max_num_batched_tokens, mpk.num_workers), dtype=mi.int64, name="argmax_part_index", io_category="cuda_tensor")
            argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

            w = mpk.attach_input(torch_tensor=model.model.embed_tokens.weight, name="embed_tokens")
            mpk.embed_layer(input=x, weight=w, output=y, grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
            x = y
            for i, layer in enumerate(model.model.layers):
                qkv_sig = _linear_sig("qkv_proj", fused_outdim_1 // world_size, hidden_size)
                attn_sig = _attn_sig("self_attention")
                oproj_sig = _linear_sig("o_proj", hidden_size, num_local_q_heads * head_dim)
                gatedup_sig = _linear_sig("gatedup_proj", fused_outdim_2 // world_size, hidden_size)
                downproj_sig = _linear_sig("down_proj", hidden_size, intermediate_size // world_size)
                w_norm = mpk.attach_input(torch_tensor=layer.input_layernorm.weight, name=f"layer_{i}_input_layernorm")
                w_q = mpk.attach_input(torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj")
                w_k = mpk.attach_input(torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj")
                w_v = mpk.attach_input(torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj")
                w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, num_groups=model.config.num_key_value_heads // world_size, name=f"layer_{i}_qkv_proj")
                mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
                mpk.linear_layer(
                    input=rmsnorm_out,
                    weight=w_qkv,
                    output=attn_in,
                    grid_dim=_linear_grid(
                        tune_cfg,
                        "linear_layer",
                        output_size=w_qkv.dim(0),
                        default=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), args.use_cutlass_kernel), split_batch_size, 1),
                        swap_xy=False,
                        call_sig=qkv_sig,
                    ),
                    block_dim=(128, 1, 1),
                    pipe_stage=_pipe(tune_cfg, "linear_layer", call_sig=qkv_sig),
                )
                w_q_norm = mpk.attach_input(torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm")
                w_k_norm = mpk.attach_input(torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm")
                k_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache")
                v_cache = mpk.attach_input(torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache")
                mpk.paged_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=_paged_attention_grid(
                        tune_cfg,
                        "paged_attention_layer",
                        default=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                        call_sig=attn_sig,
                    ),
                    block_dim=(128, 1, 1),
                    max_tokens=_max_tokens(tune_cfg, "paged_attention_layer", call_sig=attn_sig),
                )
                w = mpk.attach_input(torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj")
                mpk.linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    output=attn_proj_out,
                    grid_dim=_linear_grid(
                        tune_cfg,
                        "linear_with_residual_layer",
                        output_size=hidden_size,
                        default=(split_batch_size, hidden_size // 64, 1),
                        swap_xy=True,
                        call_sig=oproj_sig,
                    ),
                    block_dim=(128, 1, 1),
                    pipe_stage=_pipe(tune_cfg, "linear_with_residual_layer", call_sig=oproj_sig),
                )
                x = attn_proj_out
                if world_size > 1:
                    mpk.allreduce_layer(input=attn_proj_out, buffer=allreduce_buf, output=attn_allreduce_out, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
                    x = attn_allreduce_out
                w_norm = mpk.attach_input(torch_tensor=layer.post_attention_layernorm.weight, name=f"layer_{i}_post_attn_layernorm")
                w_gate_proj = mpk.attach_input(torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj")
                w_up_proj = mpk.attach_input(torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj")
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0), args.use_cutlass_kernel)
                w_gatedup = mpk.shuffle_tensors(inputs=[w_gate_proj, w_up_proj], shuffled_dim=0, num_groups=rmsnorm_num_tasks // 2, name=f"layer_{i}_gatedup_proj")
                mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
                mpk.linear_layer(
                    input=rmsnorm_out,
                    weight=w_gatedup,
                    output=mlp_mid,
                    grid_dim=_linear_grid(
                        tune_cfg,
                        "linear_layer",
                        output_size=w_gatedup.dim(0),
                        default=(rmsnorm_num_tasks, split_batch_size, 1),
                        swap_xy=False,
                        call_sig=gatedup_sig,
                    ),
                    block_dim=(128, 1, 1),
                    pipe_stage=_pipe(tune_cfg, "linear_layer", call_sig=gatedup_sig),
                )
                mpk.silu_mul_layer(input=mlp_mid, output=silu_mul_out, grid_dim=(rmsnorm_num_tasks // 2, 1, 1), block_dim=(128, 1, 1))
                w = mpk.attach_input(torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj")
                mpk.linear_with_residual_layer(
                    input=silu_mul_out,
                    weight=w,
                    residual=x,
                    output=mlp_out,
                    grid_dim=_linear_grid(
                        tune_cfg,
                        "linear_with_residual_layer",
                        output_size=hidden_size,
                        default=(split_batch_size, hidden_size // 64, 1),
                        swap_xy=True,
                        call_sig=downproj_sig,
                    ),
                    block_dim=(128, 1, 1),
                    pipe_stage=_pipe(tune_cfg, "linear_with_residual_layer", call_sig=downproj_sig),
                )
                x = mlp_out
                if world_size > 1:
                    mpk.allreduce_layer(input=mlp_out, buffer=allreduce_buf, output=mlp_final, grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1))
                    x = mlp_final
            w_norm = mpk.attach_input(torch_tensor=model.model.norm.weight, name="model_norm_weight")
            w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
            mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(mpk.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_proj,
                output=argmax_in,
                grid_dim=_linear_grid(
                    tune_cfg,
                    "linear_layer",
                    output_size=w_proj.dim(0),
                    default=(w_proj.dim(0) // 256, split_batch_size, 1),
                    swap_xy=False,
                    call_sig=_linear_sig("lm_head", vocab_size, hidden_size),
                ),
                block_dim=(128, 1, 1),
                pipe_stage=_pipe(
                    tune_cfg,
                    "linear_layer",
                    call_sig=_linear_sig("lm_head", vocab_size, hidden_size),
                ),
            )
            mpk.argmax_partial_layer(input=argmax_in, output=(argmax_part_value, argmax_part_index), grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
            mpk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=argmax_out, grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
            return mpk

        if args.autotune_worker_task_json is not None:
            if args.autotune_worker_result_json is None:
                raise ValueError("--autotune-worker-result-json is required in worker mode")
            with open(args.autotune_worker_task_json, "r") as f:
                task = json.load(f)
            try:
                phase = task.get("phase", "both")
                mpk_worker = build_mpk(task["cfg"], autotune_case=task["case"])
                artifact_dir = task["artifact_dir"]
                ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
                so_path = task.get("so_path", os.path.join(artifact_dir, f"test{ext_suffix}"))
                compile_ms = 0.0
                run_wait_ms = 0.0
                run_ms = 0.0
                if phase in ("both", "compile"):
                    t0_compile = time.perf_counter()
                    emitted = mpk_worker.emit_task_graph_files(artifact_dir=artifact_dir, prefix="test")
                    need_compile = not (phase == "both" and os.path.exists(so_path))
                    if need_compile:
                        compile_cmd = mpk_worker.build_compile_command(emitted["cuda_code_path"], so_path)
                        proc = subprocess.run(
                            compile_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        if proc.returncode != 0:
                            with open(os.path.join(artifact_dir, "compile.log"), "w") as f:
                                f.write(proc.stdout or "")
                            with open(args.autotune_worker_result_json, "w") as f:
                                json.dump({"ok": False, "phase": "compile", "returncode": proc.returncode}, f)
                            sys.exit(0)
                    compile_ms = (time.perf_counter() - t0_compile) * 1000.0
                    if phase == "compile":
                        with open(args.autotune_worker_result_json, "w") as f:
                            json.dump(
                                {
                                    "ok": True,
                                    "phase": "compile",
                                    "so_path": so_path,
                                    "compile_ms": compile_ms,
                                },
                                f,
                            )
                        sys.exit(0)
                if phase == "run" and not os.path.exists(so_path):
                    with open(args.autotune_worker_result_json, "w") as f:
                        json.dump({"ok": False, "phase": "run", "error": f"missing so_path: {so_path}"}, f)
                    sys.exit(0)
                run_lock_fd = None
                run_lock_file = os.environ.get("MIRAGE_AUTOTUNE_RUN_LOCK_FILE", "")
                try:
                    if run_lock_file:
                        print(f"[autotune_worker] waiting run lock: {run_lock_file}", flush=True)
                        t0_wait = time.perf_counter()
                        run_lock_fd = _acquire_run_lock(run_lock_file)
                        run_wait_ms = (time.perf_counter() - t0_wait) * 1000.0
                        print(f"[autotune_worker] acquired run lock: {run_lock_file}", flush=True)
                    t0_run = time.perf_counter()
                    if phase == "run":
                        # Keep runtime graph state aligned with the compiled artifact.
                        mpk_worker.emit_task_graph_files(artifact_dir=artifact_dir, prefix="test")
                    mpk_worker.load_compiled_module_from_path(so_path)
                    mpk_worker.initialize_runtime()
                    repeat_n = max(1, int(args.autotune_kernel_repeat))
                    worker_ms_sum = 0.0
                    scheduler_ms_sum = 0.0
                    total_ms_sum = 0.0
                    timing = {}
                    if mpk_worker.set_enable_timing_func is not None:
                        mpk_worker.set_enable_timing_func(1)
                    for _ in range(repeat_n):
                        mpk_worker()
                        check_cuda_last_error_or_raise(f"autotune_worker/{task['case']['kind']}")
                        timing = mpk_worker.last_launch_timing
                        worker_ms_sum += float(timing.get("worker_ms", 0.0))
                        scheduler_ms_sum += float(timing.get("scheduler_ms", 0.0))
                        total_ms_sum += float(timing.get("total_ms", 0.0))
                    if mpk_worker.set_enable_timing_func is not None:
                        mpk_worker.set_enable_timing_func(0)
                    run_ms = worker_ms_sum / repeat_n
                    latency = (worker_ms_sum + scheduler_ms_sum) / repeat_n
                    if latency <= 0:
                        latency = total_ms_sum / repeat_n if total_ms_sum > 0 else 1e30
                    # MAX_TOKENS tunes paged attention tile depth; if less than full
                    # token count, account for repeated runs needed to cover input.
                    if task["case"]["target"] == "paged_attention_layer":
                        tc = _target_cfg(
                            task.get("cfg", {}),
                            "paged_attention_layer",
                            call_sig=task["case"].get("signature"),
                        )
                        max_tokens = int(tc.get("MAX_TOKENS", tc.get("max_tokens", args.max_num_batched_tokens)))
                        max_tokens = max(1, max_tokens)
                        repeats = int(math.ceil(args.max_num_batched_tokens / max_tokens))
                        latency *= repeats
                        run_ms *= repeats
                    timing["autotune_kernel_repeat"] = repeat_n
                    mpk_worker.finalize()
                    host_run_ms = (time.perf_counter() - t0_run) * 1000.0
                finally:
                    if run_lock_fd is not None:
                        _release_run_lock(run_lock_fd)
                        print(f"[autotune_worker] released run lock: {run_lock_file}", flush=True)
                with open(args.autotune_worker_result_json, "w") as f:
                    json.dump(
                        {
                            "ok": True,
                            "phase": "run",
                            "latency_ms": latency,
                            "timing": timing,
                            "compile_ms": compile_ms,
                            "run_wait_ms": run_wait_ms,
                            "run_ms": run_ms,
                            "host_run_ms": host_run_ms,
                        },
                        f,
                    )
                sys.exit(0)
            except Exception as e:
                with open(args.autotune_worker_result_json, "w") as f:
                    json.dump({"ok": False, "phase": "runtime", "error": str(e)}, f)
                sys.exit(0)

        tuned_cfg = {}
        if args.autotune and rank == 0:
            if not args.autotune_list:
                raise ValueError("--autotune requires --autotune-list")
            if not os.path.exists(args.autotune_list):
                raise FileNotFoundError(
                    f"autotune list not found: {args.autotune_list}. "
                    "Use --autotune-list to specify a valid JSON file."
                )
            cache_dir = os.path.dirname(args.autotune_cache)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            cache = {}
            if os.path.exists(args.autotune_cache):
                with open(args.autotune_cache, "r") as f:
                    cache = json.load(f)
            fp = model_fingerprint(model, args, world_size, total_num_requests)
            cache_entry = cache.get(fp)
            from_cache = False
            if (not args.autotune_force) and cache_entry is not None:
                tuned_cfg = cache_entry["recommended"]
                print(f"Loaded autotune config from cache: {args.autotune_cache}")
                from_cache = True
            else:
                with open(args.autotune_list, "r") as f:
                    tune_list = json.load(f)
                autotune_session = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                autotune_root = os.path.join(
                    DEFAULT_SAVE_DIR, "autotune_artifacts", fp, autotune_session
                )
                os.makedirs(autotune_root, exist_ok=True)
                print(f"[autotune] artifact_session={autotune_session} root={autotune_root}", flush=True)
                ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
                all_results = {}
                profile_gpu_ids = [x.strip() for x in args.autotune_profile_gpus.split(",") if x.strip()]
                if not profile_gpu_ids:
                    profile_gpu_ids = ["0"]
                run_lock_dir = os.path.join(DEFAULT_SAVE_DIR, "autotune_run_locks")
                os.makedirs(run_lock_dir, exist_ok=True)
                cases = build_autotune_cases()
                reference_by_sig = {}
                reference_paths = []
                if args.autotune_reference_logs:
                    for raw in args.autotune_reference_logs.split(","):
                        p = raw.strip()
                        if p:
                            reference_paths.append(p)
                if reference_paths:
                    pat = re.compile(
                        r"\[autotune\] target=([^ ]+) case=([^ ]+) sig=([^ ]+) "
                        r"best_run_ms=([0-9.]+) cfg=(\{.*\})"
                    )
                    loaded_items = 0
                    for log_path in reference_paths:
                        if not os.path.exists(log_path):
                            print(f"[autotune][guide] skip missing log: {log_path}", flush=True)
                            continue
                        try:
                            with open(log_path, "r") as f:
                                for ln in f:
                                    m = pat.search(ln)
                                    if m is None:
                                        continue
                                    target_name = m.group(1)
                                    case_name = m.group(2)
                                    call_sig = m.group(3)
                                    best_run_ms = float(m.group(4))
                                    cand = ast.literal_eval(m.group(5))
                                    if not isinstance(cand, dict):
                                        continue
                                    reference_by_sig[call_sig] = {
                                        "target": target_name,
                                        "case": case_name,
                                        "cand": cand,
                                        "best_run_ms": best_run_ms,
                                        "log_path": log_path,
                                    }
                                    loaded_items += 1
                        except Exception as e:
                            print(
                                f"[autotune][guide] failed to parse log={log_path}: {e}",
                                flush=True,
                            )
                    print(
                        f"[autotune][guide] loaded={loaded_items} signatures={len(reference_by_sig)} "
                        f"from_logs={len(reference_paths)} force={int(bool(args.autotune_reference_force))}",
                        flush=True,
                    )
                sanitizer_enabled = bool(args.autotune_debug_sanitizer)
                sanitizer_limit = max(0, int(args.autotune_debug_sanitizer_limit))
                sanitizer_bin = args.autotune_debug_sanitizer_bin
                if sanitizer_enabled:
                    resolved = shutil.which(sanitizer_bin)
                    if resolved is None:
                        print(
                            f"[autotune][sanitizer] disabled: binary not found: {sanitizer_bin}",
                            flush=True,
                        )
                        sanitizer_enabled = False
                    else:
                        sanitizer_bin = resolved
                        print(
                            f"[autotune][sanitizer] enabled bin={sanitizer_bin} limit={sanitizer_limit}",
                            flush=True,
                        )
                sanitizer_state = {"enabled": sanitizer_enabled, "done": 0}

                def maybe_run_debug_sanitizer(rec, reason):
                    if not sanitizer_state["enabled"] or sanitizer_state["done"] >= sanitizer_limit:
                        return
                    sanitizer_state["done"] += 1
                    artifact_dir = rec["artifact_dir"]
                    os.makedirs(artifact_dir, exist_ok=True)
                    debug_idx = int(sanitizer_state["done"])
                    task_path = os.path.join(
                        artifact_dir, f"compute_sanitizer_task_{debug_idx}.json"
                    )
                    result_path = os.path.join(
                        artifact_dir, f"compute_sanitizer_result_{debug_idx}.json"
                    )
                    log_path = os.path.join(
                        artifact_dir, f"compute_sanitizer_memcheck_{debug_idx}.log"
                    )
                    debug_so_path = os.path.join(
                        artifact_dir, f"test.compute_sanitizer_{debug_idx}{ext_suffix}"
                    )
                    task_payload = {
                        "phase": "both",
                        "case": rec["case"],
                        "cfg": rec["cfg"],
                        "artifact_dir": artifact_dir,
                        "so_path": debug_so_path,
                    }
                    with open(task_path, "w") as f:
                        json.dump(task_payload, f, indent=2)

                    cmd = [
                        sanitizer_bin,
                        "--tool",
                        "memcheck",
                        "--show-backtrace",
                        "yes",
                        "--target-processes",
                        "all",
                        "--error-exitcode",
                        "86",
                        sys.executable,
                        os.path.abspath(__file__),
                        "--use-mirage",
                        "--max-num-batched-tokens",
                        str(args.max_num_batched_tokens),
                        "--max-num-batched-requests",
                        str(args.max_num_batched_requests),
                        "--page-size",
                        str(args.page_size),
                        "--max-num-pages",
                        str(args.max_num_pages),
                        "--max-seq-length",
                        str(args.max_seq_length),
                        "--autotune-worker-task-json",
                        task_path,
                        "--autotune-worker-result-json",
                        result_path,
                    ]
                    if args.model_path is not None:
                        cmd.extend(["--model-path", str(args.model_path)])
                    else:
                        cmd.extend(["--model", str(args.model)])
                    # In synthetic worker mode, decode-length knobs are ignored by
                    # kernel profiling and can make logs misleading.
                    if not args.autotune_worker_synthetic:
                        if args.prompt_tokens is not None:
                            cmd.extend(["--prompt-tokens", str(args.prompt_tokens)])
                        if args.max_new_tokens is not None:
                            cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
                        if args.ignore_eos:
                            cmd.append("--ignore-eos")
                    if args.split_kv_cache:
                        cmd.append("--split-kv-cache")
                    if args.use_cutlass_kernel:
                        cmd.append("--use-cutlass-kernel")
                    if not args.autotune_worker_synthetic:
                        cmd.append("--no-autotune-worker-synthetic")

                    print(
                        f"[autotune][sanitizer] start idx={debug_idx}/{sanitizer_limit} "
                        f"case={rec['case']['kind']} cand={rec['cand']} reason={reason}",
                        flush=True,
                    )
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    with open(log_path, "w") as f:
                        f.write(proc.stdout or "")
                    print(
                        f"[autotune][sanitizer] done idx={debug_idx}/{sanitizer_limit} "
                        f"ret={proc.returncode} log={log_path}",
                        flush=True,
                    )

                for case in cases:
                    target = case["target"]
                    call_sig = case["signature"]
                    case_kind = case["kind"]
                    if args.autotune_reference_strict_signatures and reference_paths:
                        if call_sig not in reference_by_sig:
                            print(
                                f"[autotune][guide] strict_skip target={target} case={case_kind} "
                                f"sig={call_sig} (not found in reference logs)",
                                flush=True,
                            )
                            continue
                    candidates = build_pruned_candidates(target, tune_list.get(target, []), case=case)
                    reference_entry = reference_by_sig.get(call_sig)
                    if (
                        reference_entry is not None
                        and reference_entry.get("target") == target
                        and isinstance(reference_entry.get("cand"), dict)
                    ):
                        ref_cand = reference_entry["cand"]
                        valid, reason = validate_autotune_candidate(target, ref_cand, case=case)
                        if not valid:
                            print(
                                f"[autotune][guide] ignore invalid target={target} case={case_kind} "
                                f"sig={call_sig} cand={ref_cand} reason={reason}",
                                flush=True,
                            )
                        else:
                            ref_key = json.dumps(ref_cand, sort_keys=True)
                            reordered = [ref_cand]
                            seen_keys = {ref_key}
                            for cand in candidates:
                                cand_key = json.dumps(cand, sort_keys=True)
                                if cand_key in seen_keys:
                                    continue
                                reordered.append(cand)
                                seen_keys.add(cand_key)
                            candidates = reordered
                            print(
                                f"[autotune][guide] prioritize target={target} case={case_kind} "
                                f"sig={call_sig} cand={ref_cand} src={reference_entry['log_path']}",
                                flush=True,
                            )
                    if not candidates:
                        continue
                    records = []
                    seen = set()
                    for i, cand in enumerate(candidates):
                        sig = json.dumps(cand, sort_keys=True)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        valid, reason = validate_autotune_candidate(target, cand, case=case)
                        if not valid:
                            print(
                                f"[autotune][skip] target={target} case={case_kind} "
                                f"cand={cand} reason={reason}"
                            )
                            continue
                        cfg = copy.deepcopy(tuned_cfg)
                        _ensure_target_dict(cfg, target)
                        cfg[target]["by_signature"][call_sig] = cand
                        artifact_dir = os.path.join(autotune_root, f"{target}_{case_kind}_{i}")
                        records.append(
                            {
                                "cfg": cfg,
                                "cand": cand,
                                "artifact_dir": artifact_dir,
                                "case": case,
                            }
                        )
                    if not records:
                        continue
                    use_multiproc_workers = len(profile_gpu_ids) > 1

                    def _compile_one_candidate(rec):
                        artifact_dir = rec["artifact_dir"]
                        os.makedirs(artifact_dir, exist_ok=True)
                        compile_log = os.path.join(artifact_dir, "compile.log")
                        so_path = os.path.join(artifact_dir, f"test{ext_suffix}")
                        try:
                            mpk_local = build_mpk(rec["cfg"], autotune_case=rec["case"])
                            emitted = mpk_local.emit_task_graph_files(artifact_dir=artifact_dir, prefix="test")
                            compile_cmd = mpk_local.build_compile_command(emitted["cuda_code_path"], so_path)
                            proc = subprocess.run(
                                compile_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            with open(compile_log, "w") as f:
                                f.write(proc.stdout or "")
                            if proc.returncode != 0:
                                try:
                                    mpk_local.finalize()
                                except Exception:
                                    pass
                                return False, rec, f"nvcc ret={proc.returncode}"
                            rec["mpk"] = mpk_local
                            rec["so_path"] = so_path
                            return True, rec, ""
                        except Exception as e:
                            return False, rec, str(e)

                    def _profile_one_candidate_in_worker(rec, gpu_id, slot_id):
                        artifact_dir = rec["artifact_dir"]
                        os.makedirs(artifact_dir, exist_ok=True)
                        task_path = os.path.join(artifact_dir, f"worker_task_gpu{gpu_id}_slot{slot_id}.json")
                        result_path = os.path.join(artifact_dir, f"worker_result_gpu{gpu_id}_slot{slot_id}.json")
                        profile_log = os.path.join(artifact_dir, f"profile_gpu{gpu_id}_slot{slot_id}.log")
                        so_path = os.path.join(artifact_dir, f"test.gpu{gpu_id}.slot{slot_id}{ext_suffix}")
                        task_payload = {
                            "phase": "both",
                            "case": rec["case"],
                            "cfg": rec["cfg"],
                            "artifact_dir": artifact_dir,
                            "so_path": so_path,
                        }
                        # Avoid stale artifacts from previous runs polluting the current result.
                        for stale_path in (result_path, profile_log):
                            try:
                                if os.path.exists(stale_path):
                                    os.remove(stale_path)
                            except Exception:
                                pass
                        with open(task_path, "w") as f:
                            json.dump(task_payload, f, indent=2)

                        cmd = [
                            sys.executable,
                            os.path.abspath(__file__),
                            "--use-mirage",
                            "--max-num-batched-tokens",
                            str(args.max_num_batched_tokens),
                            "--max-num-batched-requests",
                            str(args.max_num_batched_requests),
                            "--page-size",
                            str(args.page_size),
                            "--max-num-pages",
                            str(args.max_num_pages),
                            "--max-seq-length",
                            str(args.max_seq_length),
                            "--autotune-worker-task-json",
                            task_path,
                            "--autotune-worker-result-json",
                            result_path,
                        ]
                        if args.model_path is not None:
                            cmd.extend(["--model-path", str(args.model_path)])
                        else:
                            cmd.extend(["--model", str(args.model)])
                        # In synthetic worker mode, decode-length knobs are ignored by
                        # kernel profiling and can make logs misleading.
                        if not args.autotune_worker_synthetic:
                            if args.prompt_tokens is not None:
                                cmd.extend(["--prompt-tokens", str(args.prompt_tokens)])
                            if args.max_new_tokens is not None:
                                cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
                            if args.ignore_eos:
                                cmd.append("--ignore-eos")
                        if args.split_kv_cache:
                            cmd.append("--split-kv-cache")
                        if args.use_cutlass_kernel:
                            cmd.append("--use-cutlass-kernel")
                        if not args.autotune_worker_synthetic:
                            cmd.append("--no-autotune-worker-synthetic")
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        env["MIRAGE_AUTOTUNE_RUN_LOCK_FILE"] = os.path.join(
                            run_lock_dir, f"gpu{gpu_id}.run.lock"
                        )
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env,
                        )
                        with open(profile_log, "w") as f:
                            f.write(proc.stdout or "")
                        if not os.path.exists(result_path):
                            return (
                                False,
                                None,
                                f"worker exit rc={proc.returncode}; missing worker result json: {result_path}",
                            )
                        with open(result_path, "r") as f:
                            result = json.load(f)
                        if not result.get("ok", False):
                            return False, None, result.get("error", str(result))
                        required = ("compile_ms", "run_wait_ms", "run_ms")
                        if any(k not in result for k in required):
                            return (
                                False,
                                None,
                                f"stale/incomplete worker result schema at {result_path}; "
                                f"keys={sorted(list(result.keys()))}",
                            )
                        return True, result, ""

                    profiled_records = []
                    completed = 0
                    if not use_multiproc_workers:
                        compile_jobs = max(1, args.autotune_compile_jobs)
                        print(
                            f"[autotune][pipeline-start] target={target} case={case_kind} "
                            f"num_candidates={len(records)} compile_workers={compile_jobs} profile_mode=serial",
                            flush=True,
                        )
                        with ThreadPoolExecutor(max_workers=compile_jobs) as compile_ex:
                            fut2rec = {compile_ex.submit(_compile_one_candidate, rec): rec for rec in records}
                            for fut in as_completed(fut2rec):
                                ok, rec, err = fut.result()
                                if not ok:
                                    completed += 1
                                    print(
                                        f"[autotune][compile-progress] target={target} case={case_kind} "
                                        f"compiled_ok=0 done={completed}/{len(records)} cand={rec['cand']} err={err}",
                                        flush=True,
                                    )
                                    maybe_run_debug_sanitizer(rec, f"compile_fail: {err}")
                                    continue
                                print(
                                    f"[autotune][compile-progress] target={target} case={case_kind} "
                                    f"compiled_ok=1 queued_for_profile cand={rec['cand']}",
                                    flush=True,
                                )
                                mpk_local = rec["mpk"]
                                try:
                                    mpk_local.load_compiled_module_from_path(rec["so_path"])
                                    mpk_local.initialize_runtime()
                                    repeat_n = max(1, int(args.autotune_kernel_repeat))
                                    worker_ms_sum = 0.0
                                    scheduler_ms_sum = 0.0
                                    total_ms_sum = 0.0
                                    if mpk_local.set_enable_timing_func is not None:
                                        mpk_local.set_enable_timing_func(1)
                                    for _ in range(repeat_n):
                                        mpk_local()
                                        check_cuda_last_error_or_raise(f"autotune_inproc/{case_kind}")
                                        timing = mpk_local.last_launch_timing
                                        worker_ms_sum += float(timing.get("worker_ms", 0.0))
                                        scheduler_ms_sum += float(timing.get("scheduler_ms", 0.0))
                                        total_ms_sum += float(timing.get("total_ms", 0.0))
                                    if mpk_local.set_enable_timing_func is not None:
                                        mpk_local.set_enable_timing_func(0)
                                    latency = (worker_ms_sum + scheduler_ms_sum) / repeat_n
                                    if latency <= 0:
                                        latency = total_ms_sum / repeat_n if total_ms_sum > 0 else 1e30
                                    if target == "paged_attention_layer":
                                        tc = _target_cfg(rec["cfg"], "paged_attention_layer", call_sig=call_sig)
                                        max_tokens = int(tc.get("MAX_TOKENS", tc.get("max_tokens", args.max_num_batched_tokens)))
                                        max_tokens = max(1, max_tokens)
                                        repeats = int(math.ceil(args.max_num_batched_tokens / max_tokens))
                                        latency *= repeats
                                        worker_ms_sum *= repeats
                                    rec["latency"] = latency
                                    rec["run_ms"] = worker_ms_sum / repeat_n
                                    profiled_records.append(rec)
                                    completed += 1
                                    print(
                                        f"[autotune][profile-progress] target={target} case={case_kind} "
                                        f"done={completed}/{len(records)} ok=1 run_ms={rec['run_ms']:.3f} "
                                        f"latency_ms={latency:.3f} "
                                        f"cand={rec['cand']}",
                                        flush=True,
                                    )
                                except Exception as e:
                                    completed += 1
                                    print(
                                        f"[autotune][profile-progress] target={target} case={case_kind} "
                                        f"done={completed}/{len(records)} ok=0 cand={rec['cand']} err={e}",
                                        flush=True,
                                    )
                                    maybe_run_debug_sanitizer(rec, f"profile_fail: {e}")
                                finally:
                                    try:
                                        mpk_local.finalize()
                                    except Exception:
                                        pass
                                    rec.pop("mpk", None)
                    else:
                        per_gpu_workers = max(1, int(args.autotune_workers_per_gpu))
                        gpu_ids = [int(gid) for gid in profile_gpu_ids] or [0]
                        compile_slots = []
                        # compile queue lanes are interleaved across GPUs for uniform dispatch.
                        for slot_id in range(per_gpu_workers):
                            for gpu_id in gpu_ids:
                                compile_slots.append((gpu_id, slot_id))
                        max_parallel_compile = max(1, min(int(args.autotune_compile_jobs), len(compile_slots)))
                        compile_slots = compile_slots[:max_parallel_compile]
                        pending_compile = list(records)
                        print(
                            f"[autotune][pipeline-start] target={target} case={case_kind} "
                            f"num_candidates={len(records)} mode=multiproc_phase_both "
                            f"compile_queue_workers={len(compile_slots)} run_queue_tokens_per_gpu=1",
                            flush=True,
                        )
                        with ThreadPoolExecutor(max_workers=len(compile_slots)) as exec_pool:
                            running = {}

                            def submit_next_compile(slot):
                                if not pending_compile:
                                    return
                                rec = pending_compile.pop(0)
                                gpu_id, slot_id = slot
                                fut = exec_pool.submit(_profile_one_candidate_in_worker, rec, gpu_id, slot_id)
                                running[fut] = (rec, slot)
                                print(
                                    f"[autotune][compile-queue][dispatch] target={target} case={case_kind} "
                                    f"gpu={gpu_id} slot={slot_id} cand={rec['cand']}",
                                    flush=True,
                                )

                            for slot in compile_slots:
                                submit_next_compile(slot)

                            while running:
                                done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                                for fut in done:
                                    rec, slot = running.pop(fut)
                                    gpu_id, slot_id = slot
                                    try:
                                        ok, result, err = fut.result()
                                    except Exception as e:
                                        ok, result, err = False, None, str(e)
                                    completed += 1
                                    if ok:
                                        rec["latency"] = float(result.get("latency_ms", 1e30))
                                        rec["run_ms"] = float(result.get("run_ms", rec["latency"]))
                                        profiled_records.append(rec)
                                        compile_ms = float(result.get("compile_ms", 0.0))
                                        wait_ms = float(result.get("run_wait_ms", 0.0))
                                        run_ms = rec["run_ms"]
                                        print(
                                            f"[autotune][run-queue][complete] target={target} case={case_kind} "
                                            f"done={completed}/{len(records)} ok=1 gpu={gpu_id} slot={slot_id} "
                                            f"compile_ms={compile_ms:.1f} wait_run_ms={wait_ms:.1f} run_ms={run_ms:.1f} "
                                            f"latency_ms={rec['latency']:.3f} cand={rec['cand']}",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"[autotune][run-queue][complete] target={target} case={case_kind} "
                                            f"done={completed}/{len(records)} ok=0 gpu={gpu_id} slot={slot_id} "
                                            f"cand={rec['cand']} err={err}",
                                            flush=True,
                                        )
                                        maybe_run_debug_sanitizer(rec, f"profile_fail_multiproc: {err}")
                                    submit_next_compile(slot)

                    records = profiled_records
                    if not records:
                        print(
                            f"[autotune][warn] target={target} case={case_kind} "
                            f"no valid candidates after worker profiling"
                        , flush=True)
                        continue
                    best = min(records, key=lambda x: x.get("run_ms", x["latency"]))
                    if (
                        reference_entry is not None
                        and reference_entry.get("target") == target
                        and isinstance(reference_entry.get("cand"), dict)
                    ):
                        ref_key = json.dumps(reference_entry["cand"], sort_keys=True)
                        rec_by_key = {json.dumps(rec["cand"], sort_keys=True): rec for rec in records}
                        ref_rec = rec_by_key.get(ref_key)
                        if ref_rec is not None:
                            best_run = float(best.get("run_ms", best["latency"]))
                            ref_run = float(ref_rec.get("run_ms", ref_rec["latency"]))
                            if args.autotune_reference_force:
                                best = ref_rec
                                print(
                                    f"[autotune][guide] force_select target={target} case={case_kind} "
                                    f"sig={call_sig} run_ms={ref_run:.3f}",
                                    flush=True,
                                )
                            elif ref_run <= best_run * 1.03:
                                best = ref_rec
                                print(
                                    f"[autotune][guide] prefer_select target={target} case={case_kind} "
                                    f"sig={call_sig} ref_run_ms={ref_run:.3f} best_run_ms={best_run:.3f} "
                                    f"(within 3%)",
                                    flush=True,
                                )
                    if target not in all_results or not isinstance(all_results[target], dict):
                        all_results[target] = {}
                    all_results[target][call_sig] = [
                        {
                            "candidate": rec["cand"],
                            "run_ms": rec.get("run_ms", rec["latency"]),
                            "latency_ms": rec["latency"],
                        }
                        for rec in sorted(records, key=lambda x: x.get("run_ms", x["latency"]))
                    ]
                    _ensure_target_dict(tuned_cfg, target)
                    tuned_cfg[target]["by_signature"][call_sig] = best["cand"]
                    print(
                        f"[autotune] target={target} case={case_kind} sig={call_sig} "
                        f"best_run_ms={best.get('run_ms', best['latency']):.3f} "
                        f"cfg={best['cand']}"
                    , flush=True)
                cache[fp] = {
                    "model_id": args.model if args.model_path is None else args.model_path,
                    "batch_size": total_num_requests,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "world_size": world_size,
                    "artifact_session": autotune_session,
                    "recommended": tuned_cfg,
                    "results": all_results,
                }
                with open(args.autotune_cache, "w") as f:
                    json.dump(cache, f, indent=2)
                print(f"Saved autotune cache to {args.autotune_cache}")
        if args.autotune and world_size > 1:
            if dist.is_initialized():
                dist.barrier()

        mpk = build_mpk(tuned_cfg if args.autotune else None)
        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])
        mpk.compile(output_dir=args.output_dir)

    # g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    warmup = 0
    # Decode up to user cap or buffer size
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))
    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
            next_token = logits.argmax(dim=-1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if (not args.ignore_eos) and next_token == model.config.eos_token_id:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        tokens_generated = max(0, end_idx - prompt_len)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, tokens_generated, run_time / max(tokens_generated, 1)
            )
        )
        
        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        check_cuda_last_error_or_raise("e2e_mpk")
        run_time = starter.elapsed_time(ender)

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = safe_decode_tokens(tokenizer, generated_ids)
            print(response)
        
        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        print("Prompt length {}, generate length {}, per-token latency (both prefill and decode): {:.3f} ms".format(
              prompt_lengths[0], step.max().item() + 1 - prompt_lengths[0], run_time / (step.max().item() + 1)
            )
        )

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            response_text = safe_decode_tokens(tokenizer, tokens[0, :end_idx])
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
