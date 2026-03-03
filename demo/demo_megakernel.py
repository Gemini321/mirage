import argparse
import os

import torch


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


def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = False):
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


def _make_minimal_persistent_kernel_meta(
    *,
    max_seq_length: int,
    max_num_pages: int,
    max_num_batched_requests: int,
    max_num_batched_tokens: int,
):
    """
    PersistentKernel currently expects a fixed set of CUDA tensors as its runtime
    ABI. For this demo we only compile a single `linear_hopper` task, so we
    allocate the smallest possible dummy buffers that satisfy shape assertions.
    """
    device = "cuda"
    total_num_requests = 1

    step = torch.zeros((total_num_requests,), dtype=torch.int32, device=device)
    tokens = torch.zeros((total_num_requests, max_seq_length), dtype=torch.long, device=device)
    input_tokens = torch.zeros((max_num_batched_tokens, 1), dtype=torch.long, device=device)
    output_tokens = torch.zeros((max_num_batched_tokens, 1), dtype=torch.long, device=device)
    num_new_tokens = torch.zeros((total_num_requests,), dtype=torch.int32, device=device)
    prompt_lengths = torch.zeros((total_num_requests,), dtype=torch.int32, device=device)

    qo_indptr_buffer = torch.zeros((max_num_batched_requests + 1,), dtype=torch.int32, device=device)
    paged_kv_indptr_buffer = torch.zeros((max_num_batched_requests + 1,), dtype=torch.int32, device=device)
    paged_kv_indices_buffer = torch.zeros((max_num_pages,), dtype=torch.int32, device=device)
    paged_kv_last_page_len_buffer = torch.zeros((max_num_batched_requests,), dtype=torch.int32, device=device)

    return {
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
    }


def _set_num_active_tokens(meta_tensors, max_num_batched_requests, num_active_tokens):
    meta_tensors["qo_indptr_buffer"].zero_()
    meta_tensors["qo_indptr_buffer"][max_num_batched_requests] = num_active_tokens


def _make_rope_tables(max_seq_length: int, head_dim: int, device: str):
    half = head_dim // 2
    if head_dim % 2 != 0:
        raise ValueError("rope expects even head_dim")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq_length, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    cos = torch.cat([cos_half, cos_half], dim=1).to(torch.bfloat16)
    sin = torch.cat([sin_half, sin_half], dim=1).to(torch.bfloat16)
    return cos, sin


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    x_f = x.float()
    w_f = weight.float()
    rms = torch.mean(x_f * x_f, dim=-1, keepdim=True)
    out = x_f * torch.rsqrt(rms + eps) * w_f
    return out.to(torch.bfloat16).float()


def _reference_silu_mul(x: torch.Tensor):
    half = x.shape[1] // 2
    gate = x[:, :half].float()
    up = x[:, half:].float()
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()


def _check_allclose(name: str, out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float):
    out_f = out.float()
    ref_f = ref.float()
    diff = out_f - ref_f
    abs_err = diff.abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    denom = ref_f.abs().clamp_min(1e-6)
    max_rel = (abs_err / denom).max().item()
    ok = torch.allclose(out_f, ref_f, atol=atol, rtol=rtol)
    print(
        f"{name} check={'PASS' if ok else 'FAIL'} "
        f"atol={atol} rtol={rtol} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} max_rel={max_rel:.6g}"
    )
    return ok

def _reference_paged_attention_causal_prefill(
    *,
    qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qk_norm: bool,
    rope: bool,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reference for this demo's paged-attention setup:
    - single request, prefill only (prefix len = 0)
    - causal mask
    - K/V come from the same QKV tensor (no history in cache)
    """
    if qkv.ndim != 2:
        raise ValueError(f"expected qkv to be 2D [T, C], got shape={tuple(qkv.shape)}")
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    num_tokens = qkv.shape[0]
    num_qo_per_kv = num_q_heads // num_kv_heads
    q_cols_per_kv = num_qo_per_kv * head_dim
    block_cols = q_cols_per_kv + 2 * head_dim
    if qkv.shape[1] != num_kv_heads * block_cols:
        raise ValueError(
            f"qkv cols mismatch: expected {num_kv_heads * block_cols}, got {qkv.shape[1]}"
        )

    qkv_f = qkv.float().view(num_tokens, num_kv_heads, block_cols)
    q = qkv_f[..., :q_cols_per_kv].view(num_tokens, num_kv_heads, num_qo_per_kv, head_dim)
    k = qkv_f[..., q_cols_per_kv : q_cols_per_kv + head_dim]
    v = qkv_f[..., q_cols_per_kv + head_dim :]

    if qk_norm:
        q_rms = torch.mean(q * q, dim=-1, keepdim=True)
        q = q * torch.rsqrt(q_rms + eps) * q_norm_weight.float().view(1, 1, 1, head_dim)
        k_rms = torch.mean(k * k, dim=-1, keepdim=True)
        k = k * torch.rsqrt(k_rms + eps) * k_norm_weight.float().view(1, 1, head_dim)

    if rope:
        cos_f = cos.float()[:num_tokens].view(num_tokens, 1, head_dim)
        sin_f = sin.float()[:num_tokens].view(num_tokens, 1, head_dim)
        half = head_dim // 2
        if head_dim % 2 != 0:
            raise ValueError("rope reference expects even head_dim")

        def apply_rope(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., :half]
            x2 = x[..., half : 2 * half]
            c1 = cos_f[..., :half]
            s1 = sin_f[..., :half]
            y1 = x1 * c1 - x2 * s1
            y2 = x1 * s1 + x2 * c1
            return torch.cat([y1, y2, x[..., 2 * half :]], dim=-1)

        # Apply RoPE to Q (per query head) and K (per kv head).
        q = apply_rope(q)
        k = apply_rope(k)

    # Align precision with the Hopper kernel:
    # - Q/K are materialized in shared memory as bf16 after norm/rope.
    # - softmax probabilities are quantized to bf16 before multiplying with V.
    # - output is written as bf16.
    q = q.to(torch.bfloat16).float()
    k = k.to(torch.bfloat16).float()
    v = v.to(torch.bfloat16).float()

    scale = head_dim**-0.5
    out = torch.empty((num_tokens, num_q_heads, head_dim), device=q.device, dtype=torch.float32)
    causal = torch.triu(
        torch.ones((num_tokens, num_tokens), device=q.device, dtype=torch.bool), diagonal=1
    )

    for kvh in range(num_kv_heads):
        k_h = k[:, kvh, :]  # [T, D]
        v_h = v[:, kvh, :]  # [T, D]
        for qoh in range(num_qo_per_kv):
            q_h = q[:, kvh, qoh, :]  # [T, D]
            scores = (q_h @ k_h.t()) * scale  # [T, T]
            scores = scores.masked_fill(causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = probs.to(torch.bfloat16).float()
            o_h = probs @ v_h  # [T, D]
            o_h = o_h.to(torch.bfloat16).float()
            out[:, kvh * num_qo_per_kv + qoh, :] = o_h

    return out.reshape(num_tokens, num_q_heads * head_dim)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile a minimal Mirage megakernel. For --kernel=linear, use the high-level "
            "PersistentKernel.linear_layer / linear_with_residual_layer APIs (sm80+)."
        )
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-regs", action="store_true")
    parser.add_argument("--keep-cuda-artifacts", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Export a single Perfetto trace for one mpk() call.",
    )
    parser.add_argument(
        "--profiling-name",
        type=str,
        default="profile",
        help="Trace name used when exporting a Perfetto profile (requires --profile).",
    )
    parser.add_argument(
        "--bench-cublaslt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also benchmark a PyTorch GEMM with BLAS backend forced to cuBLASLt (linear kernel only).",
    )
    parser.add_argument(
        "--ncu-profile-mm",
        action="store_true",
        help="Wrap the cuBLASLt torch.mm() call with cudaProfilerStart/Stop and an NVTX range for Nsight Compute.",
    )
    parser.add_argument(
        "--ncu-profile-iters",
        type=int,
        default=1,
        help="Number of torch.mm() iterations inside the Nsight Compute profiling window.",
    )
    parser.add_argument(
        "--ncu-profile-only",
        action="store_true",
        help="Exit right after the Nsight Compute profiling window (skip further checks).",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="linear",
        choices=(
            "linear",
            "linear_with_residual",
            "rmsnorm",
            "embed",
            "silu_mul",
            "argmax_partial",
            "argmax_reduce",
            "paged_attention",
            "qwen3_full",
        ),
        help="Which kernel or Qwen3 graph to build.",
    )
    parser.add_argument("--num-workers", type=int, default=96)
    parser.add_argument("--num-schedulers", type=int, default=48)

    # Linear options.
    # Match the default (token, hidden) layout used by `demo/qwen3/demo_hopper.py`:
    # input:  [max_num_batched_tokens, hidden_size]  (default: 8 x 5120)
    # weight: [out_features, hidden_size]            (default: 10240 x 5120, i.e. fused QKV proj, 64 q_head and 8 kv_head)
    # output: [max_num_batched_tokens, out_features]
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=153600)
    parser.add_argument(
        "--has-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the *_with_residual_* linear task variant and compute y = x @ w.T + residual.",
    )

    # Paged attention options.
    parser.add_argument("--num-q-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--qwen3-layers", type=int, default=1)
    parser.add_argument("--argmax-tasks", type=int, default=4)

    parser.add_argument("--atol", type=float, default=1e-1, help="Absolute tolerance for --check.")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for --check.")
    parser.add_argument(
        "--check-fp32-ref",
        action="store_true",
        help="Compute reference matmul in fp32 (default).",
    )
    parser.add_argument(
        "--check-bf16-ref",
        action="store_true",
        help="Compute reference matmul in bf16 (closer to tensorcore behavior).",
    )
    args = parser.parse_args()

    if args.check_fp32_ref and args.check_bf16_ref:
        raise SystemExit("Pick at most one of --check-fp32-ref / --check-bf16-ref.")
    if not args.check_fp32_ref and not args.check_bf16_ref:
        args.check_bf16_ref = True

    if args.kernel in ("paged_attention", "qwen3_full"):
        if args.max_tokens > args.max_seq_len:
            raise SystemExit("--max-tokens must be <= --max-seq-len")
        if args.num_kv_heads <= 0 or args.num_q_heads % args.num_kv_heads != 0:
            raise SystemExit("--num-q-heads must be divisible by --num-kv-heads")
        if args.max_tokens >= args.page_size:
            raise SystemExit(
                "For this demo, pick --max-tokens < --page-size to avoid last_page_len=0."
            )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("MIRAGE_HOME", repo_root)

    try:
        import mirage as mi
    except ImportError as e:
        raise SystemExit(
            f"Failed to import mirage: {e}\n"
            f"Hint: build the native libs first (e.g. `cmake -S . -B build && cmake --build build -j`).\n"
            f"MIRAGE_HOME={os.environ.get('MIRAGE_HOME')}"
        )

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    if args.kernel in ("paged_attention", "qwen3_full") and cc < 80:
        raise SystemExit(
            f"--kernel={args.kernel} requires Ampere (sm80+), got cc={cc} ({props.name})."
        )
    if args.kernel in (
        "linear",
        "linear_with_residual",
        "rmsnorm",
        "embed",
        "silu_mul",
        "argmax_partial",
        "argmax_reduce",
    ) and cc < 80:
        raise SystemExit(
            f"--kernel={args.kernel} requires Ampere (sm80+), got cc={cc} ({props.name})."
        )

    if args.kernel in ("paged_attention", "qwen3_full"):
        max_seq_length = args.max_seq_len
        max_num_batched_requests = 1
        max_num_batched_tokens = args.max_tokens
        page_size = args.page_size
        max_num_pages = (max_seq_length + page_size - 1) // page_size
        if max_num_pages < 4:
            page_size = max(1, max_seq_length // 4)
            max_num_pages = (max_seq_length + page_size - 1) // page_size
            if args.max_tokens >= page_size:
                raise SystemExit(
                    f"--max-tokens must be < page_size; auto page_size={page_size}"
                )
        prompt_len = args.max_tokens
    else:
        # Minimal values to satisfy PersistentKernel ABI.
        max_num_pages = 1
        page_size = 4096
        max_seq_length = 1
        max_num_batched_requests = 1
        max_num_batched_tokens = max(1, args.batch_size)
        prompt_len = max(1, args.batch_size)

    meta_tensors = _make_minimal_persistent_kernel_meta(
        max_seq_length=max_seq_length,
        max_num_pages=max_num_pages,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    # Offline mode only runs the task graph if `prepare_next_batch()` returns
    # true at the first END_OF_TASK_GRAPH event. With all-zero meta tensors,
    # num_tokens becomes 0 and the scheduler terminates immediately, leaving the
    # worker spinning in the fetch/control path (and spamming MIRAGE_ADMISSION_DEBUG logs).
    meta_tensors["prompt_lengths"][0] = prompt_len
    meta_tensors["tokens"][0, 0] = 0
    if args.kernel in ("linear", "linear_with_residual", "rmsnorm", "embed", "silu_mul"):
        # Ampere linear task uses `qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]`
        # as `num_active_tokens` inside the kernel. The minimal meta tensors are
        # all-zero by default, which makes the kernel skip all output writes
        # and leave `y` as zeros.
        _set_num_active_tokens(meta_tensors, max_num_batched_requests, args.batch_size)
    elif args.kernel in ("argmax_partial", "argmax_reduce"):
        _set_num_active_tokens(meta_tensors, max_num_batched_requests, args.batch_size)
    elif args.kernel in ("paged_attention", "qwen3_full"):
        # One request (id=0), prefill-only, single page.
        num_pages = (args.max_tokens + page_size - 1) // page_size
        meta_tensors["qo_indptr_buffer"].zero_()
        meta_tensors["qo_indptr_buffer"][0] = 0
        meta_tensors["qo_indptr_buffer"][1] = args.max_tokens
        meta_tensors["paged_kv_indptr_buffer"].zero_()
        meta_tensors["paged_kv_indptr_buffer"][0] = 0
        meta_tensors["paged_kv_indptr_buffer"][1] = num_pages
        meta_tensors["paged_kv_indices_buffer"].zero_()
        meta_tensors["paged_kv_indices_buffer"][:num_pages] = torch.arange(
            num_pages, device="cuda", dtype=torch.int32
        )
        meta_tensors["paged_kv_last_page_len_buffer"].zero_()
        meta_tensors["paged_kv_last_page_len_buffer"][0] = args.max_tokens
    # NOTE: For the split worker/scheduler runtime, keep this demo to a single
    # scheduler CTA to avoid multi-scheduler races in minimal setups.
    num_workers, num_schedulers = args.num_workers, args.num_schedulers
    # num_workers, num_schedulers = mi.get_configurations_from_gpu(0)

    print(f'num_workers: {num_workers}, num_schedulers: {num_schedulers}')
    profiler_tensor = None
    if args.profiling:
        # Hopper worker kernel uses 2 warp-groups per CTA (blockDim=2*WORKER_NUM_THREADS).
        num_groups = 2 if cc == 90 else 1
        stride = num_workers * num_groups
        profile_rows = 3000
        profiler_tensor = torch.zeros(
            1 + stride * profile_rows, dtype=torch.uint64, device="cuda"
        ).contiguous()

    mpk = mi.PersistentKernel(
        mode="onepass",
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        # For correctness checking, stop after the first END_OF_TASK_GRAPH.
        # With eos_token_id=-1 the offline runtime can keep decoding until
        # max_seq_length, overwriting outputs across iterations.
        eos_token_id=0 if args.kernel in ("paged_attention", "qwen3_full") else -1,
        meta_tensors=meta_tensors,
        profiler_tensor=profiler_tensor,
        trace_name=args.profiling_name if args.profiling else "",
        spec_decode_config=None,
        use_cutlass_kernel=False,
    )

    if args.kernel == "paged_attention":
        num_qo_per_kv = args.num_q_heads // args.num_kv_heads
        q_cols_per_kv = num_qo_per_kv * args.head_dim
        qkv_cols = args.num_kv_heads * (q_cols_per_kv + 2 * args.head_dim)

        qkv_t = torch.randn((args.max_tokens, qkv_cols), device="cuda", dtype=torch.bfloat16)
        k_cache_t = torch.zeros(
            (max_num_pages, page_size, args.num_kv_heads, args.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_cache_t = torch.zeros_like(k_cache_t)
        q_norm_t = torch.ones((args.head_dim,), device="cuda", dtype=torch.bfloat16)
        k_norm_t = torch.ones((args.head_dim,), device="cuda", dtype=torch.bfloat16)
        if args.rope:
            cos_t, sin_t = _make_rope_tables(max_seq_length, args.head_dim, "cuda")
        else:
            cos_t = torch.zeros((max_seq_length, args.head_dim), device="cuda", dtype=torch.bfloat16)
            sin_t = torch.zeros_like(cos_t)
        out_t = torch.empty(
            (args.max_tokens, args.num_q_heads * args.head_dim), device="cuda", dtype=torch.bfloat16
        )

        qkv = mpk.attach_input(qkv_t, name="qkv")
        k_cache = mpk.attach_input(k_cache_t, name="k_cache")
        v_cache = mpk.attach_input(v_cache_t, name="v_cache")
        q_norm = mpk.attach_input(q_norm_t, name="q_norm")
        k_norm = mpk.attach_input(k_norm_t, name="k_norm")
        cos = mpk.attach_input(cos_t, name="cos")
        sin = mpk.attach_input(sin_t, name="sin")
        out = mpk.attach_input(out_t, name="out")

        params = [
            args.num_q_heads,
            args.num_kv_heads,
            1 if args.qk_norm else 0,
            1 if args.rope else 0,
            max_seq_length,
            page_size,
        ]
        tb_graph = mi.TBGraph(mi.CyTBGraph((1, args.num_kv_heads, 1), (256, 1, 1), 1, 64))
        tb_graph.new_input(qkv, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin, (-1, -1, -1), -1, True)
        tb_graph.new_input(out, (-1, 1, -1), -1, True)
        mpk.kn_graph.customized([qkv, k_cache, v_cache, q_norm, k_norm, cos, sin, out], tb_graph)
        mpk.kn_graph.register_task(
            tb_graph, "paged_attention_hopper" if cc >= 90 else "paged_attention", params
        )
    elif args.kernel in ("linear", "linear_with_residual"):
        x_t = torch.randn((args.batch_size, args.in_features), device="cuda", dtype=torch.bfloat16)
        w_t = torch.randn((args.out_features, args.in_features), device="cuda", dtype=torch.bfloat16)
        y_t = torch.empty((args.batch_size, args.out_features), device="cuda", dtype=torch.bfloat16)
        residual_t = None
        if args.has_residual or args.kernel == "linear_with_residual":
            residual_t = torch.randn(
                (args.batch_size, args.out_features), device="cuda", dtype=torch.bfloat16
            )

        x = mpk.attach_input(x_t, name="x")
        w = mpk.attach_input(w_t, name="w")
        y = mpk.attach_input(y_t, name="y")
        residual = mpk.attach_input(residual_t, name="residual") if residual_t is not None else None

        # grid_x = grid_for_rmsnorm_linear_layer(args.out_features)
        grid_x = args.out_features // 256
        split_batch_size = max(args.batch_size // 16, 1)
        if args.out_features % grid_x != 0:
            raise SystemExit(
                f"linear grid mismatch: out_features={args.out_features} not divisible by grid_x={grid_x}"
            )
        print(f"[linear] grid_dim=({split_batch_size}, {grid_x}, 1) tasks={split_batch_size * grid_x} out_per_task={args.out_features // grid_x}")

        block_x = 256 if cc >= 90 else 128
        if args.kernel == "linear_with_residual" or args.has_residual:
            mpk.linear_with_residual_layer(
                input=x,
                weight=w,
                residual=residual,
                output=y,
                grid_dim=(split_batch_size, grid_x, 1),
                block_dim=(block_x, 1, 1),
            )
        else:
            mpk.linear_layer(
                input=x,
                weight=w,
                output=y,
                grid_dim=(grid_x, split_batch_size, 1),
                block_dim=(block_x, 1, 1),
            )
    elif args.kernel == "rmsnorm":
        x_t = torch.randn((args.batch_size, args.hidden_size), device="cuda", dtype=torch.bfloat16)
        w_t = torch.randn((args.hidden_size,), device="cuda", dtype=torch.bfloat16)
        y_t = torch.empty((args.batch_size, args.hidden_size), device="cuda", dtype=torch.bfloat16)
        x = mpk.attach_input(x_t, name="x")
        w = mpk.attach_input(w_t, name="w")
        y = mpk.attach_input(y_t, name="y")
        mpk.rmsnorm_layer(
            input=x,
            weight=w,
            output=y,
            grid_dim=(args.batch_size, 1, 1),
            block_dim=(128 if cc < 90 else 256, 1, 1),
        )
    elif args.kernel == "embed":
        input_tokens_t = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, 1),
            device="cuda",
            dtype=torch.int64,
        )
        meta_tensors["input_tokens"].copy_(input_tokens_t)
        weight_t = torch.randn((args.vocab_size, args.hidden_size), device="cuda", dtype=torch.bfloat16)
        out_t = torch.empty((args.batch_size, args.hidden_size), device="cuda", dtype=torch.bfloat16)
        input_tokens = mpk.attach_input(meta_tensors["input_tokens"], name="input_tokens")
        weight = mpk.attach_input(weight_t, name="weight")
        out = mpk.attach_input(out_t, name="out")
        mpk.embed_layer(
            input=input_tokens,
            weight=weight,
            output=out,
            grid_dim=(1, 1, 1),
            block_dim=(128 if cc < 90 else 256, 1, 1),
            input_source=1,
        )
    elif args.kernel == "silu_mul":
        in_features = args.intermediate_size * 2
        x_t = torch.randn((args.batch_size, in_features), device="cuda", dtype=torch.bfloat16)
        y_t = torch.empty((args.batch_size, in_features // 2), device="cuda", dtype=torch.bfloat16)
        x = mpk.attach_input(x_t, name="x")
        y = mpk.attach_input(y_t, name="y")
        mpk.silu_mul_layer(
            input=x,
            output=y,
            grid_dim=(1, 1, 1),
            block_dim=(128 if cc < 90 else 256, 1, 1),
        )
    elif args.kernel == "argmax_partial":
        if args.vocab_size % args.argmax_tasks != 0:
            raise SystemExit("--vocab-size must be divisible by --argmax-tasks for argmax_partial")
        chunk = args.vocab_size // args.argmax_tasks
        x_t = torch.zeros((args.batch_size, args.vocab_size), device="cuda", dtype=torch.bfloat16)
        for task in range(args.argmax_tasks):
            x_t[:, task * chunk] = float(task + 1)
        out_val_t = torch.empty((args.batch_size, args.argmax_tasks), device="cuda", dtype=torch.bfloat16)
        out_idx_t = torch.empty((args.batch_size, args.argmax_tasks), device="cuda", dtype=torch.int64)
        x = mpk.attach_input(x_t, name="x")
        out_val = mpk.attach_input(out_val_t, name="out_val")
        out_idx = mpk.attach_input(out_idx_t, name="out_idx")
        mpk.argmax_partial_layer(
            input=x,
            output=(out_val, out_idx),
            grid_dim=(args.argmax_tasks, 1, 1),
            block_dim=(128 if cc < 90 else 256, 1, 1),
        )
    elif args.kernel == "argmax_reduce":
        if args.vocab_size % args.argmax_tasks != 0:
            raise SystemExit("--vocab-size must be divisible by --argmax-tasks for argmax_reduce")
        chunk = args.vocab_size // args.argmax_tasks
        x_t = torch.zeros((args.batch_size, args.vocab_size), device="cuda", dtype=torch.bfloat16)
        for task in range(args.argmax_tasks):
            x_t[:, task * chunk] = float(task + 1)
        partial_vals = torch.empty((args.batch_size, args.argmax_tasks), device="cuda", dtype=torch.bfloat16)
        partial_idx = torch.empty((args.batch_size, args.argmax_tasks), device="cuda", dtype=torch.int64)
        x_f = x_t.float()
        for task in range(args.argmax_tasks):
            start = task * chunk
            end = start + chunk
            vals, idxs = x_f[:, start:end].max(dim=1)
            partial_vals[:, task] = vals.to(torch.bfloat16)
            partial_idx[:, task] = idxs.to(torch.int64)
        out_t = torch.empty((args.batch_size, 1), device="cuda", dtype=torch.int64)
        in_val = mpk.attach_input(partial_vals, name="partial_vals")
        in_idx = mpk.attach_input(partial_idx, name="partial_idx")
        out = mpk.attach_input(out_t, name="out")
        mpk.argmax_partial_output_size = chunk
        mpk.argmax_reduce_layer(
            input=(in_val, in_idx),
            output=out,
            grid_dim=(1, 1, 1),
            block_dim=(128 if cc < 90 else 256, 1, 1),
        )

    args.output_dir = args.output_dir if args.output_dir is not None else os.getcwd()

    mpk.compile(
        output_dir=args.output_dir,
        report_regs=args.report_regs,
        keep_cuda_artifacts=args.keep_cuda_artifacts,
        noinline_task_wrappers=True,
    )

    if args.profiling:
        torch.cuda.synchronize()
        mpk()
        print(f"[profile] exported trace name={args.profiling_name!r}")
        # Disable exporting during warmup/bench loops (keeps kernel instrumentation).
        mpk.profiler_tensor = None

    if args.warmup_iters < 0 or args.bench_iters <= 0:
        raise SystemExit("--warmup-iters must be >= 0 and --bench-iters must be >= 1")
    if args.ncu_profile_mm and args.ncu_profile_iters <= 0:
        raise SystemExit("--ncu-profile-iters must be >= 1")

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.warmup_iters):
        mpk()
    # Do not reassign output tensors after `attach_input(...)`: the persistent
    # kernel is still wired to the originally attached storage. If you want a
    # clean output buffer between warmup and benchmark, reset it in-place.
    if args.kernel == "paged_attention":
        out_t.zero_()
    elif args.kernel == "embed":
        out_t.zero_()
    elif args.kernel == "argmax_partial":
        out_val_t.zero_()
        out_idx_t.zero_()
    elif args.kernel == "argmax_reduce":
        out_t.zero_()
    elif args.kernel == "qwen3_full":
        logits_t.zero_()
        output_tokens_t.zero_()
    else:
        y_t.zero_()
    torch.cuda.synchronize()
    starter.record()
    for _ in range(args.bench_iters):
        mpk()
    torch.cuda.synchronize()
    ender.record()
    ender.synchronize()
    run_time_ms = starter.elapsed_time(ender)
    mpk_avg_ms = run_time_ms / args.bench_iters
    print(f"[timing] mpk() total={run_time_ms:.3f} ms iters={args.bench_iters} avg={mpk_avg_ms:.3f} ms")

    # cuBLASLt baseline timing (PyTorch GEMM). Keep synchronization minimal:
    # - one sync after warmup
    # - one event sync after the benchmark loop
    if args.kernel == "linear" and args.bench_cublaslt:
        import torch.backends.cuda as cuda_backends

        prev_blas = cuda_backends.preferred_blas_library()
        # cuda_backends.preferred_blas_library("cublaslt")
        try:
            y_cublas = torch.empty(
                (args.batch_size, args.out_features), device="cuda", dtype=torch.bfloat16
            )
            residual_cublas = residual_t if args.has_residual else None

            w_t_t = w_t.t()
            for _ in range(args.warmup_iters):
                torch.mm(x_t, w_t_t, out=y_cublas)
                if residual_cublas is not None:
                    y_cublas.add_(residual_cublas)
            torch.cuda.synchronize()

            if args.ncu_profile_mm:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("cublaslt_mm")
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(args.ncu_profile_iters):
                    torch.mm(x_t, w_t_t, out=y_cublas)
                    if residual_cublas is not None:
                        y_cublas.add_(residual_cublas)
                torch.cuda.cudart().cudaProfilerStop()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                print(f"[ncu] profiled torch.mm iters={args.ncu_profile_iters}")
                if args.ncu_profile_only:
                    return

            starter.record()
            for _ in range(args.bench_iters):
                torch.mm(x_t, w_t_t, out=y_cublas)
                if residual_cublas is not None:
                    y_cublas.add_(residual_cublas)
            torch.cuda.synchronize()
            ender.record()
            ender.synchronize()
            cublaslt_ms = starter.elapsed_time(ender)
            cublaslt_avg_ms = cublaslt_ms / args.bench_iters
            ratio = cublaslt_avg_ms / mpk_avg_ms if cublaslt_avg_ms > 0 else float("inf")
            print(
                f"[timing] cublaslt(mm) total={cublaslt_ms:.3f} ms iters={args.bench_iters} avg={cublaslt_avg_ms:.3f} ms speedup(cublaslt/mpk)={ratio:.3f}x"
            )
        finally:
            cuda_backends.preferred_blas_library(prev_blas)

    if args.kernel == "paged_attention":
        # Correctness check (this demo's prefill-only setup).
        ref = _reference_paged_attention_causal_prefill(
            qkv=qkv_t,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            qk_norm=args.qk_norm,
            rope=args.rope,
            q_norm_weight=q_norm_t,
            k_norm_weight=k_norm_t,
            cos=cos_t,
            sin=sin_t,
        )
        out = out_t.float()
        diff = out - ref
        abs_err = diff.abs()
        max_abs = abs_err.max().item()
        mean_abs = abs_err.mean().item()
        denom = ref.abs().clamp_min(1e-6)
        max_rel = (abs_err / denom).max().item()
        ok = torch.allclose(out, ref, atol=args.atol, rtol=args.rtol)
        print(
            "check="
            f"{'PASS' if ok else 'FAIL'} "
            f"atol={args.atol} rtol={args.rtol} "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} max_rel={max_rel:.6g} "
            f"out[0,0]={out_t[0,0].item()}"
        )
        return

    if args.kernel in ("linear", "linear_with_residual"):
        if args.check_bf16_ref:
            ref = x_t @ w_t.t()
            if residual_t is not None:
                ref = ref + residual_t
            ref = ref.float()
        else:
            ref = x_t.float() @ w_t.float().t()
            if residual_t is not None:
                ref = ref + residual_t.float()
        _check_allclose("linear", y_t, ref, args.atol, args.rtol)
    elif args.kernel == "rmsnorm":
        ref = _reference_rmsnorm(x_t, w_t)
        _check_allclose("rmsnorm", y_t, ref, args.atol, args.rtol)
    elif args.kernel == "embed":
        ref = weight_t[input_tokens_t[:, 0]].float()
        _check_allclose("embed", out_t, ref, args.atol, args.rtol)
    elif args.kernel == "silu_mul":
        ref = _reference_silu_mul(x_t)
        _check_allclose("silu_mul", y_t, ref, args.atol, args.rtol)
    elif args.kernel == "argmax_partial":
        chunk = args.vocab_size // args.argmax_tasks
        x_f = x_t.float()
        ref_val = torch.empty_like(out_val_t, dtype=torch.float32)
        ref_idx = torch.empty_like(out_idx_t, dtype=torch.int64)
        for task in range(args.argmax_tasks):
            start = task * chunk
            end = start + chunk
            vals, idxs = x_f[:, start:end].max(dim=1)
            ref_val[:, task] = vals
            ref_idx[:, task] = idxs
        _check_allclose("argmax_partial_val", out_val_t, ref_val, args.atol, args.rtol)
        if torch.equal(out_idx_t.cpu(), ref_idx.cpu()):
            print("argmax_partial_idx check=PASS")
        else:
            print("argmax_partial_idx check=FAIL")
    elif args.kernel == "argmax_reduce":
        chunk = args.vocab_size // args.argmax_tasks
        x_f = x_t.float()
        ref_tokens = torch.argmax(x_f, dim=1)
        out_f = out_t[:, 0].to(torch.int64)
        if torch.equal(out_f.cpu(), ref_tokens.cpu()):
            print("argmax_reduce check=PASS")
        else:
            print("argmax_reduce check=FAIL")
    elif args.kernel == "qwen3_full":
        with torch.no_grad():
            ref_embed = embed_w_t[input_tokens_t[:, 0]].float()
            ref_rms1 = _reference_rmsnorm(ref_embed.to(torch.bfloat16), norm_1)
            ref_qkv = (ref_rms1.to(torch.bfloat16) @ w_qkv_t.t()).to(torch.bfloat16).float()
            ref_attn = _reference_paged_attention_causal_prefill(
                qkv=ref_qkv.to(torch.bfloat16),
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                qk_norm=args.qk_norm,
                rope=args.rope,
                q_norm_weight=q_norm_t,
                k_norm_weight=k_norm_t,
                cos=cos_t,
                sin=sin_t,
            )
            ref_attn_proj = (
                (ref_attn.to(torch.bfloat16) @ w_o_t.t()).to(torch.bfloat16).float() + ref_embed
            ).to(torch.bfloat16).float()
            ref_rms2 = _reference_rmsnorm(ref_attn_proj.to(torch.bfloat16), norm_2)
            ref_mlp_mid = (ref_rms2.to(torch.bfloat16) @ w_gatedup_t.t()).to(torch.bfloat16).float()
            ref_silu = _reference_silu_mul(ref_mlp_mid.to(torch.bfloat16))
            ref_mlp_out = (
                (ref_silu.to(torch.bfloat16) @ w_down_t.t()).to(torch.bfloat16).float()
                + ref_attn_proj
            ).to(torch.bfloat16).float()
            ref_rms3 = _reference_rmsnorm(ref_mlp_out.to(torch.bfloat16), norm_3)
            ref_logits = (ref_rms3.to(torch.bfloat16) @ w_lm_t.t()).to(torch.bfloat16).float()
            ref_argmax = torch.argmax(ref_logits, dim=1)

        qwen_atol = max(args.atol, 1.0)
        _check_allclose("qwen3_logits", logits_t, ref_logits, qwen_atol, args.rtol)
        out_tokens = output_tokens_t[:, 0].to(torch.int64)
        if torch.equal(out_tokens.cpu(), ref_argmax.cpu()):
            print("qwen3_argmax check=PASS")
        else:
            print("qwen3_argmax check=FAIL")


if __name__ == "__main__":
    main()
