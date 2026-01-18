#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


LAT_RE = re.compile(
    r"Prompt length\s+(?P<prompt>\d+),\s*generate length\s+(?P<gen>\d+),\s*"
    r"per-token latency \(both prefill and decode\):\s*(?P<ms>[0-9.]+)\s*ms"
)


def _parse_int_set(spec: str) -> List[int]:
    """
    Parse a comma list of ints and/or ranges:
      - "16,32,64"
      - "16:65:16" (start:end:step, end inclusive if hits exactly)
      - "16:64"    (start:end, step=1)
    """
    out: List[int] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if ":" not in part:
            out.append(int(part))
            continue
        fields = part.split(":")
        if len(fields) not in (2, 3):
            raise ValueError(f"bad range {part!r} (expected start:end[:step])")
        start = int(fields[0])
        end = int(fields[1])
        step = int(fields[2]) if len(fields) == 3 else 1
        if step == 0:
            raise ValueError(f"bad range {part!r} (step=0)")
        if (end - start) * step < 0:
            raise ValueError(f"bad range {part!r} (step sign)")
        v = start
        if step > 0:
            while v <= end:
                out.append(v)
                v += step
        else:
            while v >= end:
                out.append(v)
                v += step
    if not out:
        raise ValueError("empty integer set")
    return out


def _ensure_command_exists(cmd: str) -> None:
    from shutil import which

    if which(cmd) is None:
        raise RuntimeError(f"required command not found in PATH: {cmd!r}")


def _run(
    argv: Sequence[str],
    *,
    cwd: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _conda_run(
    env_name: str,
    cmd_argv: Sequence[str],
    *,
    cwd: str,
    timeout_s: Optional[int],
) -> subprocess.CompletedProcess:
    # Prefer `conda run` (non-interactive, no shell init).
    return _run(
        ["conda", "run", "-n", env_name, "--no-capture-output", *cmd_argv],
        cwd=cwd,
        timeout_s=timeout_s,
    )


def _conda_python_eval(
    env_name: str,
    code: str,
    *,
    cwd: str,
    timeout_s: Optional[int],
) -> subprocess.CompletedProcess:
    return _conda_run(env_name, ["python", "-c", code], cwd=cwd, timeout_s=timeout_s)


def _make_prompt_in_env(
    *,
    env_name: str,
    repo_dir: str,
    model_name: str,
    target_tokens: int,
    seed_text: str = "hello",
) -> str:
    # Generate a prompt string whose tokenized length is exactly target_tokens,
    # by repeating a seed token, truncating, then decoding.
    code = r"""
import os
from transformers import AutoTokenizer

model = os.environ["MIRAGE_BENCH_MODEL"]
target = int(os.environ["MIRAGE_BENCH_PROMPT_TOKENS"])
seed = os.environ.get("MIRAGE_BENCH_SEED", "hello")

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ids = []
chunk = tok.encode(seed, add_special_tokens=False)
if not chunk:
    chunk = tok.encode("hello", add_special_tokens=False)
if not chunk:
    raise SystemExit("tokenizer produced empty encoding")
while len(ids) < target:
    ids.extend(chunk)
ids = ids[:target]
text = tok.decode(ids, skip_special_tokens=True)
print(text)
"""
    env = dict(os.environ)
    env["MIRAGE_BENCH_MODEL"] = model_name
    env["MIRAGE_BENCH_PROMPT_TOKENS"] = str(target_tokens)
    env["MIRAGE_BENCH_SEED"] = seed_text
    cp = subprocess.run(
        ["conda", "run", "-n", env_name, "--no-capture-output", "python", "-c", code],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"prompt generation failed (env={env_name} model={model_name} tokens={target_tokens}):\n{cp.stdout}"
        )
    return cp.stdout.strip()


@dataclass
class BenchResult:
    env: str
    repo: str
    model: str
    target_prompt_tokens: int
    target_gen_tokens: int
    max_seq_len: int
    prompt_len: Optional[int]
    gen_len: Optional[int]
    per_token_ms: Optional[float]
    prompt_text: str
    generated_text: Optional[str]
    wall_s: float
    returncode: int
    matched_line: Optional[str]
    output_line: Optional[str]
    output_tail: str


def _tail(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _extract_latency(stdout: str) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[str]]:
    matches = list(LAT_RE.finditer(stdout))
    if not matches:
        return None, None, None, None
    m = matches[-1]
    return (
        int(m.group("prompt")),
        int(m.group("gen")),
        float(m.group("ms")),
        m.group(0),
    )


def _extract_response(stdout: str) -> Optional[str]:
    """
    demo/qwen3/demo.py prints the decoded text as a standalone line right before
    the latency line. Heuristically capture the last non-empty line before the
    latency line.
    """
    lines = [ln.strip() for ln in stdout.splitlines()]
    if not lines:
        return None
    latency_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if LAT_RE.search(lines[i]):
            latency_idx = i
            break
    if latency_idx is None:
        return None
    for j in range(latency_idx - 1, -1, -1):
        if lines[j]:
            return lines[j]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark demo/qwen3/demo.py --use-mirage across two conda envs and repos.\n"
            "Defaults:\n"
            "  neutrino  -> /root/mirage\n"
            "  baseline  -> /root/test/mirage\n"
        )
    )
    parser.add_argument(
        "--prompt-lens",
        default="128,256,512,1024,2048,4096",
        help="Prompt token lengths, e.g. '32,64,128' or '32:256:32'",
    )
    parser.add_argument(
        "--gen-lens",
        default="256,512,1024",
        help="Generate lengths (max-new-tokens), e.g. '64,128' or '64:512:64'",
    )
    parser.add_argument(
        "--models",
        # default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B",
        default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-8B",
        help="Comma-separated model names passed to demo.py --model (e.g. 'Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B')",
    )
    parser.add_argument(
        "--envs",
        default="neutrino,baseline",
        help="Comma-separated conda env names to test (default: neutrino,baseline)",
    )
    parser.add_argument(
        "--neutrino-repo",
        default="/root/mirage",
        help="Repo dir for env neutrino (default: /root/mirage)",
    )
    parser.add_argument(
        "--baseline-repo",
        default="/root/test/mirage",
        help="Repo dir for env baseline (default: /root/test/mirage)",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="Per-run timeout seconds (default: 1800)",
    )
    parser.add_argument(
        "--out",
        default="outputs/qwen3/demo_bench_summary",
        help="Output prefix (writes .csv, .json, .md) relative to cwd (default: outputs/qwen3/demo_bench_summary)",
    )
    parser.add_argument(
        "--seed-text",
        default="hello",
        help="Seed text repeated to synthesize prompts (default: 'hello')",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra args appended to demo.py invocation (shell-escaped string)",
    )
    args = parser.parse_args()

    _ensure_command_exists("conda")

    prompt_lens = _parse_int_set(args.prompt_lens)
    gen_lens = _parse_int_set(args.gen_lens)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    if not models:
        raise SystemExit("--models is empty")

    repo_for_env = {
        "neutrino": args.neutrino_repo,
        "baseline": args.baseline_repo,
    }

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    results: List[BenchResult] = []

    for env_name in envs:
        repo_dir = repo_for_env.get(env_name, None)
        if repo_dir is None:
            raise SystemExit(f"unknown env {env_name!r}; expected one of: {', '.join(repo_for_env)}")
        if not Path(repo_dir).exists():
            raise SystemExit(f"repo dir does not exist: {repo_dir}")

        for model_name in models:
            for p_len in prompt_lens:
                prompt_text = _make_prompt_in_env(
                    env_name=env_name,
                    repo_dir=repo_dir,
                    model_name=model_name,
                    target_tokens=p_len,
                    seed_text=args.seed_text,
                )
                for g_len in gen_lens:
                    # demo.py uses args.max_seq_length to allocate token buffer.
                    max_seq_len = p_len + g_len + 8
                    cmd = [
                        "python",
                        "demo/qwen3/demo.py",
                        "--use-mirage",
                        "--model",
                        model_name,
                        "--prompt",
                        prompt_text,
                        "--max-seq-length",
                        str(max_seq_len),
                        "--max-new-tokens",
                        str(g_len),
                        "--ignore-eos",
                    ]
                    if args.extra_args.strip():
                        cmd += shlex.split(args.extra_args)

                    t0 = time.time()
                    try:
                        cp = _conda_run(env_name, cmd, cwd=repo_dir, timeout_s=args.timeout_s)
                        timed_out = False
                    except subprocess.TimeoutExpired as e:
                        cp = subprocess.CompletedProcess(
                            args=e.cmd,
                            returncode=124,
                            stdout=(e.stdout or "") + "\n[TIMEOUT]\n" + (e.stderr or ""),
                        )
                        timed_out = True
                    wall_s = time.time() - t0

                    prompt_len, gen_len, per_tok_ms, matched = _extract_latency(cp.stdout or "")
                    output_line = matched
                    response_text = _extract_response(cp.stdout or "")
                    results.append(
                        BenchResult(
                            env=env_name,
                            repo=repo_dir,
                            model=model_name,
                            target_prompt_tokens=p_len,
                            target_gen_tokens=g_len,
                            max_seq_len=max_seq_len,
                            prompt_len=prompt_len,
                            gen_len=gen_len,
                            per_token_ms=per_tok_ms,
                            prompt_text=prompt_text,
                            generated_text=response_text,
                            wall_s=wall_s,
                            returncode=cp.returncode,
                            matched_line=matched,
                            output_line=output_line,
                            output_tail=_tail(cp.stdout or "", 120),
                        )
                    )

                    status = "OK" if (cp.returncode == 0 and matched) else "FAIL"
                    if timed_out:
                        status = "TIMEOUT"
                    print(
                        f"[{status}] env={env_name} model={model_name} prompt={p_len} gen={g_len} "
                        f"ms={per_tok_ms if per_tok_ms is not None else 'NA'} wall={wall_s:.1f}s"
                    )

    # Write CSV + JSON.
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(results[0]).keys()) if results else [],
        )
        if results:
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    with json_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Render a compact markdown table focused on the requested line content.
    md_lines = [
        "# demo/qwen3/demo.py benchmark summary",
        "",
        "| env | model | target_prompt | target_gen | prompt_len | gen_len | per_token_ms |",
        "| --- | ----- | ------------: | ---------: | ---------: | ------: | ----------: |",
    ]
    for r in results:
        md_lines.append(
            "| {env} | {model} | {tp} | {tg} | {pl} | {gl} | {ms} |".format(
                env=r.env,
                model=r.model,
                tp=r.target_prompt_tokens,
                tg=r.target_gen_tokens,
                pl=r.prompt_len if r.prompt_len is not None else "",
                gl=r.gen_len if r.gen_len is not None else "",
                ms=f"{r.per_token_ms:.3f}" if r.per_token_ms is not None else "",
            )
        )
    md_lines.append("")
    md_lines.append("## Detailed Outputs")
    md_lines.append("")
    for r in results:
        title = (
            f"env={r.env} model={r.model} "
            f"target_prompt={r.target_prompt_tokens} target_gen={r.target_gen_tokens}"
        )
        md_lines.append(f"### {title}")
        if r.output_line:
            md_lines.append("")
            md_lines.append(f"Latency: `{r.output_line}`")
        md_lines.append("")
        md_lines.append("Prompt:")
        md_lines.append("```")
        md_lines.append(r.prompt_text)
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("Response:")
        md_lines.append("```")
        md_lines.append(r.generated_text or "")
        md_lines.append("```")
        md_lines.append("")
    md_lines.append(f"Raw outputs: `{csv_path}`, `{json_path}`")
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
