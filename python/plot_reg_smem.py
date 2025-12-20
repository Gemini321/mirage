#!/usr/bin/env python3
import re

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: matplotlib. Install it with `pip install matplotlib` "
        "or `conda install -c conda-forge matplotlib`."
    ) from e


# RAW = """
# __mirage_regprobe_TASK_PAGED_ATTENTION_1_v0_dsmem139312: regs=255; smem_dynamic_used=139312; smem_static=16; smem_total=139328
# __mirage_regprobe_TASK_LINEAR_v2_dsmem56448: regs=96; smem_dynamic_used=56448; smem_static=0; smem_total=56448
# __mirage_regprobe_TASK_LINEAR_v1_dsmem56448: regs=94; smem_dynamic_used=56448; smem_static=0; smem_total=56448
# __mirage_regprobe_TASK_LINEAR_WITH_RESIDUAL_v0_dsmem56448: regs=72; smem_dynamic_used=56448; smem_static=0; smem_total=56448
# __mirage_regprobe_TASK_LINEAR_WITH_RESIDUAL_v1_dsmem56448: regs=72; smem_dynamic_used=56448; smem_static=0; smem_total=56448
# __mirage_regprobe_TASK_LINEAR_v0_dsmem56448: regs=64; smem_dynamic_used=56448; smem_static=0; smem_total=56448
# __mirage_regprobe_TASK_ARGMAX_PARTIAL_v0_dsmem448: regs=40; smem_dynamic_used=448; smem_static=0; smem_total=448
# __mirage_regprobe_TASK_ARGMAX_REDUCE_v0_dsmem448: regs=36; smem_dynamic_used=448; smem_static=0; smem_total=448
# __mirage_regprobe_TASK_RMS_NORM_v0_dsmem24592: regs=28; smem_dynamic_used=24592; smem_static=0; smem_total=24592
# __mirage_regprobe_TASK_EMBEDDING_v0_dsmem0: regs=19; smem_dynamic_used=0; smem_static=0; smem_total=0
# __mirage_regprobe_TASK_SILU_MUL_v0_dsmem0: regs=18; smem_dynamic_used=0; smem_static=0; smem_total=0
# """.strip()

RAW = """
__mirage_regprobe_TASK_PAGED_ATTENTION_HOPPER_v0_dsmem136288 regs=188 smem_dynamic_used=136288 smem_static=1056 smem_total=137344
__mirage_regprobe_TASK_ARGMAX_PARTIAL_SM100_v0_dsmem447 regs=32 smem_dynamic_used=447 smem_static=1040 smem_total=1487
__mirage_regprobe_TASK_ARGMAX_REDUCE_v0_dsmem447 regs=32 smem_dynamic_used=447 smem_static=1040 smem_total=1487
__mirage_regprobe_TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER_v0_dsmem97440 regs=32 smem_dynamic_used=97440 smem_static=1040 smem_total=98480
__mirage_regprobe_TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER_v1_dsmem97440 regs=32 smem_dynamic_used=97440 smem_static=1040 smem_total=98480
__mirage_regprobe_TASK_LINEAR_SWAPAB_HOPPER_v1_dsmem102600 regs=31 smem_dynamic_used=102600 smem_static=1040 smem_total=103640
__mirage_regprobe_TASK_LINEAR_SWAPAB_HOPPER_v2_dsmem102600 regs=31 smem_dynamic_used=102600 smem_static=1040 smem_total=103640
__mirage_regprobe_TASK_LINEAR_SWAPAB_HOPPER_v0_dsmem102600 regs=30 smem_dynamic_used=102600 smem_static=1040 smem_total=103640
__mirage_regprobe_TASK_RMS_NORM_HOPPER_v0_dsmem24608 regs=30 smem_dynamic_used=24608 smem_static=1040 smem_total=25648
__mirage_regprobe_TASK_EMBEDDING_v0_dsmem0 regs=20 smem_dynamic_used=0 smem_static=1040 smem_total=1040
__mirage_regprobe_TASK_SILU_MUL_v0_dsmem0 regs=19 smem_dynamic_used=0 smem_static=1040 smem_total=1040
""".strip()

def clean_name(name: str) -> str:
    name = re.sub(r"^__mirage_regprobe_", "", name)
    name = re.sub(r"^TASK_", "", name)
    name = re.sub(r"_dsmem\d+$", "", name)
    return name


def parse(raw: str):
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        # Accept both formats:
        # 1) "__mirage_regprobe_xxx: regs=..; smem_total=.."
        # 2) "__mirage_regprobe_xxx regs=.. smem_total=.."
        name_part = line.split(":", 1)[0].split(None, 1)[0]
        kv = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        if "regs" not in kv or "smem_total" not in kv:
            raise ValueError(f"Unrecognized line: {line!r}")
        rows.append(
            {
                "name": clean_name(name_part),
                "regs": kv["regs"],
                "smem_total": kv["smem_total"],
            }
        )
    return rows


def main():
    rows = parse(RAW)
    names = [r["name"] for r in rows]
    regs = [r["regs"] for r in rows]
    smem_kb = [r["smem_total"] / 1024.0 for r in rows]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax2 = ax1.twinx()

    x = list(range(len(names)))
    width = 0.42
    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        regs,
        width=width,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        label="regs / thread",
    )
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        smem_kb,
        width=width,
        color="#F58518",
        edgecolor="black",
        linewidth=0.8,
        label="smem_total (KB)",
    )
    ax1.set_ylim(0, 280)
    ax2.set_ylim(0, 280)

    ax1.axhline(y=256, color="black", linestyle="--", linewidth=1.0)
    ax1.axhline(y=128, color="black", linestyle="--", linewidth=1.0)
    ax1.text(
        0.2,
        256,
        "256 threads (256 regs / thread)",
        transform=ax1.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=11,
    )
    ax1.text(
        0.2,
        128,
        "512 threads (128 regs / thread)",
        transform=ax1.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=11,
    )

    ax1.set_title("Registers & Shared Memory of Tasks", fontsize=14)
    ax1.set_ylabel("regs / thread", fontsize=12)
    ax2.set_ylabel("smem_total (KB)", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    # ax1.set_ylim(0, max(max(regs) * 1.1, 256 * 1.1))

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend([bars1, bars2], ["regs / thread", "smem_total (KB)"], loc="upper right", fontsize=12)
    out = "reg_smem_bar.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
