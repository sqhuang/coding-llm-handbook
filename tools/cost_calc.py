#!/usr/bin/env python3
"""
cost_calc.py — quick LLM training / inference cost estimator.

Stdlib-only. Replaces "use a spreadsheet" pattern for the kind of back-of-envelope
math that shows up all over phase_capstone / phase_consumer.

Examples:
  # 训 1B 模型 100B tokens 在 8×H100, 估算
  python tools/cost_calc.py train --params 1e9 --tokens 100e9 --gpu h100 --n-gpu 8

  # 跑 vLLM 推理 24h 在 1×4090 国内电费
  python tools/cost_calc.py infer --gpu 4090 --hours 24 --power-rate 1.0

  # mid-training 5B token 在 8×H100
  python tools/cost_calc.py train --params 9e9 --tokens 5e9 --gpu h100 --n-gpu 8 \
      --rent-rate 2.5
"""
from __future__ import annotations

import argparse
import sys


# Cards: peak FP16/BF16 FLOPs (TFLOPS) · TGP W · approximate cloud rent $/hr (2026 Q2)
GPUS = {
    "h100":   {"tflops": 989,  "tgp": 700, "rent": 2.5,  "vram": 80,  "label": "H100 SXM 80GB"},
    "h200":   {"tflops": 989,  "tgp": 700, "rent": 3.5,  "vram": 141, "label": "H200 141GB"},
    "a100":   {"tflops": 312,  "tgp": 400, "rent": 1.2,  "vram": 80,  "label": "A100 80GB"},
    "b200":   {"tflops": 2250, "tgp": 1000,"rent": 5.0,  "vram": 192, "label": "B200 192GB"},
    "4090":   {"tflops": 165,  "tgp": 450, "rent": 0.4,  "vram": 24,  "label": "RTX 4090 24GB"},
    "5090":   {"tflops": 419,  "tgp": 575, "rent": 0.6,  "vram": 32,  "label": "RTX 5090 32GB"},
    "3090":   {"tflops": 71,   "tgp": 350, "rent": 0.3,  "vram": 24,  "label": "RTX 3090 24GB"},
    "mi300x": {"tflops": 1300, "tgp": 750, "rent": 2.0,  "vram": 192, "label": "AMD MI300X 192GB"},
}

# Effective utilization assumptions (MFU)
MFU_TRAIN = {"h100": 0.45, "h200": 0.45, "a100": 0.40, "b200": 0.40,
             "4090": 0.30, "5090": 0.35, "3090": 0.25, "mi300x": 0.35}
MFU_INFER = 0.20   # decode is memory-bound, not compute; this is rough

# CN consumer electricity ~ ¥0.6/kWh (居民) ~ ¥1.0/kWh (商业)
# US: $0.15/kWh; EU: $0.30/kWh. Default to CN residential.
DEFAULT_POWER_RATE_RMB = 0.6
USD_PER_RMB = 0.14


def train_cost(args) -> dict:
    """6N·D rule: ~6 FLOPs per parameter per token (forward+backward).
    Dense; MoE uses active params not total, so pass --params = active.
    """
    gpu = GPUS[args.gpu]
    mfu = args.mfu or MFU_TRAIN[args.gpu]

    flops = 6 * args.params * args.tokens  # 6N·D
    flops_per_sec = gpu["tflops"] * 1e12 * mfu
    seconds = flops / (flops_per_sec * args.n_gpu)
    gpu_hours = (seconds / 3600) * args.n_gpu

    rent_rate = args.rent_rate if args.rent_rate is not None else gpu["rent"]
    rent_usd = gpu_hours * rent_rate

    # Power (electricity owning hardware)
    kwh = gpu["tgp"] / 1000 * gpu_hours
    power_rate = args.power_rate if args.power_rate is not None else DEFAULT_POWER_RATE_RMB
    power_rmb = kwh * power_rate

    return {
        "gpu": gpu["label"],
        "n_gpu": args.n_gpu,
        "mfu": mfu,
        "flops": flops,
        "wall_seconds": seconds,
        "wall_hours": seconds / 3600,
        "wall_days": seconds / 86400,
        "gpu_hours": gpu_hours,
        "rent_usd": rent_usd,
        "rent_rmb": rent_usd / USD_PER_RMB,
        "kwh": kwh,
        "power_rmb": power_rmb,
    }


def infer_cost(args) -> dict:
    gpu = GPUS[args.gpu]
    gpu_hours = args.hours * args.n_gpu
    rent_rate = args.rent_rate if args.rent_rate is not None else gpu["rent"]
    rent_usd = gpu_hours * rent_rate
    kwh = gpu["tgp"] / 1000 * gpu_hours
    power_rate = args.power_rate if args.power_rate is not None else DEFAULT_POWER_RATE_RMB
    power_rmb = kwh * power_rate
    return {
        "gpu": gpu["label"],
        "n_gpu": args.n_gpu,
        "wall_hours": args.hours,
        "gpu_hours": gpu_hours,
        "rent_usd": rent_usd,
        "rent_rmb": rent_usd / USD_PER_RMB,
        "kwh": kwh,
        "power_rmb": power_rmb,
    }


def fmt_num(x: float, sig: int = 3) -> str:
    if x >= 1e12: return f"{x/1e12:.{sig}g}T"
    if x >= 1e9:  return f"{x/1e9:.{sig}g}B"
    if x >= 1e6:  return f"{x/1e6:.{sig}g}M"
    if x >= 1e3:  return f"{x/1e3:.{sig}g}k"
    return f"{x:.{sig}g}"


def print_train(r: dict):
    print(f"=== Train estimate ===")
    print(f"  hardware       : {r['n_gpu']}× {r['gpu']} @ MFU {r['mfu']:.0%}")
    print(f"  total FLOPs    : {fmt_num(r['flops'])}")
    print(f"  wall time      : {r['wall_hours']:.1f}h  ({r['wall_days']:.2f} days)")
    print(f"  GPU-hours      : {r['gpu_hours']:.0f}")
    print(f"  cloud rent     : ${r['rent_usd']:.0f}  / ¥{r['rent_rmb']:.0f}")
    print(f"  own-hw electricity : {r['kwh']:.1f} kWh = ¥{r['power_rmb']:.2f}")
    print()
    print(f"  💡 ckpt frequency hint : every {max(1, int(r['wall_hours'] // 8))} h "
          f"(损失 1 段 = {min(8, r['wall_hours']):.1f} h)")


def print_infer(r: dict):
    print(f"=== Inference estimate ===")
    print(f"  hardware       : {r['n_gpu']}× {r['gpu']}")
    print(f"  wall time      : {r['wall_hours']:.1f}h")
    print(f"  GPU-hours      : {r['gpu_hours']:.0f}")
    print(f"  cloud rent     : ${r['rent_usd']:.0f}  / ¥{r['rent_rmb']:.0f}")
    print(f"  own-hw electricity : {r['kwh']:.1f} kWh = ¥{r['power_rmb']:.2f}")


def parse_engineering_float(s: str) -> float:
    """Accept 1.5B / 100B / 5e9 / 1_000 etc."""
    s = s.strip().replace("_", "").replace(",", "")
    suffixes = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "G": 1e9}
    if s and s[-1].upper() in suffixes:
        return float(s[:-1]) * suffixes[s[-1].upper()]
    return float(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--gpu", required=True, choices=list(GPUS.keys()),
                        help="GPU type (case insensitive)")
        sp.add_argument("--n-gpu", type=int, default=1, help="number of GPUs (default 1)")
        sp.add_argument("--rent-rate", type=float, default=None,
                        help="override cloud $/hr (default: looked up by gpu)")
        sp.add_argument("--power-rate", type=float, default=None,
                        help="electricity rate (RMB/kWh, default 0.6 居民)")

    sp = sub.add_parser("train", help="estimate training cost (6N·D rule)")
    sp.add_argument("--params", type=parse_engineering_float, required=True,
                    help="model params (active for MoE), e.g. 7e9 or 7B")
    sp.add_argument("--tokens", type=parse_engineering_float, required=True,
                    help="total tokens, e.g. 100e9 or 100B")
    sp.add_argument("--mfu", type=float, default=None,
                    help="MFU override (0-1), default looked up by gpu")
    add_common(sp)

    sp = sub.add_parser("infer", help="estimate inference / serving cost")
    sp.add_argument("--hours", type=float, required=True, help="wall hours")
    add_common(sp)

    sp = sub.add_parser("list-gpus", help="list known GPUs and assumptions")
    sp.set_defaults(_listing=True)

    args = p.parse_args()

    if args.cmd == "list-gpus":
        print(f"{'gpu':<10}{'TFLOPS':>10}{'TGP(W)':>10}{'VRAM(GB)':>10}{'$/hr':>8}  label")
        for k, v in GPUS.items():
            print(f"{k:<10}{v['tflops']:>10}{v['tgp']:>10}{v['vram']:>10}{v['rent']:>8.2f}  {v['label']}")
        return 0

    if args.cmd == "train":
        # Accept any case for GPU
        args.gpu = args.gpu.lower()
        r = train_cost(args)
        print_train(r)
    elif args.cmd == "infer":
        args.gpu = args.gpu.lower()
        r = infer_cost(args)
        print_infer(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
