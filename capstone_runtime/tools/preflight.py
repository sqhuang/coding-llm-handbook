#!/usr/bin/env python3
"""
preflight.py — pre-experiment sanity check.

What it does:
  - Lists python / OS / CPU / RAM / free disk
  - If torch is installed, reports CUDA + GPU model + free VRAM
  - Reports which env vars are set vs missing (no values printed)
  - Pings HuggingFace / GLM API endpoints (if reachable)
  - Exits 0 if "you can at least run preflight + Tier-1 steps";
    exits 1 if something blocker-level is missing (no python3, no disk).

Safe to run anywhere — no model downloads, no GPU allocation.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OK = "\x1b[32m✓\x1b[0m"
WARN = "\x1b[33m⚠\x1b[0m"
BAD = "\x1b[31m✗\x1b[0m"


def check_python() -> tuple[str, bool]:
    v = sys.version_info
    ok = v >= (3, 9)
    return (f"python {v.major}.{v.minor}.{v.micro}", ok)


def check_os() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def check_disk(path: Path) -> tuple[str, bool]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    ok = free_gb >= 100  # arbitrary "is there room to breathe"
    return (f"free {free_gb:.0f} GB at {path}", ok)


def check_ram() -> tuple[str, bool]:
    # No psutil dependency — read /proc/meminfo on linux, sysctl on mac.
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        gb = kb / (1024**2)
                        return (f"{gb:.0f} GB RAM", gb >= 32)
        elif sys.platform == "darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            gb = int(out) / (1024**3)
            return (f"{gb:.0f} GB RAM", gb >= 16)
    except Exception:
        pass
    return ("RAM unknown", True)


def check_torch_gpu() -> tuple[str, bool]:
    try:
        import torch  # noqa: WPS433
    except ImportError:
        return ("torch not installed (OK on a non-GPU box)", True)
    if not torch.cuda.is_available():
        return (f"torch {torch.__version__} present but cuda unavailable", True)
    count = torch.cuda.device_count()
    names = sorted({torch.cuda.get_device_name(i) for i in range(count)})
    cap = []
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        cap.append(f"{props.total_memory / (1024**3):.0f}GB")
    return (f"torch {torch.__version__} · {count}× {', '.join(names)} ({'/'.join(cap)})", count >= 1)


def check_env_vars() -> list[tuple[str, str, bool]]:
    """Returns (var, status, hard_required)."""
    out = []
    for var, hard in [
        ("HF_TOKEN", False),
        ("GH_TOKEN", False),
        ("GLM_API_KEY", False),
        ("LLM_BASE_URL", False),
        ("CUDA_VISIBLE_DEVICES", False),
        ("HF_HOME", False),
    ]:
        present = bool(os.environ.get(var))
        status = "set" if present else "not set"
        out.append((var, status, hard))
    return out


def check_network(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    print("=" * 60)
    print(" capstone_runtime preflight")
    print("=" * 60)

    rows: list[tuple[str, str, bool, bool]] = []  # name, detail, ok, hard

    py_detail, py_ok = check_python()
    rows.append(("python", py_detail, py_ok, True))

    rows.append(("os", check_os(), True, False))

    ram_detail, ram_ok = check_ram()
    rows.append(("ram", ram_detail, ram_ok, False))

    disk_detail, disk_ok = check_disk(ROOT)
    rows.append(("disk", disk_detail, disk_ok, True))

    gpu_detail, gpu_ok = check_torch_gpu()
    # not hard-required: this script must run on a Mac too
    rows.append(("gpu", gpu_detail, gpu_ok, False))

    for name, detail, ok, hard in rows:
        glyph = OK if ok else (BAD if hard else WARN)
        print(f"  {glyph} {name:8} {detail}")

    print()
    print("  env vars (informational — set in .env or shell):")
    for var, status, _ in check_env_vars():
        glyph = OK if status == "set" else WARN
        print(f"    {glyph} {var:24}  {status}")

    print()
    print("  network reachability (informational):")
    for label, host, port in [
        ("huggingface.co",   "huggingface.co",   443),
        ("hf-mirror.com",    "hf-mirror.com",    443),
        ("api.github.com",   "api.github.com",   443),
        ("z.ai (GLM API)",   "api.z.ai",         443),
    ]:
        ok = check_network(host, port)
        glyph = OK if ok else WARN
        print(f"    {glyph} {label:24} ({host}:{port})")

    print()
    hard_failed = any(not ok for _, _, ok, hard in rows if hard)
    if hard_failed:
        print(f"  {BAD} preflight blocked — fix items marked ✗ before continuing")
        return 1

    if not gpu_ok:
        print(f"  {WARN} no GPU detected — Tier-1 steps (01, 04, 11, 19) still runnable")
        print(f"        Tier-2 GPU steps (06, 09, 12, 15…) need an H100 box")
    else:
        print(f"  {OK} ready — start with `make test` then `make step-01`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
