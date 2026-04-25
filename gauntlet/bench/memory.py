"""
Snapshot GPU memory used by the two vLLM server processes.

Two modes:
  --base-pid / --quant-pid    per-process memory from `nvidia-smi --query-compute-apps`
  (default, no PIDs)          total used / total available on GPU 0 plus notes

Writes results/memory.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def per_process_memory(pid: int) -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            return {"used_mib": int(parts[1]), "pid": pid}
    return {"used_mib": None, "pid": pid, "note": "pid not found"}


def gpu_total(gpu_index: int) -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    name, used, total = [p.strip() for p in out.splitlines()[0].split(",")]
    return {"gpu": name, "used_mib": int(used), "total_mib": int(total)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-pid", type=int, default=None)
    p.add_argument("--quant-pid", type=int, default=None)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--output", default="results/memory.json")
    args = p.parse_args()

    data: dict = {"gpu_total": gpu_total(args.gpu_index)}
    data["base"] = (
        per_process_memory(args.base_pid)
        if args.base_pid
        else {"used_mib": None, "note": "pass --base-pid for per-process memory"}
    )
    data["quant"] = (
        per_process_memory(args.quant_pid)
        if args.quant_pid
        else {"used_mib": None, "note": "pass --quant-pid for per-process memory"}
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
