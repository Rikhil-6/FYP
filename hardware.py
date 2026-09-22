"""Report the runtime device without downloading data or model weights."""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch


def report():
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": properties.name,
                            "memory_gib": round(properties.total_memory / 1024**3, 2)})
    try:
        command = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=5, check=False)
        driver_view = command.stdout.strip() if command.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        driver_view = None
    data_root = "/mnt/sdd" if os.path.isdir("/mnt/sdd") else os.getcwd()
    space = shutil.disk_usage(data_root)
    return {"torch_version": torch.__version__, "cuda_build": torch.version.cuda,
            "rocm_build": torch.version.hip, "gpu_available": bool(devices),
            "gpus": devices, "nvidia_smi": driver_view,
            "cpu_count": os.cpu_count(), "data_volume": data_root,
            "data_free_gib": round(space.free / 1024**3, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    result = report()
    print(json.dumps(result, indent=2))
    if args.require_gpu and not result["gpu_available"]:
        print("No GPU is accessible to PyTorch.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
