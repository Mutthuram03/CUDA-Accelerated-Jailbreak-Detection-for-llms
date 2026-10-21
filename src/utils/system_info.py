"""
Hardware and system telemetry collection utility.
"""

import platform
import psutil
from typing import Dict, Any


def get_system_telemetry() -> Dict[str, Any]:
    """Collects CPU, RAM, OS, and GPU telemetry for experiment reproducibility."""
    mem = psutil.virtual_memory()
    telemetry: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "host_ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "host_ram_available_gb": round(mem.available / (1024 ** 3), 2),
        "gpu_available": False,
        "gpu_devices": []
    }

    try:
        import torch
        if torch.cuda.is_available():
            telemetry["gpu_available"] = True
            telemetry["torch_cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                telemetry["gpu_devices"].append({
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_mb": round(props.total_memory / (1024 ** 2), 2)
                })
        else:
            telemetry["torch_cuda_version"] = None
    except ImportError:
        pass

    return telemetry
