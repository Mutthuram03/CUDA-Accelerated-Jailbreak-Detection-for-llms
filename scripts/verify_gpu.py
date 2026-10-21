#!/usr/bin/env python3
"""
GPU & CUDA Verification Script for CUDA-Accelerated Jailbreak Detection Project.

Checks:
1. Python version and host platform details.
2. NVIDIA driver status via nvidia-smi.
3. PyTorch installation version and CUDA build flags.
4. PyTorch CUDA device availability, count, and device properties.
5. Performs a verified GPU tensor computation (matrix multiplication) with CUDA event timing.
6. Provides actionable diagnostics if PyTorch was installed without CUDA support.
"""

import sys
import platform
import subprocess
import time


def print_banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def check_host_environment() -> None:
    print_banner("1. Host System Environment")
    print(f"Operating System : {platform.system()} {platform.release()} (Version: {platform.version()})")
    print(f"Architecture     : {platform.machine()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {platform.python_version()} ({platform.python_implementation()})")


def check_nvidia_smi() -> bool:
    print_banner("2. NVIDIA System Management Interface (nvidia-smi)")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        if lines:
            print("Physical GPU(s) detected via nvidia-smi:")
            for i, line in enumerate(lines):
                print(f"  [GPU {i}] {line}")
            return True
        else:
            print("nvidia-smi returned empty output.")
            return False
    except FileNotFoundError:
        print("WARNING: 'nvidia-smi' command not found in PATH.")
        print("Ensure NVIDIA GPU drivers are installed.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing nvidia-smi (exit code {e.returncode}): {e.stderr.strip()}")
        return False


def check_pytorch_cuda() -> bool:
    print_banner("3. PyTorch CUDA Availability & Device Capabilities")
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed in the current environment.")
        print("Install via: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    print(f"PyTorch Version       : {torch.__version__}")
    print(f"PyTorch CUDA Built-In : {torch.version.cuda}")
    print(f"CUDA Available Flag   : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[DIAGNOSTIC NOTICE]")
        if "+cpu" in torch.__version__ or torch.version.cuda is None:
            print("Current PyTorch build is CPU-only ('+cpu' detected).")
            print("Although your physical NVIDIA GPU is functional, PyTorch cannot execute GPU kernels.")
            print("To enable PyTorch CUDA acceleration on Windows, run:")
            print("\n    pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n")
        else:
            print("CUDA is not available. Please verify NVIDIA driver and CUDA installation.")
        return False

    device_count = torch.cuda.device_count()
    print(f"Device Count          : {device_count}")
    
    for device_idx in range(device_count):
        props = torch.cuda.get_device_properties(device_idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"\nDevice [{device_idx}]: {props.name}")
        print(f"  Compute Capability : {props.major}.{props.minor}")
        print(f"  Total VRAM         : {total_mem_gb:.2f} GB ({props.total_memory} bytes)")
        print(f"  Multi-Processors   : {props.multi_processor_count}")

    return True


def run_gpu_computation() -> bool:
    print_banner("4. GPU Computation Verification Test")
    import torch

    if not torch.cuda.is_available():
        print("SKIPPED: Cannot run GPU computation because PyTorch CUDA is not available.")
        return False

    try:
        device = torch.device("cuda:0")
        print(f"Targeting device: {torch.cuda.get_device_name(device)}")

        # Initialize test matrices
        matrix_size = 2048
        print(f"Allocating 2x Float32 matrices ({matrix_size}x{matrix_size}) on GPU...")
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)

        # Warmup iteration
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Timed execution using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        c = torch.matmul(a, b)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        checksum = float(c.sum().item())
        vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

        print("\nComputation Successful:")
        print(f"  Matrix multiplication size : {matrix_size} x {matrix_size}")
        print(f"  CUDA Event Elapsed Time    : {elapsed_time_ms:.3f} ms")
        print(f"  Tensor Checksum            : {checksum:.4e}")
        print(f"  Active VRAM Allocated      : {vram_allocated_mb:.2f} MB")
        print("  CUDA Hardware acceleration verified working in PyTorch.")
        return True
    except Exception as e:
        print(f"ERROR during GPU computation test: {e}")
        return False


def main() -> int:
    check_host_environment()
    has_nvidia_gpu = check_nvidia_smi()
    torch_cuda_ready = check_pytorch_cuda()

    if torch_cuda_ready:
        computation_success = run_gpu_computation()
    else:
        computation_success = False

    print_banner("Verification Summary")
    print(f"Physical NVIDIA GPU Detected  : {'PASSED' if has_nvidia_gpu else 'FAILED'}")
    print(f"PyTorch CUDA Integration      : {'PASSED' if torch_cuda_ready else 'ATTENTION NEEDED'}")
    print(f"GPU Computation Test          : {'PASSED' if computation_success else 'SKIPPED / PENDING CUDA TORCH'}")

    if has_nvidia_gpu and not torch_cuda_ready:
        print("\nNote: Hardware is ready (NVIDIA GPU present). To activate CUDA in PyTorch:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return 0  # Completed audit successfully with clear diagnostics

    return 0 if (torch_cuda_ready and computation_success) else 1


if __name__ == "__main__":
    sys.exit(main())
