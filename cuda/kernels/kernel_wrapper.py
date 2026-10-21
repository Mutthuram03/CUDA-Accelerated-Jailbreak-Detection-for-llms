"""
Kernel loading and toolchain dispatch wrapper for custom CUDA C++ kernels.
"""

import shutil
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path


def check_cuda_toolchain() -> Dict[str, Any]:
    """
    Checks if NVCC and MSVC C++ compiler (cl.exe) are accessible on PATH.
    """
    nvcc_path = shutil.which("nvcc")
    cl_path = shutil.which("cl")

    nvcc_version = None
    if nvcc_path:
        try:
            res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
            nvcc_version = res.stdout.strip().splitlines()[-1]
        except Exception:
            pass

    return {
        "nvcc_available": bool(nvcc_path),
        "nvcc_path": nvcc_path,
        "nvcc_version": nvcc_version,
        "cl_available": bool(cl_path),
        "cl_path": cl_path,
        "custom_compilation_ready": bool(nvcc_path and cl_path)
    }


def load_fused_cuda_kernel() -> Optional[Any]:
    """
    Loads custom fused CUDA kernel if toolchain is available,
    otherwise reports diagnostics and falls back to PyTorch native tensor operations.
    """
    status = check_cuda_toolchain()
    if not status["custom_compilation_ready"]:
        print("\n[CUDA Toolchain Notice]")
        if not status["nvcc_available"]:
            print("  'nvcc' compiler is not in PATH. Custom .cu kernel compilation is bypassed.")
        if not status["cl_available"]:
            print("  MSVC C++ compiler ('cl.exe') is not in PATH.")
        print("  -> Using optimized PyTorch CUDA tensor operations (sparse.mm / cuBLAS) as GPU backend.")
        return None

    try:
        import torch
        from torch.utils.cpp_extension import load_inline
        cuda_src_path = Path(__file__).parent / "fused_classifier.cu"
        with open(cuda_src_path, "r", encoding="utf-8") as f:
            cuda_source = f.read()

        module = load_inline(
            name="fused_jailbreak_cuda",
            cpp_sources="",
            cuda_sources=cuda_source,
            functions=["fused_dense_linear_sigmoid_kernel"],
            extra_cuda_cflags=["-O3"]
        )
        return module
    except Exception as e:
        print(f"Warning: Failed to compile custom CUDA kernel: {e}")
        print("Falling back to PyTorch tensor routines.")
        return None
