#!/usr/bin/env python3
"""
Repository Environment and Directory Structure Verification Script.
"""

import sys
from pathlib import Path


REQUIRED_DIRS = [
    "src",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/inference",
    "src/utils",
    "cuda",
    "cuda/kernels",
    "cuda/benchmarks",
    "data",
    "data/raw",
    "data/processed",
    "data/splits",
    "experiments",
    "experiments/baselines",
    "experiments/profiling",
    "results",
    "results/metrics",
    "results/figures",
    "tests",
    "docs",
    "configs",
    "notebooks",
    "app",
    "scripts"
]

CORE_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "yaml",
    "tqdm",
    "psutil",
    "pydantic"
]


def check_directories(repo_root: Path) -> bool:
    print("\n--- Checking Repository Directory Structure ---")
    all_ok = True
    for rel_dir in REQUIRED_DIRS:
        target = repo_root / rel_dir
        if target.is_dir():
            print(f"  [OK] {rel_dir}/")
        else:
            print(f"  [MISSING] {rel_dir}/")
            all_ok = False
    return all_ok


def check_packages() -> bool:
    print("\n--- Checking Core Python Packages ---")
    all_ok = True
    for pkg in CORE_PACKAGES:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg:<12} (version: {version})")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            all_ok = False
    return all_ok


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Repository Root: {repo_root}")

    dirs_ok = check_directories(repo_root)
    pkgs_ok = check_packages()

    print("\n--- Environment Summary ---")
    print(f"Directory Structure : {'PASSED' if dirs_ok else 'FAILED'}")
    print(f"Core Python Packages: {'PASSED' if pkgs_ok else 'FAILED'}")

    return 0 if (dirs_ok and pkgs_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
