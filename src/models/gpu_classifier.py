"""
GPU-accelerated PyTorch jailbreak classifier with explicit CPU fallback,
CUDA Event timing, and numerical correctness verification.
"""

import time
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import scipy.sparse as sp
from ..preprocessing.text_cleaner import TextCleaner

warnings.filterwarnings("ignore", message=".*Sparse invariant checks are implicitly disabled.*")



class GPUJailbreakClassifier:
    """
    High-throughput GPU inference engine for linear jailbreak classification.
    Executes batched linear projections and sigmoid activations on NVIDIA GPU
    via PyTorch CUDA tensors, with automatic and transparent CPU fallback.
    """

    def __init__(
        self,
        baseline_model: Any,
        device_override: Optional[str] = None
    ):
        """
        Initializes GPU classifier by converting trained scikit-learn model weights
        into optimized device tensors.
        """
        self.cleaner = TextCleaner()
        self.feature_extractor = baseline_model.feature_extractor
        self.decision_threshold = getattr(baseline_model, "decision_threshold", 0.5)
        self.tau_low = getattr(baseline_model, "tau_low", 0.30)
        self.tau_high = getattr(baseline_model, "tau_high", 0.70)

        import torch
        self.torch = torch

        # Device negotiation with explicit fallback reporting
        if device_override:
            self.device = torch.device(device_override)
            self.is_cuda = (self.device.type == "cuda")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.is_cuda = True
            else:
                self.device = torch.device("cpu")
                self.is_cuda = False

        # Extract weights from Logistic Regression (binary classification, Eq. 10)
        sk_clf = baseline_model.classifier
        raw_weights = sk_clf.coef_.T.astype(np.float32)  # Shape (V, 1)
        raw_bias = sk_clf.intercept_.astype(np.float32)  # Shape (1,)

        self.weight = torch.from_numpy(raw_weights).to(self.device)
        self.bias = torch.from_numpy(raw_bias).to(self.device)

    def get_execution_backend(self) -> Dict[str, Any]:
        """Returns details about active device and hardware acceleration status."""
        info = {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "weight_shape": list(self.weight.shape)
        }
        if self.is_cuda:
            info["device_name"] = self.torch.cuda.get_device_name(self.device)
            info["vram_allocated_mb"] = round(self.torch.cuda.memory_allocated(self.device) / (1024 ** 2), 2)
        else:
            info["notice"] = "Running on CPU backend (PyTorch CPU fallback active)."
        return info

    def forward_features_decomposed(self, X_sparse: sp.csr_matrix) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Executes forward inference with explicit latency decomposition (Eq. 19):
        T_total = T_pre + T_H2D + T_kernel + T_D2H + T_post
        Measures T_H2D, T_kernel, and T_D2H using CUDA events when CUDA is active.
        """
        coo = X_sparse.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape

        if self.is_cuda:
            ev_start = self.torch.cuda.Event(enable_timing=True)
            ev_h2d = self.torch.cuda.Event(enable_timing=True)
            ev_kernel = self.torch.cuda.Event(enable_timing=True)
            ev_d2h = self.torch.cuda.Event(enable_timing=True)

            ev_start.record()
            # 1. Host-to-Device transfer (T_H2D)
            i_dev = self.torch.LongTensor(indices).to(self.device, non_blocking=True)
            v_dev = self.torch.FloatTensor(values).to(self.device, non_blocking=True)
            X_tensor = self.torch.sparse_coo_tensor(i_dev, v_dev, self.torch.Size(shape), device=self.device, check_invariants=False)
            ev_h2d.record()

            # 2. Kernel execution: linear projection + sigmoid activation (T_kernel, Eq. 10, 11)
            logits = self.torch.sparse.mm(X_tensor, self.weight) + self.bias
            probs_gpu = self.torch.sigmoid(logits)
            ev_kernel.record()

            # 3. Device-to-Host transfer (T_D2H)
            probs_cpu = probs_gpu.cpu()
            ev_d2h.record()

            self.torch.cuda.synchronize()
            t_h2d = ev_start.elapsed_time(ev_h2d)
            t_kernel = ev_h2d.elapsed_time(ev_kernel)
            t_d2h = ev_kernel.elapsed_time(ev_d2h)
            probs_np = probs_cpu.detach().numpy().ravel()
        else:
            # CPU fallback timing via perf_counter
            t0 = time.perf_counter()
            i_cpu = self.torch.LongTensor(indices)
            v_cpu = self.torch.FloatTensor(values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_tensor = self.torch.sparse_coo_tensor(i_cpu, v_cpu, self.torch.Size(shape), check_invariants=False)
            t1 = time.perf_counter()
            t_h2d = (t1 - t0) * 1000.0



            logits = self.torch.sparse.mm(X_tensor, self.weight) + self.bias
            probs_gpu = self.torch.sigmoid(logits)
            t2 = time.perf_counter()
            t_kernel = (t2 - t1) * 1000.0

            probs_np = probs_gpu.detach().numpy().ravel()
            t3 = time.perf_counter()
            t_d2h = (t3 - t2) * 1000.0

        p_mal = probs_np
        p_benign = 1.0 - p_mal
        decomp = {
            "t_h2d_ms": round(t_h2d, 3),
            "t_kernel_ms": round(t_kernel, 3),
            "t_d2h_ms": round(t_d2h, 3),
            "t_gpu_total_ms": round(t_h2d + t_kernel + t_d2h, 3)
        }
        return np.column_stack([p_benign, p_mal]), decomp

    def forward_features(self, X_sparse: sp.csr_matrix) -> Tuple[np.ndarray, float]:
        """
        Executes forward inference on pre-extracted CSR sparse feature matrix.
        Returns probabilities and raw computation latency in milliseconds.
        """
        probs, decomp = self.forward_features_decomposed(X_sparse)
        return probs, decomp["t_gpu_total_ms"]

    def predict_proba(self, prompts: List[str]) -> Tuple[np.ndarray, float]:
        """
        End-to-end inference from raw prompt strings to calibrated probabilities.
        """
        cleaned = self.cleaner.clean_batch(prompts)
        X = self.feature_extractor.transform(cleaned)
        return self.forward_features(X)

    def predict(self, prompts: List[str]) -> Tuple[np.ndarray, float]:
        """Returns binary predictions using configured decision threshold."""
        probs, latency_ms = self.predict_proba(prompts)
        preds = (probs[:, 1] >= self.decision_threshold).astype(np.int32)
        return preds, latency_ms

    def predict_three_way(
        self,
        prompts: List[str],
        tau_low: Optional[float] = None,
        tau_high: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """Returns three-way risk decision array ['allow', 'flag', 'block'] (Eq. 14, 15)."""
        tl = tau_low if tau_low is not None else self.tau_low
        th = tau_high if tau_high is not None else self.tau_high
        probs, latency_ms = self.predict_proba(prompts)
        from ..evaluation.metrics import classify_three_way_decision
        decisions = classify_three_way_decision(probs[:, 1], tau_low=tl, tau_high=th)
        return decisions, latency_ms

    def verify_numerical_agreement(
        self,
        baseline_model: Any,
        test_prompts: List[str],
        atol: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Verifies exact numerical agreement between CPU scikit-learn and PyTorch GPU/CPU inference.
        """
        cpu_probs = baseline_model.predict_proba(test_prompts)
        gpu_probs, gpu_latency_ms = self.predict_proba(test_prompts)

        max_diff = float(np.max(np.abs(cpu_probs - gpu_probs)))
        agreement = bool(max_diff <= atol)

        return {
            "agreement_passed": agreement,
            "max_absolute_difference": round(max_diff, 8),
            "tolerance": atol,
            "sample_count": len(test_prompts),
            "backend": str(self.device)
        }
