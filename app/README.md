# Real-Time Detection API & Frontend (Staged for Phase 4)

This directory is reserved for deploying the validated jailbreak detection engine as a high-throughput, low-latency API and user-facing evaluation interface.

## Architecture Plan
- **Backend API**: FastAPI / Uvicorn server providing low-overhead streaming or batched inference endpoints (`/v1/detect`, `/v1/batch-detect`, `/v1/health`, `/v1/metrics`).
- **GPU Inference Engine**: Caching, CUDA streams, and asynchronous batching for concurrent prompt streams.
- **Frontend**: Responsive, modern web dashboard for interactive prompt testing, category breakdown, token attention / heuristic highlighting, and latency monitoring.

> **Status**: Per research protocol, application development begins only after the detection pipeline, CUDA kernels, and benchmark evaluations have been validated and approved.
