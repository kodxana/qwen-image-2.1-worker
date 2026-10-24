# syntax=docker/dockerfile:1
# One image serves Base or Turbo from the complete model cache detected at startup.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_DEVICE_SCHEDULE=2 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip check

COPY scripts/check_runtime.py /app/scripts/check_runtime.py
COPY qwen_worker /app/qwen_worker
RUN python /app/scripts/check_runtime.py
COPY scripts /app/scripts
COPY docs /app/docs
RUN python /app/scripts/benchmark_vram.py --list
COPY handler.py /app/handler.py
COPY examples /app/examples
CMD ["python", "-u", "/app/handler.py"]
