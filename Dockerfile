# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/opt/huggingface \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip check

COPY scripts/check_runtime.py /app/scripts/check_runtime.py
RUN python /app/scripts/check_runtime.py
COPY qwen_worker /app/qwen_worker
COPY scripts/download_model.py /app/scripts/download_model.py

# Download the pinned checkpoint into an image layer; no GPU is needed to build.
# Keep this outside /runpod-volume so a mounted volume cannot hide the weights.
RUN python /app/scripts/download_model.py

# Runtime resolves the baked snapshot locally and never checks Hugging Face.
ENV HF_HUB_OFFLINE=1

COPY handler.py /app/handler.py
COPY examples /app/examples
CMD ["python", "-u", "/app/handler.py"]
