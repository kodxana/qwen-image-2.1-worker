"""Configuration shared by the downloader and worker; no GPU imports."""

import os
from dataclasses import dataclass
from pathlib import Path

# Provisioning/test defaults only; production identifies its mounted snapshot.
MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
MODEL_REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"


BASE_MODEL_ID = "Qwen/Qwen-Image-2.1"
BASE_MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
SUPPORTED_MODELS = {BASE_MODEL_ID: BASE_MODEL_REVISION, MODEL_ID: MODEL_REVISION}
# The generic API keeps its existing range/default; Turbo's UI selects eight.
DEFAULT_STEPS = 40


OFFLOAD_MODES = {"none", "text_encoder", "model", "sequential"}


def _env_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    model_id: str = MODEL_ID
    model_revision: str | None = MODEL_REVISION
    model_path: str | None = None
    # The encoder lives in ordinary CPU memory between conditioning passes and
    # is staged to GPU on a job's conditioning cache miss. Transformer and VAE
    # stay resident; this is a large-GPU policy, not automatic device selection.
    offload: str = "text_encoder"
    vae_tiling: bool = True
    # Reject clearly oversized requests using phase estimates. Kernel workspace
    # and allocator peaks vary by GPU, so this cannot guarantee freedom from OOM.
    memory_guard: bool = True
    max_output_bytes: int = 8_000_000

    @classmethod
    def from_env(cls) -> "Settings":
        # Model selectors are read only by explicit provisioning, not serving.
        offload = os.environ.get("CPU_OFFLOAD", "text_encoder").lower()
        if offload not in OFFLOAD_MODES:
            raise ValueError("CPU_OFFLOAD must be none, text_encoder, model, or sequential")
        return cls(
            model_path=os.environ.get("MODEL_PATH") or None,
            offload=offload,
            vae_tiling=_env_bool("VAE_TILING", "true"),
            memory_guard=_env_bool("MEMORY_GUARD", "true"),
        )


def configure_cache() -> Path:
    """Select paths before Hub imports; discovering a cache must not write to it."""
    if Path("/runpod-volume/huggingface-cache").is_dir():
        default = "/runpod-volume/huggingface-cache"
    elif Path("/runpod-volume/huggingface").is_dir():
        default = "/runpod-volume/huggingface"
    elif Path("/runpod-volume").is_dir():
        default = "/runpod-volume/huggingface-cache"
    else:
        default = "/cache/huggingface"
    cache = Path(os.environ.setdefault("HF_HOME", default))
    return cache
