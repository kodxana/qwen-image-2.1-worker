"""Configuration shared by the downloader and worker; no GPU imports."""

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


@dataclass(frozen=True)
class Settings:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    model_path: str | None = None
    offload: str = "none"
    vae_tiling: bool = True
    max_output_bytes: int = 8_000_000

    @classmethod
    def from_env(cls) -> "Settings":
        offload = os.environ.get("CPU_OFFLOAD", "none").lower()
        if offload not in {"none", "model", "sequential"}:
            raise ValueError("CPU_OFFLOAD must be none, model, or sequential")
        tiling = os.environ.get("VAE_TILING", "true").lower()
        if tiling not in {"true", "false"}:
            raise ValueError("VAE_TILING must be true or false")
        return cls(
            model_id=os.environ.get("MODEL_ID", MODEL_ID),
            model_revision=os.environ.get("MODEL_REVISION", MODEL_REVISION),
            model_path=os.environ.get("MODEL_PATH") or None,
            offload=offload,
            vae_tiling=tiling == "true",
        )


def configure_cache() -> Path:
    """Call before importing huggingface_hub, which reads its paths at import time."""
    default = (
        "/runpod-volume/huggingface" if Path("/runpod-volume").is_dir() else "/cache/huggingface"
    )
    cache = Path(os.environ.setdefault("HF_HOME", default))
    cache.mkdir(parents=True, exist_ok=True)
    return cache
