"""GPU initialization performed once, before the Runpod worker starts accepting jobs."""

import logging
from pathlib import Path

from qwen_worker.config import Settings, configure_cache
from qwen_worker.service import ImageService

logger = logging.getLogger(__name__)


def download_model(settings: Settings) -> str:
    configure_cache()
    if settings.model_path:
        if not (Path(settings.model_path) / "model_index.json").is_file():
            raise ValueError("MODEL_PATH must contain a complete Diffusers model snapshot")
        return settings.model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        allow_patterns=[
            "model_index.json",
            "processor/*",
            "scheduler/*",
            "text_encoder/*",
            "transformer/*",
            "vae/*",
        ],
    )


def load_service(settings: Settings) -> ImageService:
    configure_cache()
    import torch
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to run Qwen-Image-2.1 inference")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This worker requires a GPU with BF16 support (Ampere or newer)")
    logger.info("Loading %s at %s", settings.model_id, settings.model_revision)
    source = download_model(settings)
    pipe = QwenImage21Pipeline.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
    )
    if settings.vae_tiling:
        pipe.vae.enable_tiling()
    if settings.offload == "model":
        pipe.enable_model_cpu_offload()
    elif settings.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def infer(**kwargs):
        try:
            with torch.inference_mode():
                return pipe(**kwargs)
        except Exception:
            # Failed calls bypass the pipeline's normal hook/cache cleanup.
            for cleanup in (pipe.vae.clear_cache, pipe.maybe_free_model_hooks):
                try:
                    cleanup()
                except Exception:
                    logger.exception("Pipeline cleanup failed")
            raise

    def generator(seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    logger.info("Model ready; GPU=%s; offload=%s", torch.cuda.get_device_name(), settings.offload)
    return ImageService(infer, generator, settings)
