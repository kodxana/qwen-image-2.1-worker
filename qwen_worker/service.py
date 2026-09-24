"""Request orchestration, independent of GPU loading so it can be tested on CPU."""

import json
import threading
import time
from collections.abc import Callable

from qwen_worker.config import Settings
from qwen_worker.images import decode_images, encode_image
from qwen_worker.requests import MAX_SEED, GenerationRequest, InputError


class ImageService:
    def __init__(self, pipeline, generator_factory: Callable, settings: Settings):
        self.pipeline = pipeline
        self.generator_factory = generator_factory
        self.settings = settings
        self._lock = threading.Lock()

    def handle(self, job: dict) -> dict:
        if not isinstance(job, dict):
            raise InputError("job must contain an input object")
        request = GenerationRequest.parse(job.get("input"))
        references = decode_images(request.images)
        kwargs = {
            "prompt": request.effective_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "true_cfg_scale": request.true_cfg_scale,
            "output_resolution": request.output_resolution,
            "use_kv_cache": request.use_kv_cache,
        }
        if references:
            kwargs["image"] = references
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        started = time.perf_counter()
        result = {
            "model": self.settings.model_id,
            "revision": None if self.settings.model_path else self.settings.model_revision,
            "parameters": {
                "num_inference_steps": request.steps,
                "true_cfg_scale": request.true_cfg_scale,
                "output_resolution": request.output_resolution,
                "use_kv_cache": request.use_kv_cache,
            },
            "images": [],
        }
        # Diffusers mutates its scheduler/attention state. Never overlap calls to one pipeline.
        with self._lock:
            for index in range(request.num_images):
                seed = (request.seed + index) % (MAX_SEED + 1)
                output = self.pipeline(**kwargs, generator=self.generator_factory(seed))
                result["images"].append(
                    encode_image(output.images[0], request.output_format, request.quality, seed)
                )
                if len(json.dumps(result).encode("utf-8")) > self.settings.max_output_bytes - 4096:
                    raise InputError(
                        "Encoded images exceed the response limit. Request fewer/smaller images "
                        "or use webp/jpeg output."
                    )
        result["inference_seconds"] = round(time.perf_counter() - started, 3)
        return result
