"""Job-local reuse for the pinned pipeline, without copying its denoising loop.

The service holds its inference lock and uses identical prompts/references for
all calls in a context. Complete prompt tuples (including image layout masks)
and deterministic VAE reference encodings are reused. Target noise, scheduler
state and DiT KV caches are always created afresh by the upstream pipeline.
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("qwen_worker.conditioning")


class ConditioningReuseMixin:
    before_prompt_encode = None
    before_prepare_latents = None
    _conditioning_cache = None
    _reference_index = 0

    @contextmanager
    def conditioning_context(self):
        """One immutable request only. Never nest or share between threads."""
        if self._conditioning_cache is not None:
            raise RuntimeError("A conditioning job is already active")
        self._conditioning_cache = {"prompts": {}, "references": {}}
        try:
            yield
        finally:
            # Drop GPU tensor references on success, rejection and failure.
            self._conditioning_cache = None

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        cache = self._conditioning_cache
        key = (tuple([prompt] if isinstance(prompt, str) else prompt), str(device),
               tuple(item.size for item in image) if image is not None else None)
        if cache is not None and key in cache["prompts"]:
            logger.info("Reusing prompt conditioning within current job")
            return cache["prompts"][key]
        if self.before_prompt_encode is not None:
            self.before_prompt_encode()
        started = time.perf_counter()
        result = super()._get_qwen_prompt_embeds(prompt, image, device)
        logger.info("Prompt conditioning wall time: %.3fs", time.perf_counter() - started)
        if cache is not None:
            cache["prompts"][key] = result
        return result

    def prepare_latents(self, images, batch_size, num_channels_latents, height,
                        width, dtype, device, generator, latents=None):
        if self.before_prepare_latents is not None:
            self.before_prepare_latents()
        self._reference_index = 0
        started = time.perf_counter()
        result = super().prepare_latents(images, batch_size, num_channels_latents,
                                         height, width, dtype, device, generator, latents)
        logger.info("Reference encode and target noise wall time: %.3fs", time.perf_counter() - started)
        return result

    def _encode_vae_image(self, image, generator):
        cache = self._conditioning_cache
        index = self._reference_index
        self._reference_index += 1
        key = (index, tuple(image.shape), str(image.dtype), str(image.device))
        if cache is not None and key in cache["references"]:
            logger.info("Reusing reference %d latent within current job", index + 1)
            return cache["references"][key]
        # Pinned upstream uses latent_dist.mode() (argmax), which consumes no
        # generator state. Reusing this value preserves each seed's fresh noise.
        result = super()._encode_vae_image(image, generator)
        if cache is not None:
            cache["references"][key] = result
        return result


def make_pipeline_class(base_class):
    """Inherit upstream init/call signatures so Diffusers loading stays intact."""
    class WorkerQwenImage21Pipeline(ConditioningReuseMixin, base_class):
        pass

    return WorkerQwenImage21Pipeline
