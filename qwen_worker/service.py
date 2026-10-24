"""Request orchestration, independent of GPU loading so it can be tested on CPU."""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext

from qwen_worker.config import Settings
from qwen_worker.images import decode_images, encode_image
from qwen_worker.memory import GIB, MemoryEstimate, estimate, text_token_bound
from qwen_worker.requests import MAX_SEED, GenerationRequest, InputError
from qwen_worker.transport import MAX_INLINE_BYTES, fit_output, json_bytes

logger = logging.getLogger("qwen_worker.service")
OUTPUT_ENVELOPE_HEADROOM = 4096


class ImageService:
    def __init__(
        self, pipeline, generator_factory: Callable, settings: Settings,
        memory_budget: Callable[[], int] | None = None,
        release_memory: Callable[[], None] | None = None,
        phase_budgets: Callable[[], dict[str, int]] | None = None,
        job_context: Callable | None = None,
        preflight: Callable | None = None,
        sampling_steps: int | None = None,
    ):
        self.pipeline = pipeline
        self.generator_factory = generator_factory
        self.settings = settings
        self.memory_budget = memory_budget
        self.release_memory = release_memory
        self.phase_budgets = phase_budgets
        self.job_context = job_context
        self.preflight = preflight
        self.sampling_steps = sampling_steps
        self._lock = threading.Lock()

    def check_memory(self, request: GenerationRequest, references: list) -> tuple[MemoryEstimate, bool]:
        need = estimate(
            [image.size for image in references], request.output_resolution,
            request.width, request.height, len(request.effective_prompt),
            request.true_cfg_scale > 1, request.use_kv_cache, request.num_images,
            positive_text_tokens=text_token_bound(request.effective_prompt, len(references)),
            negative_text_tokens=text_token_bound(request.negative_prompt or "", len(references)),
        )
        if self.phase_budgets is not None:
            budgets = self.phase_budgets()
        elif self.memory_budget is not None:
            budget = self.memory_budget()
            budgets = dict.fromkeys(("conditioning", "denoising", "decoding"), budget)
        else:
            logger.info("Memory guard disabled; estimates remain advisory")
            return need, request.num_images > 1

        sequential = {"conditioning": need.conditioning_bytes,
                      "denoising": need.total_bytes, "decoding": need.decoding_bytes}
        parallel = {"conditioning": need.conditioning_bytes,
                    "denoising": need.parallel_total_bytes, "decoding": need.parallel_decoding_bytes}
        for phase, required in sequential.items():
            available = budgets[phase]
            logger.info("Memory estimate %s: sequential %.2f GiB, batch %.2f GiB, tensor budget %.2f GiB",
                        phase, required / GIB, parallel[phase] / GIB, available / GIB)
            if required > available:
                raise InputError(
                    f"Request needs an estimated {required / GIB:.1f} GiB for {phase} tensors "
                    f"but only {available / GIB:.1f} GiB is available after model placement. "
                    "Use fewer references, smaller dimensions/output_resolution, a shorter prompt, "
                    "or a GPU/offload configuration with more headroom."
                )
        can_parallel = request.num_images > 1 and all(
            required <= budgets[phase] for phase, required in parallel.items()
        )
        logger.info("Routing %d output(s): %s; prefix KV %.2f GiB/sample, %.2f GiB/batch (heuristic admission)",
                    request.num_images, "parallel" if can_parallel else "sequential",
                    need.kv_cache_bytes / GIB, need.parallel_kv_cache_bytes / GIB)
        return need, can_parallel

    def _check_output_size(self, result: dict) -> int:
        # Restore the documented original cap with headroom for Runpod's envelope
        # and the final timing field. Measure JSON bytes, not container size.
        size = json_bytes(result)
        if size > min(self.settings.max_output_bytes, MAX_INLINE_BYTES) - OUTPUT_ENVELOPE_HEADROOM:
            raise InputError(
                "Encoded images exceed the inline output limit. Use fewer images, "
                "smaller dimensions, or JPEG when transparency is not required."
            )
        return size

    def _fit_output(self, result: dict, originals: list, request: GenerationRequest) -> int:
        budget = min(self.settings.max_output_bytes, MAX_INLINE_BYTES) - OUTPUT_ENVELOPE_HEADROOM
        fit_output(result, originals, request, budget)
        return self._check_output_size(result)

    def _append_images(self, result: dict, output, seeds: list[int], request: GenerationRequest, originals: list) -> None:
        images = getattr(output, "images", None)
        if images is None or len(images) != len(seeds):
            actual = None if images is None else len(images)
            raise RuntimeError(f"Pipeline returned {actual} images; expected {len(seeds)}")
        started = time.perf_counter()
        for image, seed in zip(images, seeds):
            originals.append((image.copy(), seed))
            result["images"].append(encode_image(image, request.output_format, request.quality, seed))
        self._fit_output(result, originals, request)
        logger.info("Output encoding: %d image(s), %.3fs", len(seeds), time.perf_counter() - started)

    def handle(self, job: dict) -> dict:
        job_started = time.perf_counter()
        if not isinstance(job, dict):
            raise InputError("job must contain an input object")
        request = GenerationRequest.parse(job.get("input", {}))
        actual_steps = self.sampling_steps or request.steps
        if actual_steps != request.steps:
            logger.info("Native checkpoint schedule: requested %d steps, actual %d; request passed unchanged",
                        request.steps, actual_steps)
        decoded_at = time.perf_counter()
        references = decode_images(request.images)
        logger.info("Job %s: %dx%d, outputs=%d, steps=%d, CFG=%s, refs=%d, decode=%.3fs",
                    job.get("id", "unspecified_job"), request.width, request.height,
                    request.num_images, actual_steps, request.true_cfg_scale,
                    len(references), time.perf_counter() - decoded_at)
        kwargs = {
            "prompt": request.effective_prompt, "width": request.width, "height": request.height,
            "num_inference_steps": request.steps, "true_cfg_scale": request.true_cfg_scale,
            "output_resolution": request.output_resolution, "use_kv_cache": request.use_kv_cache,
        }
        if references:
            kwargs["image"] = references  # One flat reference set shared by all outputs.
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        result = {
            "model": self.settings.model_id,
            "revision": self.settings.model_revision,
            "parameters": {"num_inference_steps": actual_steps, "true_cfg_scale": request.true_cfg_scale,
                           "output_resolution": request.output_resolution, "use_kv_cache": request.use_kv_cache},
            "images": [],
        }
        started = time.perf_counter()
        job_id = job.get("id", "unspecified_job")
        route = "not started"
        stats = None
        inference_seconds = encoding_seconds = 0.0
        response_bytes = 0
        succeeded = False
        originals = []
        with self._lock:
            lock_wait = time.perf_counter() - started
            logger.info("[JOB %s] Inference lock acquired after %.3fs", job_id, lock_wait)
            try:
                # A broken deployment limit must fail before any GPU work.
                self._check_output_size({**result, "inference_seconds": 0.0})
                if self.preflight is not None:
                    self.preflight(request, references)
                # Observe available memory only after the previous job has
                # relinquished the shared mutable pipeline.
                _, can_parallel = self.check_memory(request, references)
                # Encoded response size is handled after inference by _fit_output;
                # only the phase memory budgets determine the generation route.
                route = "parallel" if can_parallel else "sequential"
                logger.info("[JOB %s] GENERATION START: route=%s, outputs=%d, steps=%d, offload=%s",
                            job_id, route, request.num_images, actual_steps, self.settings.offload)
                context = self.job_context() if self.job_context is not None else nullcontext()
                with context as stats:
                    seeds = [(request.seed + index) % (MAX_SEED + 1) for index in range(request.num_images)]
                    groups = [seeds] if can_parallel else [[seed] for seed in seeds]
                    for index, group in enumerate(groups, 1):
                        logger.info("[JOB %s] Generation pass %d/%d: %d image(s), seeds=%s",
                                    job_id, index, len(groups), len(group), group)
                        call_kwargs = dict(kwargs)
                        if can_parallel:
                            # Encode a scalar prompt once, then repeat embeddings.
                            call_kwargs["num_images_per_prompt"] = len(group)
                            generator = [self.generator_factory(seed) for seed in group]
                        else:
                            generator = self.generator_factory(group[0])
                        inference_at = time.perf_counter()
                        try:
                            output = self.pipeline(**call_kwargs, generator=generator)
                        finally:
                            call_seconds = time.perf_counter() - inference_at
                            inference_seconds += call_seconds
                        logger.info("[JOB %s] Generation pass %d/%d finished: images=%d, elapsed=%.3fs",
                                    job_id, index, len(groups), len(group), call_seconds)
                        encoding_at = time.perf_counter()
                        try:
                            self._append_images(result, output, group, request, originals)
                        finally:
                            encoding_seconds += time.perf_counter() - encoding_at
                            del output
                        if len(result["images"]) < len(originals):
                            logger.warning("[JOB %s] Partial delivery: keeping %d of %d generated images; "
                                           "remaining generation passes skipped to avoid wasting compute",
                                           job_id, len(result["images"]), len(originals))
                            break
                # Errors (including OOM) propagate to the existing handler;
                # retries here retain failed tensors and hide deterministic bugs.
                result["inference_seconds"] = round(time.perf_counter() - started, 3)
                response_bytes = self._fit_output(result, originals, request)
                succeeded = True
            finally:
                if self.release_memory is not None:
                    try:
                        self.release_memory()
                    except Exception:
                        logger.exception("Post-job memory cleanup failed")
                peaks = ("allocated=%.2f GiB, reserved=%.2f GiB" % (stats["allocated"] / GIB, stats["reserved"] / GIB)
                         if isinstance(stats, dict) and {"allocated", "reserved"} <= stats.keys() else "unavailable")
                logger.log(logging.INFO if succeeded else logging.WARNING,
                           "[JOB %s] %s: route=%s, images=%d/%d, elapsed=%.3fs, lock_wait=%.3fs, "
                           "pipeline=%.3fs, encoding=%.3fs, peak VRAM %s, response=%d bytes",
                           job_id, "COMPLETE" if succeeded else "FAILED", route, len(result["images"]),
                           request.num_images, time.perf_counter() - job_started, lock_wait,
                           inference_seconds, encoding_seconds, peaks, response_bytes)
        return result
