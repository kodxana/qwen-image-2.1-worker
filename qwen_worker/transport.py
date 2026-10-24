"""Fit completed images into the serialized inline response without regenerating."""

import json
import logging
import math
import time
from itertools import combinations

from PIL import Image

from qwen_worker.images import encode_image
from qwen_worker.requests import GenerationRequest, InputError

logger = logging.getLogger("qwen_worker.transport")
MAX_INLINE_BYTES = 10_000_000
MIN_OUTPUT_SIDE = 256


def json_bytes(result: dict) -> int:
    # ASCII escaping is at least as large as the SDK's ensure_ascii=False.
    # Count base64 AND JSON metadata; the caller reserves envelope headroom.
    return len(json.dumps(result, ensure_ascii=True).encode("utf-8"))


def fit_output(result: dict, originals: list[tuple[Image.Image, int]],
               request: GenerationRequest, budget: int) -> None:
    """Optimize, resize within a usable floor, try JPEG, then maximize count.

    Every encoding uses original CPU pixels. JPEG fallback may flatten alpha on
    white; delivery metadata explains changes and partial batches. Never retry
    inference. Candidate sizes include delivery metadata before accepting them.
    """
    by_seed = dict((seed, image) for image, seed in originals)
    alpha = {}

    def delivery() -> None:
        if "delivery" not in result:
            return
        warnings = []
        jpeg = [item for item in result["images"] if item["mime_type"] == "image/jpeg"]
        if jpeg and request.output_format != "jpeg":
            warnings.append("JPEG fallback was used to fit the response size limit.")
        for item in jpeg:
            seed = item["seed"]
            if seed not in alpha:
                original = by_seed[seed]
                alpha[seed] = "A" in original.getbands() and original.getchannel("A").getextrema()[0] < 255
        if any(alpha.get(item["seed"], False) for item in jpeg):
            warnings.append("JPEG does not support transparency; alpha was flattened on white.")
        count = len(result["images"])
        if count < request.num_images:
            warnings.append(f"Returned {count} of {request.num_images} requested images to fit the response size limit. Missing images were not retried.")
        result["delivery"] = {"requested_images": request.num_images,
                              "returned_images": count, "partial": count < request.num_images,
                              "warnings": warnings}

    delivery()
    before = json_bytes(result)
    if before <= budget:
        return
    started = time.perf_counter()
    # Leave room for the final elapsed-time digits when fitting between passes.
    target = budget - 64
    quality = request.quality
    fmt = request.output_format
    # Do not reduce an image to a meaningless 1x1 just to claim the batch fits.
    # Images already smaller than the floor are never enlarged.
    floor = max(min(1.0, MIN_OUTPUT_SIDE / min(image.size)) for image, _ in originals)

    def attempt(scale: float, *, lossy: bool = False) -> int:
        images = []
        for original, seed in originals:
            size = (max(1, int(original.width * scale)), max(1, int(original.height * scale)))
            image = original if size == original.size else original.resize(size, Image.Resampling.LANCZOS)
            images.append(encode_image(image, fmt, quality, seed, optimize=True, lossy=lossy))
        result["images"] = images
        delivery()
        return json_bytes(result)

    def report(stage: str, after: int) -> None:
        dimensions = ", ".join(f"{by_seed[item['seed']].width}x{by_seed[item['seed']].height}->{item['width']}x{item['height']}"
                               for item in result["images"])
        logger.warning("[OUTPUT COMPRESSION] %s: %d -> %d JSON bytes (budget=%d), format=%s, quality=%s, "
                       "images=%d/%d requested, dimensions=%s, elapsed=%.3fs", stage, before, after, budget,
                       fmt, quality if fmt != "png" else "n/a (PNG)", len(result["images"]),
                       request.num_images, dimensions, time.perf_counter() - started)

    size = attempt(1.0)
    if size <= target:
        report("encoder optimization at requested quality", size)
        return

    lossy = fmt in {"jpeg", "webp"}
    if lossy:
        for quality in dict.fromkeys(min(request.quality, q) for q in (85, 70, 55, 40)):
            if fmt == "jpeg" and quality == request.quality:
                continue
            size = attempt(1.0, lossy=True)
            if size <= target:
                report("quality reduction", size)
                return

    scale = 1.0
    for _ in range(6):
        if scale <= floor:
            break
        scale = max(floor, scale * min(0.8, math.sqrt(max(1, target) / max(1, size)) * 0.94))
        size = attempt(scale, lossy=lossy)
        if size <= target:
            report("resized to fit", size)
            return
    if scale > floor:
        scale = floor
        size = attempt(scale, lossy=lossy)
        if size <= target:
            report("resized to minimum usable size", size)
            return

    # Format/quality fallback happens before dropping any generated images.
    previous_quality = quality
    fmt = "jpeg"
    result["delivery"] = {}
    for quality in dict.fromkeys(min(request.quality, q) for q in (85, 70, 55, 40, 25, 15, 5, 1)):
        if request.output_format == "jpeg" and quality >= previous_quality:
            continue  # Already tried this quality (or better) at this size.
        size = attempt(scale, lossy=True)
        if size <= target:
            report("JPEG fallback", size)
            return
    # The last encoded batch is JPEG at the lowest permitted quality.
    # Search every subset (at most 15) so a large first image cannot hide a
    # larger fitting count of later, smaller images. Preserve their seed order.
    encoded = result["images"]
    for count in range(len(encoded) - 1, 0, -1):
        for subset in combinations(encoded, count):
            result["images"] = list(subset)
            delivery()
            size = json_bytes(result)
            if size <= target:
                report("partial batch", size)
                return
    raise InputError("Encoded images exceed the inline output limit even after JPEG compression. "
                     "Not even one image and its metadata fit the configured limit.")
