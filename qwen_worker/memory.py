"""Conservative admission heuristics for the pinned BF16 Qwen-Image 2.1 model.

Weights are budgeted separately by runtime placement. KV storage is calculated
from the architecture; activation/workspace allowances require GPU calibration.
Admission is not a guarantee against allocator or kernel workspace OOMs.
"""

import math
from dataclasses import dataclass

GIB = 1024**3
KV_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
ACTIVATION_BYTES_PER_TOKEN = 16 * 4096 * 2
FIXED_OVERHEAD_BYTES = 3 * GIB
PIXELS_PER_TOKEN_SIDE = 16


def reference_tokens(width: int, height: int, output_resolution: int) -> int:
    ratio = width / height
    target_width = math.sqrt(output_resolution * output_resolution * ratio)
    target_height = target_width / ratio
    target_width = round(target_width / 32) * 32
    target_height = round(target_height / 32) * 32
    return (target_width // PIXELS_PER_TOKEN_SIDE) * (target_height // PIXELS_PER_TOKEN_SIDE)


def text_token_bound(text: str, reference_count: int = 0) -> int:
    # Byte-level tokenization needs at most one token per UTF-8 byte of ordinary
    # text. Reserve template/special tokens and per-reference labels separately.
    return len(text.encode("utf-8")) + 128 + 16 * reference_count


@dataclass(frozen=True)
class MemoryEstimate:
    text_tokens: int
    negative_text_tokens: int
    image_tokens: int
    prefix_tokens: int
    negative_prefix_tokens: int
    target_tokens: int
    branches: int
    batch_size: int
    kv_cache_bytes: int
    parallel_kv_cache_bytes: int
    activation_bytes: int
    overhead_bytes: int
    total_bytes: int
    parallel_activations_bytes: int
    parallel_overhead_bytes: int
    parallel_total_bytes: int
    conditioning_bytes: int
    decoding_bytes: int
    parallel_decoding_bytes: int


def estimate(
    reference_sizes: list[tuple[int, int]],
    output_resolution: int,
    width: int,
    height: int,
    prompt_chars: int,
    true_cfg: bool,
    use_kv_cache: bool,
    batch_size: int = 1,
    *,
    positive_text_tokens: int | None = None,
    negative_text_tokens: int | None = None,
) -> MemoryEstimate:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Legacy callers provide character counts; use the maximum UTF-8 width.
    # The service supplies a tighter byte-based bound for each branch.
    text_tokens = positive_text_tokens if positive_text_tokens is not None else 4 * prompt_chars + 128
    neg_text = (negative_text_tokens if negative_text_tokens is not None else text_tokens) if true_cfg else 0
    image_tokens = sum(reference_tokens(w, h, output_resolution) for w, h in reference_sizes)
    prefix = text_tokens + image_tokens
    neg_prefix = neg_text + image_tokens if true_cfg else 0
    prefix_sum = prefix + neg_prefix
    target = (width // PIXELS_PER_TOKEN_SIDE) * (height // PIXELS_PER_TOKEN_SIDE)
    branches = 2 if true_cfg else 1

    kv_cache = prefix_sum * KV_BYTES_PER_TOKEN if use_kv_cache else 0
    activations = (prefix_sum + target * branches) * ACTIVATION_BYTES_PER_TOKEN
    overhead = FIXED_OVERHEAD_BYTES
    seq_total = kv_cache + activations + overhead
    # Every output has its own physical prefix K/V tensors. Reference and target
    # activations both scale with batch size in the pinned transformer.
    parallel_kv = kv_cache * batch_size
    parallel_activations = activations * batch_size
    parallel_overhead = overhead + max(0, batch_size - 1) * GIB // 2
    parallel_total = parallel_kv + parallel_activations + parallel_overhead

    # Encoder runs once per prompt branch, even for a parallel output batch.
    # Upstream still requests all hidden states; include retained layers plus
    # temporary vision/language workspaces. Latent image tokens deliberately
    # overestimate the vision-token count. This is an uncalibrated allowance.
    conditioning = max(prefix, neg_prefix) * 64 * 4096 * 2 + overhead
    # Condition KV remains live during VAE decode in upstream __call__.
    pixels_bytes = width * height * 4 * 4
    return MemoryEstimate(
        text_tokens=text_tokens, negative_text_tokens=neg_text,
        image_tokens=image_tokens, prefix_tokens=prefix, negative_prefix_tokens=neg_prefix,
        target_tokens=target, branches=branches, batch_size=batch_size,
        kv_cache_bytes=kv_cache, parallel_kv_cache_bytes=parallel_kv,
        activation_bytes=activations, overhead_bytes=overhead, total_bytes=seq_total,
        parallel_activations_bytes=parallel_activations,
        parallel_overhead_bytes=parallel_overhead, parallel_total_bytes=parallel_total,
        conditioning_bytes=conditioning, decoding_bytes=seq_total + pixels_bytes,
        parallel_decoding_bytes=parallel_total + pixels_bytes * batch_size,
    )
