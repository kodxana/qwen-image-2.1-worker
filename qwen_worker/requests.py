"""Validate requests before decoding images or doing GPU work."""

import math
import secrets
from dataclasses import dataclass
from typing import Any

ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
MAX_SEED = 2**63 - 1
MAX_INPUT_CHARS = 8_000_000


class InputError(ValueError):
    """A request the client can correct without restarting the worker."""


def integer(data: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = data.get(key, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise InputError(f"{key} must be an integer between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    num_images: int
    images: tuple[str, ...]
    transparent: bool
    output_format: str
    quality: int
    true_cfg_scale: float
    negative_prompt: str | None
    output_resolution: int
    use_kv_cache: bool

    @classmethod
    def parse(cls, data: Any) -> "GenerationRequest":
        if not isinstance(data, dict):
            raise InputError("input must be a JSON object")
        allowed = {
            "prompt",
            "width",
            "height",
            "aspect_ratio",
            "num_inference_steps",
            "seed",
            "num_images",
            "images",
            "transparent",
            "output_format",
            "quality",
            "true_cfg_scale",
            "negative_prompt",
            "output_resolution",
            "use_kv_cache",
        }
        unknown = set(data) - allowed
        if unknown:
            raise InputError(f"Unknown input fields: {', '.join(sorted(unknown))}")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20_000:
            raise InputError("prompt must contain 1 to 20000 characters")
        if "aspect_ratio" in data:
            ratio = data["aspect_ratio"]
            if not isinstance(ratio, str) or ratio not in ASPECT_RATIOS:
                raise InputError(f"aspect_ratio must be one of {', '.join(ASPECT_RATIOS)}")
            if "width" in data or "height" in data:
                raise InputError("Use aspect_ratio or width/height, not both")
            width, height = ASPECT_RATIOS[ratio]
        else:
            if ("width" in data) != ("height" in data):
                raise InputError("width and height must be supplied together")
            width = integer(data, "width", 1024, 256, 4096)
            height = integer(data, "height", 1024, 256, 4096)
        if width % 32 or height % 32:
            raise InputError("width and height must be multiples of 32")
        if width * height > 4_500_000:
            raise InputError("Output dimensions must total at most 4500000 pixels")
        images = data.get("images", [])
        if not isinstance(images, list) or len(images) > 10:
            raise InputError("images must be a list of up to 10 base64 images or image data URIs")
        if any(not isinstance(item, str) or not item for item in images):
            raise InputError("Every images entry must be a non-empty base64 string")
        if sum(map(len, images)) > MAX_INPUT_CHARS:
            raise InputError("Combined base64 reference images exceed 8000000 characters")
        transparent = data.get("transparent", False)
        if type(transparent) is not bool:
            raise InputError("transparent must be a boolean")
        output_format = data.get("output_format", "png")
        if output_format not in ("png", "webp", "jpeg"):
            raise InputError("output_format must be png, webp, or jpeg")
        if transparent and output_format == "jpeg":
            raise InputError("Transparent output requires png or webp")
        cfg = data.get("true_cfg_scale", 1.0)
        if type(cfg) not in {float, int} or not 1 <= cfg <= 20 or not math.isfinite(cfg):
            raise InputError("true_cfg_scale must be a finite number between 1 and 20")
        negative = data.get("negative_prompt")
        if negative is not None and (not isinstance(negative, str) or len(negative) > 20_000):
            raise InputError("negative_prompt must be a string of at most 20000 characters")
        if negative is not None and cfg <= 1:
            raise InputError("negative_prompt requires true_cfg_scale greater than 1")
        if cfg > 1 and negative is None:
            negative = ""
        resolution = integer(data, "output_resolution", 1024, 512, 2048)
        if resolution not in (512, 1024, 2048):
            raise InputError("output_resolution must be 512, 1024, or 2048")
        use_kv_cache = data.get("use_kv_cache", True)
        if type(use_kv_cache) is not bool:
            raise InputError("use_kv_cache must be a boolean")
        seed = integer(data, "seed", -1, -1, MAX_SEED)
        if seed == -1:
            seed = secrets.randbelow(MAX_SEED + 1)
        return cls(
            prompt=prompt.strip(),
            width=width,
            height=height,
            steps=integer(data, "num_inference_steps", 40, 1, 100),
            seed=seed,
            num_images=integer(data, "num_images", 1, 1, 4),
            images=tuple(images),
            transparent=transparent,
            output_format=output_format,
            quality=integer(data, "quality", 95, 1, 100),
            true_cfg_scale=float(cfg),
            negative_prompt=negative,
            output_resolution=resolution,
            use_kv_cache=use_kv_cache,
        )

    @property
    def effective_prompt(self) -> str:
        if self.transparent:
            return (
                f"This is an RGBA image with transparency. {self.prompt} "
                "The image has alpha channel and the background is transparent."
            )
        return self.prompt
