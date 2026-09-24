"""Bounded image decoding and serialization; never fetch arbitrary input URLs."""

import base64
import binascii
import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from qwen_worker.requests import InputError

MAX_REFERENCE_PIXELS = 20_000_000
MAX_TOTAL_REFERENCE_PIXELS = 40_000_000


def decode_images(values: tuple[str, ...]) -> list[Image.Image]:
    images = []
    total_pixels = 0
    for value in values:
        if value.startswith("data:"):
            header, separator, value = value.partition(",")
            if not separator or not header.startswith("data:image/") or ";base64" not in header:
                raise InputError("Reference data URIs must contain base64 image data")
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InputError("Reference image contains invalid base64") from exc
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    # At the minimum preprocessing resolution (512), a <=100:1 ratio
                    # keeps both dimensions nonzero when upstream rounds to 32 pixels.
                    if max(image.size) > min(image.size) * 100:
                        raise InputError("Reference image aspect ratio must be at most 100:1")
                    pixels = image.width * image.height
                    total_pixels += pixels
                    if pixels > MAX_REFERENCE_PIXELS or total_pixels > MAX_TOTAL_REFERENCE_PIXELS:
                        raise InputError("Reference images exceed the decoded pixel limit")
                    if getattr(image, "n_frames", 1) != 1:
                        raise InputError("Reference images must contain a single frame")
                    image.load()
                    oriented = ImageOps.exif_transpose(image)
                    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
                    images.append(oriented.convert("RGBA" if has_alpha else "RGB"))
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise InputError("Reference image is invalid, unsupported, or too large") from exc
    return images


def encode_image(image: Image.Image, output_format: str, quality: int, seed: int) -> dict:
    image = image.copy()
    buffer = io.BytesIO()
    if output_format == "jpeg":
        if "A" in image.getbands():
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    elif output_format == "webp":
        image.save(buffer, format="WEBP", quality=quality, lossless="A" in image.getbands())
    else:
        image.save(buffer, format="PNG")
    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": f"image/{output_format}",
        "width": image.width,
        "height": image.height,
        "seed": seed,
    }
