"""CPU-only layout admission for the pinned Qwen-Image-2.1 pipeline.

Use the processor's own size/token expansion helpers without preprocessing
pixels or running the conditioner. Position bounds include the image grids,
not just text length. No tensors or references are retained between jobs.
"""

import math
from dataclasses import dataclass

from qwen_worker.requests import GenerationRequest, InputError

POSITIVE_POSITIONS = 8192
NEGATIVE_POSITIONS = 1024
IMAGE_TOKENS_PER_SLOT = 4
IMAGE_MARKER = "<image1><|vision_start|><|image_pad|><|vision_end|>"


def reference_dimensions(width: int, height: int, resolution: int) -> tuple[int, int]:
    # Same rounding as pinned pipeline.calculate_dimensions; parity-tested.
    ratio = width / height
    target_width = math.sqrt(resolution * resolution * ratio)
    target_height = target_width / ratio
    return round(target_width / 32) * 32, round(target_height / 32) * 32


def validate_rope_layout(image_mask: list[bool], shapes: list[tuple[int, int, int]]) -> None:
    """Check exactly the indices emitted by pinned QwenImage21Rope.forward.

    Positive rows 8192..9215 actually contain negative frequencies. Testing
    only physical array bounds would incorrectly accept those positions.
    """
    if any(frame != 1 or height < 1 or width < 1 for frame, height, width in shapes):
        raise InputError("Unsupported image grid in prompt/reference layout")
    if sum(image_mask) != sum(height * width for _, height, width in shapes):
        raise InputError("Image tokens do not match the reference and target grids")

    def check_positive(last: int):
        if last >= POSITIVE_POSITIONS:
            raise InputError(
                "Prompt and reference layout exceed the model's positional limits. "
                "Shorten the prompt or reduce reference processing resolution/count."
            )

    cursor = position = 0
    for _, height, width in shapes:
        try:
            start = image_mask.index(True, cursor)
        except ValueError as exc:
            raise InputError("Missing image tokens in prompt/reference layout") from exc
        end = start + height * width
        if end > len(image_mask) or not all(image_mask[start:end]):
            raise InputError("Image tokens must form the expected reference and target grids")
        text_len = start - cursor
        if text_len:
            check_positive(position + text_len - 1)
        position += text_len
        check_positive(position)  # Every image token's frame coordinate.
        for side in (height, width):
            if side - side // 2 > NEGATIVE_POSITIONS:
                raise InputError("Reference grid exceeds the model's spatial positional limits")
            check_positive(side // 2 - 1)
        cursor = end
        position += max(height, width)
    if cursor < len(image_mask):
        check_positive(position + len(image_mask) - cursor - 1)


@dataclass(frozen=True)
class BranchLayout:
    name: str
    image_mask: list[bool]
    shapes: list[tuple[int, int, int]]


class RequestPreflight:
    def __init__(self, pipe):
        self.pipe = pipe
        processor = pipe.processor
        for method in ("_get_num_multimodal_tokens", "get_text_with_replacements"):
            if not callable(getattr(processor, method, None)):
                raise RuntimeError(f"The pinned processor must support {method}")
        if pipe.vae_scale_factor != 16 or processor.image_token != "<|image_pad|>":
            raise RuntimeError("Unsupported processor/latent layout for positional validation")
        if pipe._img_token_id != processor.image_token_id or IMAGE_MARKER not in pipe.prompt_template_ti2i:
            raise RuntimeError("Unsupported pipeline prompt template/image token")

    def layouts(self, request: GenerationRequest, references: list) -> list[BranchLayout]:
        pipe, processor = self.pipe, self.pipe.processor
        dimensions = [reference_dimensions(*image.size, request.output_resolution) for image in references]
        shapes = [(1, height // 16, width // 16) for width, height in dimensions]
        if dimensions:
            # Uses smart_resize from the real image processor, without pixels.
            counts = processor._get_num_multimodal_tokens(
                image_sizes=[(height, width) for width, height in dimensions]
            ).num_image_tokens
            if len(counts) != len(shapes) or any(
                int(count) * IMAGE_TOKENS_PER_SLOT != height * width
                for count, (_, height, width) in zip(counts, shapes)
            ):
                raise InputError("Reference processor slots do not match the model's latent grids")
            replacements = [processor.image_token * int(count) for count in counts]
            markers = " ".join(
                f"<image{i}><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(1, len(references) + 1)
            )
            template = pipe.prompt_template_ti2i.replace(IMAGE_MARKER, markers)
        else:
            replacements = []
            template = pipe.prompt_template_t2i

        target_shape = (1, request.height // 16, request.width // 16)
        target_tokens = target_shape[1] * target_shape[2]
        branches = [("positive", request.effective_prompt)]
        if request.true_cfg_scale > 1:
            branches.append(("negative", request.negative_prompt or " "))
        layouts = []
        for name, prompt in branches:
            text = template.format(prompt)
            if text.count(processor.image_token) != len(references):
                field = "prompt" if name == "positive" else "negative_prompt"
                raise InputError(f"{field} contains unsupported image placeholder tokens")
            expanded, _ = processor.get_text_with_replacements([text], replacements, [], [])
            # Same single-sample, left-padded tokenization as the pipeline.
            inputs = processor.tokenizer(expanded, padding=True, padding_side="left",
                                         return_attention_mask=True)
            valid_ids = [token for token, valid in zip(inputs["input_ids"][0], inputs["attention_mask"][0]) if valid]
            retained = valid_ids[pipe._drop_idx:]
            mask = [token == pipe._img_token_id for token in retained]
            # Each VLM image slot becomes four latent positions in the DiT.
            mask = [flag for flag in mask for _ in range(IMAGE_TOKENS_PER_SLOT if flag else 1)]
            mask.extend([True] * target_tokens)
            layouts.append(BranchLayout(name, mask, shapes + [target_shape]))
        return layouts

    def __call__(self, request: GenerationRequest, references: list) -> None:
        for layout in self.layouts(request, references):
            validate_rope_layout(layout.image_mask, layout.shapes)
