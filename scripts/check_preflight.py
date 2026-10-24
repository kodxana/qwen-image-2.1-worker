"""Verify positional admission with the real pinned CPU processor and RoPE.

Requires installed worker dependencies and a local processor directory, not
model weights or a CUDA GPU. A tiny fake encoder exposes upstream token masks;
this is preprocessing/index verification, not model/image-quality inference.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/review/preflight_verification.json")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import Qwen3VLProcessor
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    from qwen_worker.preflight import RequestPreflight, reference_dimensions, validate_rope_layout
    from qwen_worker.requests import GenerationRequest, InputError

    torch.set_num_threads(4)
    fixture = json.loads((ROOT / "docs/vram/fixture.json").read_text())
    for cls in (QwenImage21Pipeline, QwenImage21Rope):
        file = Path(inspect.getfile(cls))
        # Git may check out the identical source with CRLF on Windows.
        if hashlib.sha256(file.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != fixture["diffusers_source_sha256"][file.name]:
            raise RuntimeError("Use the pinned Diffusers source for this parity check")
    if hashlib.sha256((args.processor_path / "tokenizer.json").read_bytes()).hexdigest() != fixture["tokenizer_json_sha256"]:
        raise RuntimeError("Use the pinned checkpoint processor")

    class MaskEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.Identity()))

        def forward(self, input_ids, **kwargs):
            # One feature channel is enough for upstream mask extraction.
            hidden = self.model.language_model.norm(input_ids.float().unsqueeze(-1))
            return SimpleNamespace(hidden_states=[hidden])

    processor = Qwen3VLProcessor.from_pretrained(str(args.processor_path), local_files_only=True)
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=MaskEncoder(), processor=processor, transformer=None)
    model_output = pipe.image_processor.postprocess(torch.zeros(1, 4, 32, 32), output_type="pil")[0]
    if model_output.mode != "RGBA" or model_output.info:
        raise AssertionError("PNG routing requires metadata-free RGB(A) pipeline outputs")
    validator = RequestPreflight(pipe)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    verified = []
    samples = []
    for resolution in (1024, 2048):
        for count in (1, 4, 10):
            samples.append((resolution, [(64, 64)] * count, False))
    # EXIF orientation is handled by decode_images; include both orientations,
    # thin references, multiple references, and the transparent prompt wrapper.
    samples += [(512, [(256, 512), (512, 256)], True),
                (1024, [(2000, 20)], False), (2048, [(20, 2000)], True),
                (1024, [], True)]
    for resolution, sizes, transparent in samples:
        refs = [Image.new("RGBA", size, (40, 80, 120, 128)) for size in sizes]
        request = GenerationRequest.parse({"prompt": fixture["prompt"], "width": resolution, "height": resolution,
                                           "output_resolution": resolution, "true_cfg_scale": 2, "transparent": transparent})
        expected = validator.layouts(request, refs)
        resized = [pipe.image_processor.resize(ref, width=reference_dimensions(*ref.size, resolution)[0],
                                               height=reference_dimensions(*ref.size, resolution)[1]) for ref in refs]
        for layout, prompt in zip(expected, (request.effective_prompt, request.negative_prompt)):
            with torch.inference_mode():
                _, _, mask = pipe._get_qwen_prompt_embeds(prompt, resized or None, device=torch.device("cpu"))
            mask = torch.cat([mask[0], torch.ones((resolution // 16) ** 2 // 4, dtype=mask.dtype)])
            actual_mask = torch.repeat_interleave(mask, torch.where(mask, 4, 1)).bool()
            if actual_mask.tolist() != layout.image_mask:
                raise AssertionError(f"Upstream mask differs: {resolution}, {sizes}, {layout.name}")
            validate_rope_layout(layout.image_mask, layout.shapes)
            actual_freqs = rope(layout.shapes, actual_mask, torch.device("cpu"))
            if actual_freqs.shape[0] != len(layout.image_mask):
                raise AssertionError("Upstream positional length differs")
            verified.append({"resolution": resolution, "references": len(refs), "branch": layout.name,
                             "transparent": transparent, "position_count": len(layout.image_mask)})
        print(f"Verified {resolution}, {len(refs)} reference(s), both branches", flush=True)

    # The formerly admitted 19,999-character prompt, now rejected before encoder.
    long_request = GenerationRequest.parse({"prompt": "a " * 9999 + "a", "width": 512, "height": 512})
    try:
        validator(long_request, [])
    except InputError:
        long_rejected = True
    else:
        raise AssertionError("Long prompt was not rejected")
    negative_request = GenerationRequest.parse({"prompt": "short", "negative_prompt": long_request.prompt,
                                                 "true_cfg_scale": 2})
    try:
        validator(negative_request, [])
    except InputError:
        negative_rejected = True
    else:
        raise AssertionError("Long negative prompt was not rejected")

    # Compare exact upstream positions against validation at both physical and
    # semantic table boundaries, including reference-driven position advances.
    boundary = [([False] * 8191 + [True], [(1, 1, 1)], True),
                ([False] * 8192 + [True], [(1, 1, 1)], False),
                ([True] * 2048, [(1, 1, 2048)], True),
                ([True] * 2049, [(1, 1, 2049)], False)]
    for mask, shapes, valid in boundary:
        try:
            validate_rope_layout(mask, shapes)
        except InputError:
            if valid:
                raise
        else:
            if not valid:
                raise AssertionError("An aliased position was admitted")
        if valid:
            rope(shapes, torch.tensor(mask), torch.device("cpu"))

    class IndexProbe:
        def __init__(self):
            self.minimum = self.maximum = 0

        def to(self, device):
            return self

        def __getitem__(self, indices):
            self.minimum = int(indices.min())
            self.maximum = int(indices.max())
            return indices[:, None]

    # Execute unchanged upstream forward with recording frequency tables, so
    # semantic bounds can be checked even when an invalid positive index aliases
    # a physical negative-frequency row instead of throwing an IndexError.
    rng = random.Random(42)
    for _ in range(200):
        shapes, mask = [], []
        for block in range(rng.randint(1, 3)):
            height, width = rng.choice(((1, rng.randint(1, 2100)), (rng.randint(1, 2100), 2)))
            mask.extend([False] * rng.randint(0, 9000))
            mask.extend([True] * (height * width))
            shapes.append((1, height, width))
        mask.extend([False] * rng.randint(0, 100))
        probes = [IndexProbe() for axis in range(3)]
        QwenImage21Rope.forward(SimpleNamespace(freqs=probes), shapes, torch.tensor(mask), torch.device("cpu"))
        expected_valid = all(probe.minimum >= -1024 and probe.maximum < 8192 for probe in probes)
        try:
            validate_rope_layout(mask, shapes)
        except InputError:
            if expected_valid:
                raise AssertionError("Validator rejected indices supported by upstream")
        else:
            if not expected_valid:
                raise AssertionError("Validator admitted unsupported upstream indices")

    rng = random.Random(42)
    for _ in range(1000):
        width, height = rng.randint(32, 4000), rng.randint(32, 4000)
        resolution = rng.choice((512, 1024, 2048))
        official = calculate_dimensions(resolution**2, width / height)[:2]
        if reference_dimensions(width, height, resolution) != official:
            raise AssertionError("Reference resizing differs from upstream")
    result = {"kind": "CPU processor/RoPE verification; no model weights or GPU inference",
              "processor": type(processor).__name__, "image_processor": type(processor.image_processor).__name__,
              "verified_layouts": verified, "long_prompt_rejected": long_rejected,
              "long_negative_prompt_rejected": negative_rejected,
              "boundary_cases": len(boundary), "upstream_index_comparisons": 200,
              "metadata_free_rgba_output_verified": True,
              "reference_resize_comparisons": 1000}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(verified)} real layouts, {len(boundary)} boundaries, 200 index and 1000 resize comparisons")


if __name__ == "__main__":
    main()
