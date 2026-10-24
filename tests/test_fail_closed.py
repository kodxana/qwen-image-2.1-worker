"""CPU regressions for early rejection, warm-worker survival and inline routing."""

import base64
from contextlib import contextmanager
import importlib
import io
import json
import logging
import random
import re
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

from PIL import Image

from qwen_worker.config import Settings
from qwen_worker.images import decode_images, encode_image
from qwen_worker.output_budget import png_byte_bound
from qwen_worker.preflight import IMAGE_MARKER, RequestPreflight, validate_rope_layout
from qwen_worker.requests import GenerationRequest, InputError, MAX_SEED
from qwen_worker.service import ImageService, OUTPUT_ENVELOPE_HEADROOM
from scripts.audit_fail_closed import docker_context_includes


def as_base64(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def metadata_bomb():
    raw = base64.b64decode(as_base64(Image.new("RGB", (1, 1))))
    data = b"comment\0\0" + zlib.compress(b"x" * (2 * 1024**2))
    chunk = struct.pack(">I", len(data)) + b"zTXt" + data + struct.pack(">I", zlib.crc32(b"zTXt" + data) & 0xffffffff)
    return base64.b64encode(raw[:33] + chunk + raw[33:]).decode()


class Tokenizer:
    def __call__(self, texts, **kwargs):
        ids = [[999 if part == "<|image_pad|>" else 1
                for part in re.findall(r"<\|image_pad\|>|.", text, re.S)] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def fake_pipe():
    def expand(texts, replacements, *_):
        parts = iter(replacements)
        return [re.sub(r"<\|image_pad\|>", lambda match: next(parts), text) for text in texts], []

    processor = SimpleNamespace(
        tokenizer=Tokenizer(), image_token="<|image_pad|>", image_token_id=999,
        get_text_with_replacements=expand,
        _get_num_multimodal_tokens=lambda image_sizes: SimpleNamespace(
            num_image_tokens=[(h // 32) * (w // 32) for h, w in image_sizes]),
    )
    return SimpleNamespace(processor=processor, _img_token_id=999, _drop_idx=0, vae_scale_factor=16,
                           prompt_template_t2i="{}", prompt_template_ti2i=IMAGE_MARKER + "{}")


class MalformedInputTests(unittest.TestCase):
    def setUp(self):
        self.handler = importlib.import_module("handler")
        level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, level)
        self.pipeline = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (256, 256))]))
        self.service = ImageService(self.pipeline, lambda seed: seed, Settings())
        self.patch = patch.object(self.handler, "get_service", return_value=self.service)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def assert_rejected_warm(self, inputs):
        result = self.handler.handler({"input": inputs})
        self.assertEqual(set(result), {"error"})
        self.pipeline.assert_not_called()
        result = self.handler.handler({"input": {"prompt": "valid", "width": 256, "height": 256}})
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(self.pipeline.call_count, 1)

    def test_png_metadata_error_is_input_error_and_next_job_succeeds(self):
        self.assert_rejected_warm({"prompt": "test", "images": [metadata_bomb()]})

    def test_positive_surrogate_is_rejected_without_refresh(self):
        self.assert_rejected_warm(json.loads('{"prompt":"\\ud800"}'))

    def test_negative_surrogate_is_rejected_without_refresh(self):
        self.assert_rejected_warm({"prompt": "test", "negative_prompt": "\udfff", "true_cfg_scale": 2})

    def test_positional_rejection_keeps_the_same_worker_usable(self):
        self.service.preflight = RequestPreflight(fake_pipe())
        self.assert_rejected_warm({"prompt": "a" * 9000})

    def test_valid_multilingual_text_is_preserved(self):
        text = "شجرة خضراء 🌳 中文"
        request = GenerationRequest.parse({"prompt": text, "negative_prompt": text, "true_cfg_scale": 2})
        self.assertEqual(request.prompt, text)
        self.assertEqual(request.negative_prompt, text)

    def test_explicit_image_validation_errors_keep_their_message(self):
        with self.assertRaisesRegex(InputError, "aspect ratio"):
            decode_images((as_base64(Image.new("RGB", (1024, 1))),))

    def test_memory_error_is_not_classified_as_malformed_image(self):
        with patch("qwen_worker.images.Image.open", side_effect=MemoryError("host RAM")):
            with self.assertRaises(MemoryError):
                decode_images((as_base64(Image.new("RGB", (1, 1))),))

    def test_model_value_error_still_requests_refresh(self):
        self.pipeline.side_effect = ValueError("model failure")
        result = self.handler.handler({"input": {"prompt": "test"}})
        self.assertTrue(result["refresh_worker"])

    def test_cuda_oom_still_requests_refresh(self):
        class OOM(RuntimeError):
            pass
        self.pipeline.side_effect = OOM("device full")
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, is_available=lambda: False))
        with patch.object(self.handler, "torch", torch):
            result = self.handler.handler({"input": {"prompt": "test"}})
        self.assertTrue(result["refresh_worker"])
        self.assertIn("CUDA out of memory", result["error"])


class LayoutTests(unittest.TestCase):
    def test_positive_boundary_including_target_frame(self):
        validate_rope_layout([False] * 8191 + [True], [(1, 1, 1)])
        with self.assertRaises(InputError):
            validate_rope_layout([False] * 8192 + [True], [(1, 1, 1)])

    def test_negative_spatial_boundary_and_unused_final_cursor(self):
        validate_rope_layout([False] * 8191 + [True] * 2048, [(1, 1, 2048)])
        with self.assertRaises(InputError):
            validate_rope_layout([True] * 2049, [(1, 1, 2049)])

    def test_reference_grid_advances_position_before_later_text(self):
        validate_rope_layout([True] * 2048 + [False] * 6144, [(1, 1, 2048)])
        with self.assertRaises(InputError):
            validate_rope_layout([True] * 2048 + [False] * 6145, [(1, 1, 2048)])

    def test_wrong_count_and_fragmented_image_slots_are_rejected(self):
        for mask in ([True, False, True], [True, True, True, False]):
            with self.subTest(mask=mask), self.assertRaises(InputError):
                validate_rope_layout(mask, [(1, 2, 2)])
        with self.assertRaises(InputError):
            validate_rope_layout([True, False, True, True, True], [(1, 2, 2)])

    def test_adjacent_image_blocks_remain_separate(self):
        validate_rope_layout([True] * 8, [(1, 2, 2), (1, 2, 2)])

    def test_both_cfg_branches_validate_before_pipeline_and_cleanup_runs(self):
        pipeline, release, budget = Mock(), Mock(), Mock(return_value=10**15)
        service = ImageService(pipeline, lambda seed: seed, Settings(memory_guard=False),
                               memory_budget=budget, release_memory=release, preflight=RequestPreflight(fake_pipe()))
        with self.assertRaises(InputError):
            service.handle({"input": {"prompt": "short", "negative_prompt": "a" * 9000, "true_cfg_scale": 2}})
        pipeline.assert_not_called()
        budget.assert_not_called()
        release.assert_called_once()

    def test_transparency_text_participates_in_validation(self):
        validator = RequestPreflight(fake_pipe())
        inputs = {"prompt": "a" * 8150, "width": 256, "height": 256}
        validator(GenerationRequest.parse(inputs), [])
        with self.assertRaises(InputError):
            validator(GenerationRequest.parse({**inputs, "transparent": True}), [])

    def test_reserved_image_placeholders_are_rejected_without_conditioning(self):
        for references in ([], [SimpleNamespace(size=(1024, 1024))]):
            with self.assertRaisesRegex(InputError, "placeholder"):
                RequestPreflight(fake_pipe())(GenerationRequest.parse({"prompt": "<|image_pad|>"}), references)

    def test_preprocessor_slot_mismatch_is_rejected(self):
        pipe = fake_pipe()
        pipe.processor._get_num_multimodal_tokens = lambda **kwargs: SimpleNamespace(num_image_tokens=[1])
        with self.assertRaisesRegex(InputError, "slots"):
            RequestPreflight(pipe)(GenerationRequest.parse({"prompt": "test"}), [SimpleNamespace(size=(1024, 1024))])


class OutputTests(unittest.TestCase):
    def service(self, *, noise=False, settings=None, **extra):
        calls = []
        def pipeline(**kwargs):
            calls.append(kwargs)
            size = kwargs["width"], kwargs["height"]
            pixels = random.Random(42).randbytes(size[0] * size[1] * 4) if noise else None
            image = Image.frombytes("RGBA", size, pixels) if noise else Image.new("RGBA", size, (20, 40, 60, 128))
            return SimpleNamespace(images=[image.copy() for _ in range(kwargs.get("num_images_per_prompt", 1))])
        return ImageService(pipeline, lambda seed: seed, settings or Settings(), **extra), calls

    def test_small_safe_png_batch_remains_parallel(self):
        service, calls = self.service()
        result = service.handle({"input": {"prompt": "test", "width": 512, "height": 512, "num_images": 4, "seed": 42}})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["generator"], [42, 43, 44, 45])
        self.assertEqual([image["seed"] for image in result["images"]], [42, 43, 44, 45])

    def test_large_compressible_batch_runs_parallel_with_seed_wrap(self):
        entered, exited = Mock(), Mock()
        @contextmanager
        def context():
            entered()
            try:
                yield
            finally:
                exited()
        release = Mock()
        service, calls = self.service(release_memory=release, job_context=context)
        result = service.handle({"input": {"prompt": "test", "num_images": 4, "seed": MAX_SEED}})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["generator"], [MAX_SEED, 0, 1, 2])
        self.assertEqual(calls[0]["num_images_per_prompt"], 4)
        self.assertEqual(len(result["images"]), 4)
        self.assertEqual(set(result), {"model", "revision", "parameters", "images", "inference_seconds"})
        for item in result["images"]:
            self.assertEqual(set(item), {"image_base64", "mime_type", "width", "height", "seed"})
            image = Image.open(io.BytesIO(base64.b64decode(item["image_base64"])))
            self.assertEqual(image.getpixel((0, 0)), (20, 40, 60, 128))
        entered.assert_called_once()
        exited.assert_called_once()
        release.assert_called_once()

    def test_impossible_metadata_limit_rejects_before_gpu_and_keeps_worker_warm(self):
        release = Mock()
        service, calls = self.service(noise=True, settings=Settings(max_output_bytes=4096), release_memory=release)
        handler = importlib.import_module("handler")
        with patch.object(handler, "get_service", return_value=service), patch.object(handler.logger, "disabled", True):
            result = handler.handler({"input": {"prompt": "test", "width": 256, "height": 256, "num_images": 4}})
        self.assertEqual(set(result), {"error"})
        self.assertIn("inline output limit", result["error"])
        self.assertEqual(len(calls), 0)
        release.assert_called_once()

    def test_aggregate_overflow_is_compressed_without_regeneration_and_exits_job_context(self):
        active = []
        @contextmanager
        def context():
            active.append(True)
            try:
                yield
            finally:
                active.clear()
        service, calls = self.service(noise=True, settings=Settings(max_output_bytes=500_000), job_context=context)
        result = service.handle({"input": {"prompt": "test", "width": 256, "height": 256, "num_images": 4, "seed": 42}})
        self.assertEqual(len(calls), 1)
        self.assertEqual([item["seed"] for item in result["images"]], [42, 43, 44, 45])
        self.assertLessEqual(len(json.dumps(result).encode()) + OUTPUT_ENVELOPE_HEADROOM, 500_000)
        self.assertTrue(all(item["width"] == 256 and item["mime_type"] == "image/jpeg" for item in result["images"]))
        self.assertFalse(result["delivery"]["partial"])
        self.assertEqual(calls[0]["num_images_per_prompt"], 4)
        self.assertFalse(active)

    def test_requested_webp_and_jpeg_quality_and_count_are_preserved(self):
        for fmt in ("webp", "jpeg"):
            with self.subTest(fmt=fmt):
                service, calls = self.service()
                with patch("qwen_worker.service.encode_image", wraps=encode_image) as encoder:
                    result = service.handle({"input": {"prompt": "test", "width": 256, "height": 256,
                                                      "num_images": 2, "output_format": fmt, "quality": 73}})
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["num_images_per_prompt"], 2)
                self.assertEqual(len(result["images"]), 2)
                self.assertTrue(all(call.args[1:3] == (fmt, 73) for call in encoder.call_args_list))
                self.assertTrue(all(item["mime_type"] == f"image/{fmt}" for item in result["images"]))

    def test_exact_inline_cap_boundary_is_authoritative(self):
        inputs = {"prompt": "test", "width": 256, "height": 256, "seed": 42}
        with patch("qwen_worker.service.time.perf_counter", return_value=1):
            service, _ = self.service()
            result = service.handle({"input": inputs})
            length = len(json.dumps(result, ensure_ascii=True).encode())
            for delta, passes in ((0, True), (-1, False)):
                service, _ = self.service(settings=Settings(max_output_bytes=length + OUTPUT_ENVELOPE_HEADROOM + delta))
                if passes:
                    self.assertEqual(service._check_output_size(result), length)
                else:
                    with self.assertRaises(InputError):
                        service._check_output_size(result)

    def test_png_bound_covers_real_flat_and_high_entropy_encodings(self):
        for size in ((1, 1), (256, 256), (512, 256), (2752, 32), (32, 4096), (1024, 1024)):
            for mode in ("RGB", "RGBA"):
                for noisy in (False, True):
                    with self.subTest(size=size, mode=mode, noisy=noisy):
                        image = (Image.frombytes(mode, size, random.Random(42).randbytes(size[0] * size[1] * len(mode)))
                                 if noisy else Image.new(mode, size))
                        out = io.BytesIO()
                        image.save(out, format="PNG")
                        self.assertLessEqual(len(out.getvalue()), png_byte_bound(*size))

    def test_unknown_encoder_settings_have_no_claimed_bound(self):
        with patch("qwen_worker.output_budget.pillow_version", "unknown"):
            self.assertIsNone(png_byte_bound(256, 256))
        with patch("qwen_worker.output_budget.ImageFile.MAXBLOCK", 1):
            self.assertIsNone(png_byte_bound(256, 256))


class ContainerContextTests(unittest.TestCase):
    def test_required_sources_survive_dockerignore_and_private_files_do_not(self):
        text = (Path(__file__).resolve().parents[1] / ".dockerignore").read_text()
        for path in ("qwen_worker/preflight.py", "qwen_worker/output_budget.py", "qwen_worker/transport.py", "scripts/benchmark_vram.py",
                     "scripts/vram_matrix.py", "docs/vram/fixture.json", "docs/qwen-image-2.1-spec.json"):
            self.assertTrue(docker_context_includes(path, text), path)
        for path in (".env", "notes.md", ".local-dev/verify-env/python.exe", "outputs/private.json"):
            self.assertFalse(docker_context_includes(path, text), path)
        before_docs_allowlist = text.split("!docs/")[0]
        self.assertFalse(docker_context_includes("docs/vram/fixture.json", before_docs_allowlist))

    def test_parent_exceptions_and_later_exclusions_are_evaluated_in_order(self):
        text = "**\n!docs/\ndocs/private/\n!docs/private/public.json\n"
        self.assertTrue(docker_context_includes("docs/file.json", text))
        self.assertFalse(docker_context_includes("docs/private/secret.json", text))
        self.assertTrue(docker_context_includes("docs/private/public.json", text))


if __name__ == "__main__":
    unittest.main()
