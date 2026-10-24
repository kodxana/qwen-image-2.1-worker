"""Real Pillow encodings and exact JSON accounting, with no CUDA or API jobs."""

import base64
import io
import json
import random
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from qwen_worker.config import Settings
from qwen_worker.images import encode_image
from qwen_worker.memory import GIB
from qwen_worker.requests import GenerationRequest, InputError
from qwen_worker.service import ImageService, OUTPUT_ENVELOPE_HEADROOM
from qwen_worker.transport import fit_output, json_bytes


def decode(item):
    image = Image.open(io.BytesIO(base64.b64decode(item["image_base64"])))
    image.load()
    return image


class TransportTests(unittest.TestCase):
    def noisy(self, size=(256, 256)):
        return Image.frombytes("RGBA", size, random.Random(42).randbytes(size[0] * size[1] * 4))

    def test_real_over_ten_mb_response_fits_existing_eight_mb_cap_without_regeneration(self):
        image = self.noisy((2048, 1024))
        original = encode_image(image, "png", 95, 42)
        self.assertGreater(json_bytes({"images": [original]}), 10_000_000)
        calls = []
        def pipeline(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[image])
        service = ImageService(pipeline, lambda seed: seed, Settings())
        result = service.handle({"id": "large-png", "input": {
            "prompt": "test", "width": 2048, "height": 1024, "seed": 42}})
        wrapped = {"id": "large-png", "status": "COMPLETED", "output": result}
        self.assertLessEqual(len(json.dumps(wrapped).encode()), 8_000_000)
        self.assertLessEqual(json_bytes(result) + OUTPUT_ENVELOPE_HEADROOM, 8_000_000)
        self.assertEqual(len(calls), 1)
        item = result["images"][0]
        decoded = decode(item)
        self.assertEqual(decoded.size, (item["width"], item["height"]))
        self.assertLess(item["width"], 2048)
        self.assertEqual(decoded.mode, "RGBA")
        self.assertLess(decoded.getchannel("A").getextrema()[0], 255)
        self.assertEqual(item["seed"], 42)

    def test_lossless_png_optimization_preserves_exact_pixels_and_dimensions(self):
        image = Image.new("RGBA", (512, 512), (20, 40, 60, 128))
        request = GenerationRequest.parse({"prompt": "test", "width": 512, "height": 512})
        result = {"images": [encode_image(image, "png", 95, 42)]}
        optimized = {"images": [encode_image(image, "png", 95, 42, optimize=True)]}
        self.assertGreater(json_bytes(result) - json_bytes(optimized), 64)
        fit_output(result, [(image, 42)], request, json_bytes(result) - 1)
        decoded = decode(result["images"][0])
        self.assertEqual(decoded.size, image.size)
        self.assertEqual(decoded.tobytes(), image.tobytes())

    def test_quality_reduction_preserves_format_and_size_and_webp_alpha(self):
        image = self.noisy()
        for fmt, budget in (("jpeg", 60_000), ("webp", 180_000)):
            with self.subTest(fmt=fmt):
                request = GenerationRequest.parse({"prompt": "test", "output_format": fmt, "quality": 95})
                result = {"images": [encode_image(image, fmt, 95, 42)]}
                self.assertGreater(json_bytes(result), budget)
                with patch("qwen_worker.transport.encode_image", wraps=encode_image) as encoder:
                    fit_output(result, [(image, 42)], request, budget)
                item = result["images"][0]
                decoded = decode(item)
                self.assertEqual(decoded.size, image.size)
                self.assertEqual(item["mime_type"], f"image/{fmt}")
                self.assertEqual(item["seed"], 42)
                self.assertLessEqual(json_bytes(result), budget)
                self.assertTrue(any(call.args[2] < 95 for call in encoder.call_args_list))
                if fmt == "webp":
                    self.assertEqual(decoded.getchannel("A").tobytes(), image.getchannel("A").tobytes())

    def test_compression_never_increases_requested_quality(self):
        image = self.noisy()
        request = GenerationRequest.parse({"prompt": "test", "output_format": "webp", "quality": 1})
        result = {"images": [encode_image(image, "webp", 1, 42)]}
        with patch("qwen_worker.transport.encode_image", wraps=encode_image) as encoder:
            fit_output(result, [(image, 42)], request, 40_000)
        self.assertTrue(all(call.args[2] == 1 for call in encoder.call_args_list))
        self.assertLessEqual(json_bytes(result), 40_000)

    def test_fitting_response_is_not_reencoded(self):
        image = Image.new("RGBA", (256, 256))
        result = {"images": [encode_image(image, "png", 95, 42)]}
        before = json.dumps(result)
        with patch("qwen_worker.transport.encode_image", side_effect=AssertionError("must not encode")):
            fit_output(result, [(image, 42)], GenerationRequest.parse({"prompt": "test"}), 8_000_000)
        self.assertEqual(json.dumps(result), before)

    def test_larger_custom_limit_cannot_bypass_ten_mb_ceiling(self):
        service = ImageService(None, None, Settings(max_output_bytes=20_000_000))
        from qwen_worker.requests import InputError
        with self.assertRaises(InputError):
            service._check_output_size({"images": [{"image_base64": "x" * 10_000_000}]})

    def test_jpeg_fallback_lowers_quality_before_dropping_images_and_reports_alpha(self):
        image = self.noisy()
        request = GenerationRequest.parse({"prompt": "test", "width": 256, "height": 256,
                                           "num_images": 2, "transparent": True})
        originals = [(Image.new("RGBA", image.size, (50, 100, 150, 0)), 42), (image, 43)]
        result = {"images": [encode_image(img, "png", 95, seed) for img, seed in originals]}
        with patch("qwen_worker.transport.encode_image", wraps=encode_image) as encoder:
            fit_output(result, originals, request, 32_000)
        self.assertEqual(len(result["images"]), 2)
        self.assertTrue(all(item["mime_type"] == "image/jpeg" for item in result["images"]))
        self.assertEqual(decode(result["images"][0]).getpixel((0, 0)), (255, 255, 255))
        self.assertTrue(any(call.args[1] == "jpeg" and call.args[2] < 85 for call in encoder.call_args_list))
        self.assertIn("flattened on white", " ".join(result["delivery"]["warnings"]))
        self.assertFalse(result["delivery"]["partial"])
        self.assertLessEqual(json_bytes(result), 32_000)

    def test_partial_batch_keeps_maximum_count_with_seed_order_and_metadata_in_budget(self):
        image = self.noisy()
        originals = [(image, seed) for seed in (42, 43, 44, 45)]
        request = GenerationRequest.parse({"prompt": "test", "width": 256, "height": 256, "num_images": 4})
        one = json_bytes(encode_image(image, "jpeg", 1, 42, optimize=True, lossy=True))
        budget = 2 * one + 1500
        result = {"images": [encode_image(img, "png", 95, seed) for img, seed in originals]}
        fit_output(result, originals, request, budget)
        self.assertEqual([item["seed"] for item in result["images"]], [42, 43])
        self.assertEqual(result["delivery"]["returned_images"], 2)
        self.assertEqual(result["delivery"]["requested_images"], 4)
        self.assertTrue(result["delivery"]["partial"])
        self.assertLessEqual(json_bytes(result), budget)

    def test_subset_search_can_skip_large_first_image_to_return_more_later_images(self):
        small = self.noisy()
        originals = [(self.noisy((1024, 1024)), 42), (small, 43), (small, 44), (small, 45)]
        request = GenerationRequest.parse({"prompt": "test", "num_images": 4})
        one = json_bytes(encode_image(small, "jpeg", 1, 43, optimize=True, lossy=True))
        budget = 3 * one + 1500
        result = {"images": [encode_image(img, "png", 95, seed) for img, seed in originals]}
        fit_output(result, originals, request, budget)
        self.assertEqual([item["seed"] for item in result["images"]], [43, 44, 45])
        self.assertLessEqual(json_bytes(result), budget)

    def test_sequential_partial_delivery_skips_remaining_gpu_calls(self):
        image = self.noisy()
        calls = []
        def pipeline(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[image])
        one = json_bytes(encode_image(image, "jpeg", 1, 42, optimize=True, lossy=True))
        service = ImageService(pipeline, lambda seed: seed,
                               Settings(max_output_bytes=one + 1500 + OUTPUT_ENVELOPE_HEADROOM),
                               memory_budget=lambda: 4 * GIB)
        result = service.handle({"input": {"prompt": "test", "width": 256, "height": 256, "num_images": 4, "seed": 42}})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result["images"]), 1)
        self.assertTrue(result["delivery"]["partial"])
        self.assertLessEqual(json_bytes(result) + OUTPUT_ENVELOPE_HEADROOM, service.settings.max_output_bytes)


class GenerationRoutingTests(unittest.TestCase):
    def pipeline(self, calls):
        def generate(**kwargs):
            calls.append(kwargs)
            image = Image.new("RGBA", (kwargs["width"], kwargs["height"]), (20, 40, 60, 128))
            return SimpleNamespace(images=[image] * kwargs.get("num_images_per_prompt", 1))
        return generate

    def test_four_image_batch_uses_memory_admission_for_every_format_and_response_budget(self):
        reference = encode_image(Image.new("RGB", (1024, 1024)), "png", 95, 0)["image_base64"]
        for fmt in ("png", "jpeg", "webp"):
            for limit in (8_000_000, 500_000):
                with self.subTest(format=fmt, response_limit=limit):
                    calls = []
                    # Comparable to the model-offloaded A40 phase budgets in the job log.
                    service = ImageService(self.pipeline(calls), lambda seed: seed,
                                           Settings(offload="model", max_output_bytes=limit),
                                           phase_budgets=lambda: {"conditioning": 27 * GIB,
                                                                  "denoising": 30 * GIB, "decoding": 43 * GIB})
                    result = service.handle({"input": {"prompt": "test", "width": 1024, "height": 1024,
                        "images": [reference] * 3, "num_images": 4, "use_kv_cache": False,
                        "output_format": fmt, "seed": 42}})
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]["num_images_per_prompt"], 4)
                    self.assertEqual(calls[0]["generator"], [42, 43, 44, 45])
                    self.assertEqual([image["seed"] for image in result["images"]], [42, 43, 44, 45])
                    self.assertLessEqual(json_bytes(result) + OUTPUT_ENVELOPE_HEADROOM, limit)

    def test_each_phase_budget_retains_byte_exact_admission(self):
        inputs = {"prompt": "test", "width": 1024, "height": 1024, "num_images": 4, "seed": 42}
        request = GenerationRequest.parse(inputs)
        probe = ImageService(None, None, Settings())
        need, _ = probe.check_memory(request, [])
        parallel = {"conditioning": need.conditioning_bytes, "denoising": need.parallel_total_bytes,
                    "decoding": need.parallel_decoding_bytes}
        sequential = {"conditioning": need.conditioning_bytes, "denoising": need.total_bytes,
                      "decoding": need.decoding_bytes}
        for phase in parallel:
            for delta in (0, -1):
                with self.subTest(phase=phase, delta_bytes=delta):
                    budgets = {**parallel, phase: parallel[phase] + delta}
                    calls = []
                    service = ImageService(self.pipeline(calls), lambda seed: seed, Settings(),
                                           phase_budgets=lambda: budgets)
                    if budgets[phase] < sequential[phase]:
                        with self.assertRaises(InputError):
                            service.handle({"input": inputs})
                        self.assertEqual(calls, [])
                    else:
                        result = service.handle({"input": inputs})
                        self.assertEqual(len(calls), 1 if delta == 0 else 4)
                        self.assertEqual(len(result["images"]), 4)


class JobLogTests(unittest.TestCase):
    def test_final_summary_reports_actual_route_time_and_whole_job_peaks(self):
        width = 1024
        for budget, expected in ((80 * GIB, "parallel"), (4 * GIB, "sequential")):
            @contextmanager
            def context():
                stats = {}
                try:
                    yield stats
                finally:
                    stats.update(allocated=3 * 2**30, reserved=4 * 2**30)
            def pipeline(**kwargs):
                return SimpleNamespace(images=[Image.new("RGBA", (width, width))
                                              for _ in range(kwargs.get("num_images_per_prompt", 1))])
            service = ImageService(pipeline, lambda seed: seed, Settings(), job_context=context,
                                   memory_budget=lambda: budget)
            with self.subTest(route=expected), self.assertLogs("qwen_worker.service", level="INFO") as logs:
                service.handle({"id": "summary", "input": {
                    "prompt": "test", "width": width, "height": width, "num_images": 4}})
            summary = next(line for line in logs.output if "COMPLETE:" in line)
            for text in (f"route={expected}", "images=4/4", "elapsed=", "pipeline=", "encoding=",
                         "peak VRAM allocated=3.00 GiB, reserved=4.00 GiB", "response="):
                self.assertIn(text, summary)

    def test_failure_summary_and_cleanup_preserve_original_exception(self):
        @contextmanager
        def context():
            stats = {}
            try:
                yield stats
            finally:
                stats.update(allocated=5 * 2**30, reserved=6 * 2**30)
        def pipeline(**kwargs):
            raise RuntimeError("original failure")
        service = ImageService(pipeline, lambda seed: seed, Settings(), job_context=context)
        with self.assertLogs("qwen_worker.service", level="WARNING") as logs:
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                service.handle({"id": "failure", "input": {"prompt": "test"}})
        self.assertIn("FAILED:", logs.output[-1])
        self.assertIn("peak VRAM allocated=5.00 GiB, reserved=6.00 GiB", logs.output[-1])


if __name__ == "__main__":
    unittest.main()
