"""CPU validation of case coverage, exact payloads, and measurement bookkeeping."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from collections import Counter

from scripts.benchmark_vram import ProcessSampler, Recorder
from scripts.vram_matrix import ROOT, build_cases, merge_measurements
from qwen_worker.config import Settings
from qwen_worker.requests import GenerationRequest, InputError
from qwen_worker.service import ImageService


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / "docs/vram/fixture.json").read_text())
        self.spec = json.loads((ROOT / "docs/qwen-image-2.1-spec.json").read_text())
        self.cases = build_cases(self.fixture, self.spec)

    def case(self, case_id):
        return next(c for c in self.cases if c["case_id"] == case_id)

    def test_all_requested_combinations_are_present_once(self):
        dimensions = {(c["cfg_scale"], c["reference_count"], c["kv_cache"], c["execution"], c["output_width"])
                      for c in self.cases}
        self.assertEqual(len(self.cases), 48)
        self.assertEqual(len(dimensions), 48)
        self.assertEqual(len({c["case_id"] for c in self.cases}), 48)
        self.assertTrue(all(c["outputs_per_job"] == 4 for c in self.cases))
        self.assertTrue(all(c["output_width"] == c["reference_processing_resolution"] for c in self.cases))

    def test_unequal_cfg_prefix_lengths_are_counted_exactly(self):
        c = self.case("2048-cfg2-ref10-kvon-parallel")
        # Ten 128x128 reference grids per branch; positive 92, negative 70.
        expected = 4 * ((10 * 128 * 128 + 92) + (10 * 128 * 128 + 70)) * 524288
        self.assertEqual(c["exact_persistent_kv_bytes"], expected)
        self.assertEqual(expected / 2**30, 640.31640625)

    def test_parallel_cache_has_four_physical_copies(self):
        for c in self.cases:
            if c["execution"] == "sequential":
                parallel = self.case(c["case_id"].replace("sequential", "parallel"))
                self.assertEqual(parallel["exact_persistent_kv_bytes"], 4 * c["exact_persistent_kv_bytes"])

    def test_80gib_routing_matches_the_real_service_guard_for_all_cases(self):
        for c in self.cases:
            resident = c["exact_bf16_parameter_bytes_dit_and_vae"]
            weights = c["exact_bf16_parameter_bytes_all_components"]
            service = ImageService(None, None, Settings(), phase_budgets=lambda: {
                "conditioning": 80 * 2**30 - weights,
                "denoising": 80 * 2**30 - resident,
                "decoding": 80 * 2**30 - resident,
            })
            request = GenerationRequest.parse({
                "prompt": self.fixture["prompt"], "negative_prompt": None,
                "width": c["output_width"], "height": c["output_height"],
                "num_images": 4, "output_resolution": c["reference_processing_resolution"],
                "true_cfg_scale": c["cfg_scale"], "use_kv_cache": c["kv_cache"],
            })
            references = [SimpleNamespace(size=(1024, 1024))] * c["reference_count"]
            try:
                _, parallel = service.check_memory(request, references)
                actual = "parallel" if parallel else "sequential"
            except InputError:
                actual = "reject"
            self.assertEqual(c["ideal_80gib_guard_route_for_four_outputs"], actual, c["case_id"])

    def test_impossible_forced_rows_are_rejected_or_rerouted(self):
        impossible = [c for c in self.cases if c["forced_route_lower_bound_exceeds_80gib"]]
        self.assertEqual(len(impossible), 11)
        for c in impossible:
            route = c["ideal_80gib_guard_route_for_four_outputs"]
            self.assertTrue(route == "reject" or (route == "sequential" and c["execution"] == "parallel"))
        unique_requests = [c for c in self.cases if c["execution"] == "sequential"]
        self.assertEqual(Counter(c["ideal_80gib_guard_route_for_four_outputs"] for c in unique_requests),
                         {"parallel": 13, "sequential": 6, "reject": 5})
        self.assertEqual(Counter(c["ideal_80gib_final_route_for_four_outputs"] for c in unique_requests),
                         {"parallel": 13, "sequential": 6, "reject": 5})

    def test_cache_off_is_not_a_zero_vram_measurement(self):
        for c in self.cases:
            if not c["kv_cache"]:
                self.assertEqual(c["exact_persistent_kv_bytes"], 0)
                self.assertGreater(c["identified_eager_tensors_peak_lower_bound_bytes"], 30 * 2**30)
                self.assertIsNone(c["measured_job_peak_allocated_bytes"])
                self.assertIsNone(c["sampled_process_peak_bytes"])

    def test_uncached_eager_floor_includes_transients_not_both_cfg_workspaces(self):
        one = self.case("2048-cfg1-ref10-kvoff-parallel")
        two = self.case("2048-cfg2-ref10-kvoff-parallel")
        resident = self.spec["components"]["transformer"]["bf16_parameter_bytes"] + self.spec["components"]["vae"]["bf16_parameter_bytes"]
        triplet = 3 * 4 * (11 * 128 * 128 + 92) * 12288 * 2
        self.assertEqual(one["identified_eager_tensors_peak_lower_bound_bytes"], resident + triplet)
        self.assertEqual(one["identified_eager_tensors_peak_lower_bound_bytes"], two["identified_eager_tensors_peak_lower_bound_bytes"])

    def test_prefill_floor_accounts_for_separate_cfg_cache_lifetimes(self):
        c = self.case("2048-cfg2-ref1-kvon-parallel")
        resident = c["exact_bf16_parameter_bytes_dit_and_vae"]
        negative_triplet = 3 * 4 * (2 * 128 * 128 + 15) * 12288 * 2
        self.assertEqual(c["identified_eager_tensors_peak_lower_bound_bytes"], resident + c["exact_persistent_kv_bytes"] + negative_triplet)
        self.assertGreater(c["identified_eager_tensors_peak_lower_bound_bytes"], 80 * 2**30)

    def test_join_uses_successful_repeats_only_and_preserves_missing_cases(self):
        case = self.cases[0]
        result = {"case": case, "model_revision": self.spec["model_revision"], "status": "ok",
                  "completed": True, "expected_measured_repeats": 3,
                  "runtime_fingerprint": "test-runtime",
                  "runs": [{"label": f"measured_{i}", "status": "ok", "fixture_cache_match": True,
                            "measured_job_peak_allocated_bytes": n,
                            "measured_job_peak_reserved_bytes": n + 10,
                            "sampled_process_peak_bytes": None} for i, n in enumerate([100, 110, 105])]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / (case["case_id"] + ".json")
            path.write_text(json.dumps(result))
            joined = merge_measurements(self.cases, Path(tmp))
        self.assertEqual(joined[0]["measured_job_peak_allocated_bytes"], 110)
        self.assertEqual(joined[0]["measured_job_peak_allocated_bytes_median"], 105)
        self.assertIsNone(joined[0]["sampled_process_peak_bytes"])
        self.assertEqual(joined[1]["measurement_status"], "not_measured")

    def test_incomplete_run_is_not_promoted_to_measured(self):
        case = self.cases[0]
        value = {"case": case, "model_revision": self.spec["model_revision"], "status": "ok",
                 "completed": False, "expected_measured_repeats": 3, "runs": []}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / (case["case_id"] + ".json")).write_text(json.dumps(value))
            joined = merge_measurements(self.cases, Path(tmp))
        self.assertEqual(joined[0]["measurement_status"], "incomplete_measurements")
        self.assertIsNone(joined[0]["measured_job_peak_allocated_bytes"])

    def test_failed_case_can_be_joined_without_fabricating_measurements(self):
        case = self.cases[0]
        value = {"case": case, "model_revision": self.spec["model_revision"], "status": "timeout",
                 "completed": True, "runs": []}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / (case["case_id"] + ".json")).write_text(json.dumps(value))
            joined = merge_measurements(self.cases, Path(tmp))
        self.assertEqual(joined[0]["measurement_status"], "timeout")
        self.assertIsNone(joined[0]["measured_job_peak_allocated_bytes"])

    def test_different_gpu_runtime_fingerprints_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, case in enumerate(self.cases[:2]):
                value = {"case": case, "model_revision": self.spec["model_revision"], "status": "ok", "completed": True,
                         "expected_measured_repeats": 1, "runtime_fingerprint": f"different-gpu-{index}",
                         "runs": [{"label": "measured_1", "status": "ok", "fixture_cache_match": True,
                                   "measured_job_peak_allocated_bytes": 100, "measured_job_peak_reserved_bytes": 120}]}
                (Path(tmp) / (case["case_id"] + ".json")).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "different GPUs/runtimes"):
                merge_measurements(self.cases, Path(tmp))


class FakeCuda:
    def __init__(self):
        self.allocated = self.reserved = self.peak_allocated = self.peak_reserved = 0

    def use(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(allocated, self.peak_allocated)
        self.peak_reserved = max(reserved, self.peak_reserved)

    def synchronize(self):
        pass

    def reset_peak_memory_stats(self):
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved

    def max_memory_allocated(self):
        return self.peak_allocated

    def max_memory_reserved(self):
        return self.peak_reserved


class MeasurementTests(unittest.TestCase):
    def test_primary_measurements_do_not_insert_phase_barriers_or_resets(self):
        cuda = FakeCuda()
        recorder = Recorder(SimpleNamespace(cuda=cuda), {})
        with patch.object(cuda, "synchronize") as sync, patch.object(cuda, "reset_peak_memory_stats") as reset:
            with recorder.phase("primary"):
                cuda.use(30, 40)
        sync.assert_not_called()
        reset.assert_not_called()
        self.assertEqual(recorder.events, [])

    def test_earlier_peak_survives_later_counter_resets(self):
        cuda = FakeCuda()
        recorder = Recorder(SimpleNamespace(cuda=cuda), {}, profile_phases=True)
        recorder.reset_job()
        with recorder.phase("conditioning"):
            cuda.use(100, 120)
        cuda.use(20, 30)
        with recorder.phase("denoising"):
            cuda.use(60, 80)
        recorder.boundary()  # Worker would now reset for the next sequential call.
        cuda.reset_peak_memory_stats()
        with recorder.phase("second_output"):
            cuda.use(50, 70)
        self.assertEqual(recorder.peak_allocated, 100)
        self.assertEqual(recorder.peak_reserved, 120)

    def test_peak_between_phases_is_preserved(self):
        cuda = FakeCuda()
        recorder = Recorder(SimpleNamespace(cuda=cuda), {}, profile_phases=True)
        recorder.reset_job()
        cuda.use(200, 250)
        cuda.use(10, 20)
        with recorder.phase("next_phase"):
            cuda.use(30, 40)
        self.assertEqual(recorder.peak_allocated, 200)
        self.assertEqual(recorder.peak_reserved, 250)

    def test_unattributed_nvidia_samples_remain_missing(self):
        sampler = ProcessSampler()
        sampler.pids = {123}
        with patch("scripts.benchmark_vram.subprocess.run", return_value=SimpleNamespace(stdout="999, 1000\n123, [N/A]\n")):
            sampler.sample()
        self.assertIsNone(sampler.peak)
        self.assertEqual(sampler.samples, 0)

    def test_ambiguous_process_rows_are_not_summed_into_a_fake_peak(self):
        sampler = ProcessSampler()
        sampler.pids = {123}
        with patch("scripts.benchmark_vram.subprocess.run", return_value=SimpleNamespace(stdout="123, 100\n123, 200\n")):
            sampler.sample()
        self.assertIsNone(sampler.peak)
        self.assertEqual(sampler.samples, 0)
        self.assertIn("ambiguous", sampler.error)


if __name__ == "__main__":
    unittest.main()
