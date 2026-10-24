"""Serving discovers cached models; only explicit tests/provisioning use selectors."""

import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qwen_worker import runtime
from qwen_worker.config import BASE_MODEL_ID, MODEL_ID, MODEL_REVISION, SUPPORTED_MODELS, Settings
from test_runtime import make_snapshot


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "hub"
        self.addCleanup(patch.stopall)
        patch.object(runtime, "configure_cache", return_value=self.root).start()
        patch.object(runtime, "_cache_roots", return_value=[self.cache]).start()
        patch.object(runtime, "_log_model_cache").start()

    def snapshot(self, model_id, revision="d" * 40, *, root=None):
        path = (root or self.cache) / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
        return make_snapshot(path, turbo=model_id == MODEL_ID)

    def test_production_selects_either_cached_model_despite_stale_model_env(self):
        for model_id in SUPPORTED_MODELS:
            with self.subTest(model=model_id), tempfile.TemporaryDirectory() as temp:
                cache = Path(temp) / "hub"
                path = self.snapshot(model_id, root=cache)
                real_import = __import__
                def guarded_import(name, *args, **kwargs):
                    if name.split(".")[0] in {"torch", "diffusers", "huggingface_hub"}:
                        raise AssertionError(f"Must discover cache before importing {name}")
                    return real_import(name, *args, **kwargs)
                with patch.dict(os.environ, {"MODEL_ID": "stale/invalid", "MODEL_REVISION": "wrong",
                                             "CPU_OFFLOAD": "model"}, clear=True), \
                        patch.object(runtime, "_cache_roots", return_value=[cache]), \
                        patch.object(runtime, "download_model") as download, \
                        patch.object(Path, "mkdir", side_effect=AssertionError("read-only cache")), \
                        patch("builtins.__import__", side_effect=guarded_import), \
                        self.assertLogs(runtime.logger, level="INFO") as logs:
                    selected = runtime.prepare_startup_model(Settings.from_env())
                self.assertEqual(selected.model_id, model_id)
                self.assertEqual(selected.model_revision, "d" * 40)
                self.assertEqual(selected.model_path, str(path))
                self.assertEqual(selected.offload, "model")
                self.assertTrue(any(f"Selected model={model_id}" in line for line in logs.output))
                download.assert_not_called()

    def test_empty_incomplete_broken_and_unsupported_caches_fail_before_queue(self):
        for state in ("empty", "partial", "broken", "unsupported", "missing-schedule"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temp:
                cache = Path(temp) / "hub"
                if state != "empty":
                    model_id = "Other/Checkpoint" if state == "unsupported" else MODEL_ID
                    path = self.snapshot(model_id, root=cache)
                    if state in {"partial", "broken"}:
                        shard = path / "transformer" / "model.safetensors"
                        shard.unlink()
                        if state == "broken":
                            shard.symlink_to(Path(temp) / "missing-volume" / "weights")
                    elif state == "missing-schedule":
                        index = json.loads((path / "model_index.json").read_text())
                        index.pop("sample_sigmas")
                        (path / "model_index.json").write_text(json.dumps(index))
                start = Mock()
                with patch.dict(os.environ, {}, clear=True), \
                        patch.object(runtime, "_cache_roots", return_value=[cache]), \
                        patch.object(runtime, "download_model") as download, \
                        patch.object(runtime, "load_service") as load, \
                        patch.dict(sys.modules, {"runpod": SimpleNamespace(serverless=SimpleNamespace(start=start))}), \
                        patch.object(sys, "argv", ["handler.py"]), \
                        self.assertLogs(runtime.logger, level="INFO") as logs:
                    with self.assertRaisesRegex(runtime.ModelCacheError, "Downloads are disabled"):
                        runpy.run_path(str(Path(__file__).resolve().parents[1] / "handler.py"), run_name="__main__")
                self.assertTrue(any("[STARTUP FATAL]" in line for line in logs.output))
                download.assert_not_called()
                load.assert_not_called()
                start.assert_not_called()

    def test_ambiguous_models_or_revisions_are_not_chosen_arbitrarily(self):
        for second_model, second_revision in ((MODEL_ID, "e" * 40), (BASE_MODEL_ID, "e" * 40)):
            with self.subTest(model=second_model), tempfile.TemporaryDirectory() as temp:
                cache = Path(temp) / "hub"
                self.snapshot(BASE_MODEL_ID, root=cache)
                self.snapshot(second_model, second_revision, root=cache)
                with patch.object(runtime, "_cache_roots", return_value=[cache]), \
                        self.assertRaisesRegex(runtime.ModelCacheError, "Ambiguous model cache.*MODEL_PATH"):
                    runtime.detect_cached_model(Settings())

    def test_same_model_revision_in_two_roots_is_one_candidate(self):
        path = self.snapshot(MODEL_ID)
        second_root = self.root / "mirror" / "hub"
        self.snapshot(MODEL_ID, root=second_root)
        with patch.object(runtime, "_cache_roots", return_value=[self.cache, second_root]):
            selected = runtime.detect_cached_model(Settings())
        self.assertEqual(selected.model_path, str(path))

    def test_complete_model_is_not_blocked_by_incomplete_other_model(self):
        path = self.snapshot(BASE_MODEL_ID)
        partial = self.snapshot(MODEL_ID)
        (partial / "vae" / "model.safetensors").unlink()
        with self.assertLogs(runtime.logger, level="WARNING") as logs:
            selected = runtime.detect_cached_model(Settings())
        self.assertEqual(selected.model_path, str(path))
        self.assertTrue(any("Incomplete/invalid" in line for line in logs.output))

    def test_explicit_local_path_detects_variant_without_fabricating_revision(self):
        for model_id in SUPPORTED_MODELS:
            with self.subTest(model=model_id):
                path = make_snapshot(self.root / model_id.split("/")[-1], turbo=model_id == MODEL_ID)
                selected = runtime.detect_cached_model(Settings(model_path=str(path)))
                self.assertEqual(selected.model_id, model_id)
                self.assertIsNone(selected.model_revision)
        with self.assertRaisesRegex(runtime.ModelCacheError, "Invalid cached MODEL_PATH"):
            runtime.detect_cached_model(Settings(model_path=str(self.root / "absent")))

    def test_production_cache_error_is_logged_clearly(self):
        with patch.dict(os.environ, {}, clear=True), self.assertLogs(runtime.logger, level="CRITICAL") as logs:
            with self.assertRaisesRegex(runtime.ModelCacheError, "Check Runpod's cached Model setting"):
                runtime.prepare_startup_model(Settings())
        self.assertIn("[STARTUP FATAL] production cache detection failed", logs.output[-1])
        self.assertIn(str(self.cache), logs.output[-1])

    def test_production_registers_after_detection_and_reuses_selected_service(self):
        for model_id in SUPPORTED_MODELS:
            with self.subTest(model=model_id), tempfile.TemporaryDirectory() as temp:
                cache = Path(temp) / "hub"
                path = self.snapshot(model_id, root=cache)
                events = []
                service = SimpleNamespace(handle=Mock(return_value={"images": []}))
                def load(settings):
                    self.assertEqual(settings.model_id, model_id)
                    self.assertEqual(settings.model_path, str(path))
                    self.assertEqual(runtime.resolve_cached_model(settings), str(path))
                    events.append("load")
                    return service
                def register(config):
                    events.append("queue")
                    for _ in range(2):
                        config["handler"]({"id": "test", "input": {"prompt": "cube"}})
                with patch.dict(os.environ, {}, clear=True), \
                        patch.object(runtime, "_cache_roots", return_value=[cache]), \
                        patch.object(runtime, "load_service", side_effect=load) as loader, \
                        patch.object(runtime, "download_model") as download, \
                        patch.dict(sys.modules, {"runpod": SimpleNamespace(serverless=SimpleNamespace(start=register))}), \
                        patch.object(sys, "argv", ["handler.py"]):
                    runpy.run_path(str(Path(__file__).resolve().parents[1] / "handler.py"), run_name="__main__")
                self.assertEqual(events, ["load", "queue"])
                self.assertEqual(service.handle.call_count, 2)
                loader.assert_called_once()
                download.assert_not_called()

    def test_test_selection_uses_model_env_or_defaults_to_turbo(self):
        cases = ({}, {"MODEL_ID": ""}, {"MODEL_ID": MODEL_ID}, {"MODEL_ID": BASE_MODEL_ID},
                 {"MODEL_ID": BASE_MODEL_ID, "MODEL_REVISION": "f" * 40})
        for selectors in cases:
            with self.subTest(selectors=selectors):
                model_id = selectors.get("MODEL_ID") or MODEL_ID
                revision = selectors.get("MODEL_REVISION") or SUPPORTED_MODELS[model_id]
                path = self.snapshot(model_id, revision)
                with patch.dict(os.environ, {"QWEN_HUB_TEST_PROVISION": "true", **selectors}, clear=True), \
                        patch.object(runtime, "download_model", return_value=str(path)) as download, \
                        patch.object(runtime, "detect_cached_model") as detect, \
                        self.assertLogs(runtime.logger, level="INFO") as logs:
                    selected = runtime.prepare_startup_model(Settings(offload="model"))
                self.assertEqual(selected.model_id, model_id)
                self.assertEqual(selected.model_revision, revision)
                self.assertEqual(selected.offload, "model")
                self.assertEqual(download.call_args.args[0].model_id, model_id)
                self.assertEqual(download.call_args.args[0].model_revision, revision)
                self.assertTrue(any("[HUB TEST MODEL]" in line for line in logs.output))
                if not selectors.get("MODEL_ID"):
                    self.assertTrue(any("Turbo default" in line for line in logs.output))
                detect.assert_not_called()

    def test_bad_test_model_or_flag_fails_before_any_download(self):
        for env in ({"QWEN_HUB_TEST_PROVISION": "true", "MODEL_ID": "Other/Model"},
                    {"QWEN_HUB_TEST_PROVISION": "typo"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), \
                    patch.object(runtime, "download_model") as download, \
                    self.assertLogs(runtime.logger, level="CRITICAL") as logs:
                with self.assertRaises(ValueError):
                    runtime.prepare_startup_model(Settings())
            self.assertIn("[STARTUP FATAL]", logs.output[-1])
            download.assert_not_called()

    def test_test_provisioning_failure_logs_selection_and_fatal_reason(self):
        with patch.dict(os.environ, {"QWEN_HUB_TEST_PROVISION": "true", "MODEL_ID": BASE_MODEL_ID}, clear=True), \
                patch.object(runtime, "download_model", side_effect=ConnectionError("test download failed")), \
                self.assertLogs(runtime.logger, level="INFO") as logs:
            with self.assertRaisesRegex(ConnectionError, "test download failed"):
                runtime.prepare_startup_model(Settings())
        self.assertTrue(any("[HUB TEST MODEL]" in line and BASE_MODEL_ID in line for line in logs.output))
        self.assertIn("[STARTUP FATAL] Hub test provisioning failed", logs.output[-1])
        self.assertIn("test download failed", logs.output[-1])

    def test_bad_test_checkpoint_fails_before_gpu_load_and_queue_registration(self):
        path = make_snapshot(self.root / "invalid-turbo", turbo=False)
        start = Mock()
        with patch.dict(os.environ, {"QWEN_HUB_TEST_PROVISION": "true"}, clear=True), \
                patch.object(runtime, "download_model", return_value=str(path)), \
                patch.object(runtime, "load_service") as load, \
                patch.dict(sys.modules, {"runpod": SimpleNamespace(serverless=SimpleNamespace(start=start))}), \
                patch.object(sys, "argv", ["handler.py"]), \
                self.assertLogs(runtime.logger, level="CRITICAL") as logs:
            with self.assertRaisesRegex(ValueError, "Turbo checkpoint is missing sample_sigmas"):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "handler.py"), run_name="__main__")
        self.assertIn("[STARTUP FATAL] Hub test provisioning failed", logs.output[-1])
        load.assert_not_called()
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
