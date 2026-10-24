"""Measure the 48-case matrix on one real CUDA GPU, in isolated case processes.

Uses the actual worker, forces the requested routing, and records allocator
peaks across all sequential outputs and phases. Never substitutes estimates for
missing measurements. Run inside the worker's CUDA/dependency environment.
"""

import argparse
from contextlib import contextmanager
import csv
import functools
import hashlib
import inspect
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.vram_matrix import build_cases


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ProcessSampler:
    """Sample this process, not total card use. Unsupported attribution stays null."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = None
        self.samples = 0
        self.error = None
        self.pids = {os.getpid()}
        # nvidia-smi may report a host PID while Python is in a PID namespace.
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("NSpid:"):
                    self.pids.update(map(int, line.split()[1:]))
        except OSError:
            pass

    def sample(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            values = []
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) == 2 and row[0].strip().isdigit() and int(row[0]) in self.pids:
                    try:
                        values.append(int(row[1].strip()) * 1024**2)
                    except ValueError:
                        continue
            if len(values) == 1:
                value = values[0]
                self.peak = value if self.peak is None else max(self.peak, value)
                self.samples += 1
            elif len(values) > 1:
                self.error = "Multiple memory rows for this PID; attribution to the single benchmark GPU is ambiguous"
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = str(exc)

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.sample()


class Recorder:
    def __init__(self, torch, case, profile_phases=False):
        self.torch, self.case = torch, case
        self.profile_phases = profile_phases
        self.events = []
        self.observations = []
        self.peak_allocated = self.peak_reserved = 0
        self.transformer_index = 0
        self.invocation = 0

    def boundary(self):
        cuda = self.torch.cuda
        self.peak_allocated = max(self.peak_allocated, cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, cuda.max_memory_reserved())

    def reset_job(self):
        self.events, self.observations = [], []
        self.peak_allocated = self.peak_reserved = 0
        self.invocation = 0
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def phase(self, name):
        if not self.profile_phases:
            # Primary comparisons must not insert per-phase GPU barriers.
            yield
            return
        cuda = self.torch.cuda
        cuda.synchronize()
        # Preserve peaks outside named phases before resetting a phase counter.
        self.boundary()
        cuda.reset_peak_memory_stats()
        baseline = cuda.memory_allocated()
        started = time.perf_counter()
        try:
            yield
        finally:
            sync_error = None
            try:
                cuda.synchronize()
            except Exception as exc:
                sync_error = str(exc)
            self.boundary()
            self.events.append({
                "phase": name, "invocation": self.invocation,
                "baseline_allocated_bytes": baseline,
                "peak_allocated_bytes": cuda.max_memory_allocated(),
                "peak_reserved_bytes": cuda.max_memory_reserved(),
                "end_allocated_bytes": cuda.memory_allocated(),
                "end_reserved_bytes": cuda.memory_reserved(),
                "instrumented_seconds": time.perf_counter() - started,
                "synchronize_error": sync_error,
            })


def install_instrumentation(service, recorder):
    # The pinned worker's infer closure captures the pipeline under this name.
    pipe = inspect.getclosurevars(service.pipeline).nonlocals.get("pipe")
    if pipe is None:
        raise RuntimeError("Worker infer closure changed; review benchmark instrumentation")

    def wrap_phase(owner, attribute, name):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with recorder.phase(name):
                return original(*args, **kwargs)

        setattr(owner, attribute, wrapped)

    wrap_phase(pipe, "_get_qwen_prompt_embeds", "conditioning_or_job_cache_lookup")
    wrap_phase(pipe, "prepare_latents", "reference_encoding_and_target_noise")
    wrap_phase(pipe.vae, "decode", "vae_decode")
    original_forward = pipe.transformer.forward

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        branches = recorder.case["cfg_scale"]
        index = recorder.transformer_index
        recorder.transformer_index += 1
        branch = "positive" if index % branches == 0 else "negative"
        first_step = index < branches
        mode = kwargs.get("kv_cache_mode")
        phase = ("prefill" if first_step else "cached_denoising" if mode == "cached" else "full_denoising") + "_" + branch
        with recorder.phase(phase):
            result = original_forward(*args, **kwargs)
        if first_step:
            cache = kwargs.get("kv_cache")
            cache_bytes = sum(
                tensor.numel() * tensor.element_size()
                for layer in cache.layer_caches for tensor in (layer.k, layer.v) if tensor is not None
            ) if cache is not None else 0
            encoder_length = kwargs["encoder_hidden_states"].shape[1]
            shapes = kwargs["img_shapes"][0]
            refs = sum(f * h * w for f, h, w in shapes[:-1])
            # One VLM slot expands to four latent positions. Read CPU shape
            # metadata only; .sum().item() on img_mask would add a GPU barrier.
            slots = refs // 4
            recorder.observations.append({
                "invocation": recorder.invocation, "branch": branch,
                "non_image_tokens": encoder_length - slots,
                "prefix_tokens": encoder_length - slots + refs,
                "physical_batch_size": kwargs["hidden_states"].shape[0],
                "actual_cache_payload_bytes": cache_bytes,
                "cache_dtype": str(cache.layer_caches[0].k.dtype) if cache is not None else None,
            })
        return result

    pipe.transformer.forward = forward
    original_infer = service.pipeline

    @functools.wraps(original_infer)
    def infer(**kwargs):
        recorder.boundary()
        recorder.invocation += 1
        recorder.transformer_index = 0
        try:
            return original_infer(**kwargs)
        finally:
            # Runtime resets counters per call; preserve all sequential calls.
            recorder.torch.cuda.synchronize()
            recorder.boundary()

    service.pipeline = infer
    original_check = service.check_memory

    def force_requested_route(request, references):
        need, _ = original_check(request, references)
        return need, recorder.case["execution"] == "parallel"

    service.check_memory = force_requested_route
    return pipe


def make_references(count, fixture):
    import base64
    from PIL import Image, ImageDraw

    encoded, hashes = [], []
    size = fixture["source_reference_width"], fixture["source_reference_height"]
    for index in range(count):
        image = Image.new("RGB", size, ((index * 31 + 60) % 256, (index * 71 + 90) % 256, (index * 97 + 120) % 256))
        draw = ImageDraw.Draw(image)
        draw.rectangle((128, 128, 896, 896), outline="white", width=24)
        draw.ellipse((256, 256, 768, 768), fill=((index * 41) % 256, 30, 180))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        hashes.append(hashlib.sha256(data).hexdigest())
        encoded.append(base64.b64encode(data).decode("ascii"))
    return encoded, hashes


def run_job(service, recorder, job, label):
    recorder.reset_job()
    baseline_allocated = recorder.torch.cuda.memory_allocated()
    baseline_reserved = recorder.torch.cuda.memory_reserved()
    status, error = "ok", None
    started = time.perf_counter()
    with ProcessSampler() as sampler:
        try:
            output = service.handle(job)
            if len(output["images"]) != 4:
                raise RuntimeError("Benchmark did not generate four images")
            del output
        except Exception as exc:
            status = "oom" if isinstance(exc, recorder.torch.cuda.OutOfMemoryError) else "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            recorder.boundary()
    actual_by_invocation = {}
    for item in recorder.observations:
        actual_by_invocation.setdefault(item["invocation"], 0)
        actual_by_invocation[item["invocation"]] += item["actual_cache_payload_bytes"]
    actual_cache = max(actual_by_invocation.values(), default=0)
    case = recorder.case
    expected_invocations = 1 if case["execution"] == "parallel" else case["outputs_per_job"]
    expected_observations = expected_invocations * case["cfg_scale"]
    fixture_match = (status == "ok" and actual_cache == case["exact_persistent_kv_bytes"]
                     and len(recorder.observations) == expected_observations
                     and all(item["non_image_tokens"] == case[item["branch"] + "_non_image_tokens"]
                             and item["physical_batch_size"] == case["physical_batch_size"]
                             and item["prefix_tokens"] == case[item["branch"] + "_prefix_tokens"]
                             for item in recorder.observations))
    if status == "ok" and not fixture_match:
        status, error = "fixture_mismatch", "Observed cache payload differs from the exact matrix; do not compare this run as the same workload"
    return {
        "label": label, "status": status, "error": error,
        "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
        "observed_peak_allocated_bytes": recorder.peak_allocated,
        "observed_peak_reserved_bytes": recorder.peak_reserved,
        "measured_job_peak_allocated_bytes": recorder.peak_allocated if status == "ok" else None,
        "measured_job_peak_reserved_bytes": recorder.peak_reserved if status == "ok" else None,
        "sampled_process_peak_bytes": sampler.peak, "process_sample_count": sampler.samples,
        "process_sample_interval_seconds": sampler.interval, "process_sampling_error": sampler.error,
        "instrumented_job_seconds": time.perf_counter() - started,
        "phase_profiling_enabled": recorder.profile_phases,
        "actual_persistent_cache_payload_bytes": actual_cache,
        "fixture_cache_match": fixture_match,
        "phase_events": recorder.events, "prefix_observations": recorder.observations,
    }


def child(args, case, fixture, spec, result_path):
    result = {"case": case, "status": "initializing", "runs": [], "model_revision": spec["model_revision"],
              "completed": False, "expected_measured_repeats": args.repeats, "warmup_jobs": args.warmups}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA GPU is required; no peak measurement can be produced here")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU through --device / CUDA_VISIBLE_DEVICES")
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                         "compute_capability": [props.major, props.minor], "uuid": str(getattr(props, "uuid", "unavailable"))}
        if case["identified_eager_tensors_peak_lower_bound_bytes"] > props.total_memory and not args.attempt_impossible:
            result["status"] = "impossible_from_identified_tensor_lower_bound"
            result["completed"] = True
            save_json(result_path, result)
            return
        from importlib.metadata import version
        from qwen_worker.config import Settings
        from qwen_worker.runtime import load_service
        settings = Settings(model_id="Qwen/Qwen-Image-2.1", model_revision=spec["model_revision"],
                            model_path=os.environ.get("MODEL_PATH"), offload="text_encoder", vae_tiling=True,
                            memory_guard=False, max_output_bytes=1_000_000_000)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        service = load_service(settings)
        torch.cuda.synchronize()
        result["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        recorder = Recorder(torch, case)
        pipe = install_instrumentation(service, recorder)
        for name, component in spec["components"].items():
            module = getattr(pipe, name)
            if sum(t.numel() for t in module.parameters()) != component["parameter_count"]:
                raise RuntimeError(f"Unexpected parameter count in {name}; checkpoint differs from matrix")
            if any(t.dtype != torch.bfloat16 for t in module.parameters()):
                raise RuntimeError(f"Unexpected parameter dtype in {name}; matrix assumes BF16 parameters")
        result["versions"] = {name: version(name) for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "Pillow")}
        if result["versions"]["torch"].split("+")[0] != "2.8.0" or result["versions"]["transformers"] != "5.17.0":
            raise RuntimeError("Use the pinned worker runtime (PyTorch 2.8.0, Transformers 5.17.0) for this matrix")
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
        source_classes = [QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler]
        hashes = {Path(inspect.getfile(cls)).name: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for cls in source_classes}
        if hashes != fixture["diffusers_source_sha256"]:
            raise RuntimeError("Installed Diffusers source differs from the pinned implementation; refresh the analysis before comparing")
        result["diffusers_source_sha256"] = hashes
        result["diffusers_commit"] = spec["diffusers_commit"]
        result["cuda_runtime"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["allocator_backend"] = torch.cuda.memory.get_allocator_backend()
        result["allocator_config"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        result["sdp_flags"] = {name: getattr(torch.backends.cuda, name)() for name in
                               ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled")}
        try:
            result["nvidia_smi_gpu_metadata"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi_gpu_metadata"] = None
        result["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}
        result["attention_processors"] = sorted({type(block.attn.processor).__name__ for block in pipe.transformer.transformer_blocks})
        result["offload"] = "text_encoder"
        result["vae_tiling"] = result["vae_slicing"] = True
        result["routing_forced"] = True
        result["memory_guard_bypassed"] = True
        result["inline_output_limit_relaxed_for_benchmark"] = True
        result["runtime_source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "qwen_worker").glob("*.py")}
        images, hashes = make_references(case["reference_count"], fixture)
        result["reference_png_sha256"] = hashes
        result["fixture_sha256"] = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
        result["instrumentation_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fingerprint_fields = ("gpu", "versions", "cuda_runtime", "cudnn_version", "allocator_backend",
                              "allocator_config", "cuda_visible_devices", "sdp_flags", "tf32", "offload",
                              "vae_tiling", "vae_slicing", "attention_processors", "runtime_source_hashes",
                              "diffusers_source_sha256", "fixture_sha256", "instrumentation_source_sha256",
                              "warmup_jobs", "expected_measured_repeats")
        result["runtime_fingerprint"] = hashlib.sha256(json.dumps(
            {key: result[key] for key in fingerprint_fields}, sort_keys=True).encode()).hexdigest()
        inputs = {"prompt": fixture["prompt"], "width": case["output_width"], "height": case["output_height"],
                  "output_resolution": case["reference_processing_resolution"], "images": images,
                  "num_images": 4, "num_inference_steps": fixture["num_inference_steps"],
                  "seed": fixture["seed"], "true_cfg_scale": case["cfg_scale"], "use_kv_cache": case["kv_cache"]}
        if case["cfg_scale"] > 1:
            inputs["negative_prompt"] = fixture["negative_prompt"]
        job = {"id": case["case_id"], "input": inputs}
        labels = ["cold_job"] + [f"warmup_{i+1}" for i in range(args.warmups - 1)] + [f"measured_{i+1}" for i in range(args.repeats)]
        for label in labels:
            run = run_job(service, recorder, job, label)
            result["runs"].append(run)
            result["status"] = run["status"]
            save_json(result_path, result)
            if run["status"] != "ok":
                break  # Exit the process after OOM/error; do not reuse its state.
        if result["status"] == "ok" and args.profile_phases:
            recorder.profile_phases = True
            result["phase_profile"] = run_job(service, recorder, job, "phase_profile")
            # Diagnostic failures do not erase completed primary measurements.
    except Exception as exc:
        result["status"] = "initialization_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["completed"] = True
    save_json(result_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="One physical GPU index or UUID")
    parser.add_argument("--case", action="append", dest="case_ids", help="Exact case ID; repeat to select cases")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/vram")
    parser.add_argument("--warmups", type=int, default=1, help="Includes the first cold job; minimum one")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200, help="Seconds per isolated case")
    parser.add_argument("--attempt-impossible", action="store_true")
    parser.add_argument("--profile-phases", action="store_true", help="Run a separate synchronized diagnostic job after primary repeats")
    parser.add_argument("--list", action="store_true", help="List all cases without requiring CUDA")
    parser.add_argument("--child-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1 or args.timeout < 1:
        parser.error("warmups, repeats, and timeout must be positive")
    if "," in args.device:
        parser.error("Select one physical GPU, not a comma-separated GPU list")
    fixture = json.loads((ROOT / "docs/vram/fixture.json").read_text())
    spec = json.loads((ROOT / "docs/qwen-image-2.1-spec.json").read_text())
    cases = build_cases(fixture, spec)
    selected = [c for c in cases if not args.case_ids or c["case_id"] in args.case_ids]
    if args.case_ids and set(args.case_ids) - {c["case_id"] for c in cases}:
        parser.error("Unknown case ID; use --list")
    if args.list:
        for c in selected:
            print(c["case_id"])
        return
    if not args.child_case and shutil.which("nvidia-smi") is None:
        raise SystemExit("No NVIDIA GPU runtime found (nvidia-smi unavailable). Use --list here; run measurements on a CUDA host.")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.child_case:
        case = next(c for c in cases if c["case_id"] == args.child_case)
        child(args, case, fixture, spec, args.output_dir / (case["case_id"] + ".json"))
        return
    summaries = []
    for c in selected:
        case_id = c["case_id"]
        destination = args.output_dir / (case_id + ".json")
        if destination.exists():
            raise SystemExit(f"Refusing to overwrite {destination}; choose a new --output-dir")
        command = [sys.executable, str(Path(__file__).resolve()), "--child-case", case_id,
                   "--output-dir", str(args.output_dir), "--warmups", str(args.warmups), "--repeats", str(args.repeats)]
        if args.attempt_impossible:
            command.append("--attempt-impossible")
        if args.profile_phases:
            command.append("--profile-phases")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device)
        with (args.output_dir / (case_id + ".log")).open("w") as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                if not destination.exists():
                    save_json(destination, {"case": c, "status": "process_failed", "exit_code": completed.returncode,
                                            "model_revision": spec["model_revision"], "completed": True, "runs": []})
                elif completed.returncode != 0:
                    partial = json.loads(destination.read_text())
                    partial.update(status="process_failed", exit_code=completed.returncode, completed=True)
                    save_json(destination, partial)
            except subprocess.TimeoutExpired:
                save_json(destination, {"case": c, "status": "timeout", "model_revision": spec["model_revision"],
                                        "completed": True, "runs": []})
        value = json.loads(destination.read_text())
        summaries.append({"case_id": case_id, "status": value["status"], "result_file": str(destination)})
        save_json(args.output_dir / "index.json", summaries)
        print(case_id + ": " + value["status"], flush=True)


if __name__ == "__main__":
    main()
