"""Reproduce cost-related failure paths on CPU, with no model or paid jobs.

This is an audit of current behavior, not a test asserting that bugs are desired.
Optional pinned source/tokenizer files enable a real-source RoPE reproduction.
"""

import argparse
import ast
import base64
import fnmatch
import hashlib
import io
import json
import logging
from pathlib import Path, PurePosixPath
import random
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, __version__ as pillow_version
from qwen_worker.config import Settings
from qwen_worker.requests import GenerationRequest, InputError
from qwen_worker.service import ImageService
from qwen_worker.preflight import validate_rope_layout


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def invoke(inputs, image=None, phase_budgets=None):
    # Import the real entrypoint, but replace its GPU service. Never load weights.
    import handler

    def generate(**kwargs):
        return SimpleNamespace(images=[image or Image.new("RGB", (kwargs["width"], kwargs["height"]))
                                       for _ in range(kwargs.get("num_images_per_prompt", 1))])
    pipeline = Mock(side_effect=generate)
    service = ImageService(pipeline, lambda seed: seed, Settings(), phase_budgets=phase_budgets)
    with patch.object(handler, "get_service", return_value=service):
        result = handler.handler({"id": "cpu-audit", "input": inputs})
    batches = [call.kwargs.get("num_images_per_prompt", 1) for call in pipeline.call_args_list]
    return {"error": result.get("error"), "refresh_worker": result.get("refresh_worker", False),
            "pipeline_calls": pipeline.call_count, "physical_output_batches": batches, "generated_outputs": sum(batches)}


class IndexVector:
    """Only the indexing operations used by the pinned RoPE forward method."""
    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    def tolist(self):
        return self.values.copy()

    def clone(self):
        return IndexVector(self.values)

    def __setitem__(self, mask, values):
        indices = [i for i, flag in enumerate(mask.values) if flag]
        if len(indices) != len(values.values):
            raise ValueError("shape mismatch in RoPE audit")
        for i, value in zip(indices, values.values):
            self.values[i] = value


class FrequencyTable:
    def to(self, device):
        return self

    def __getitem__(self, indices):
        for i in indices.values:
            if not -9216 <= i < 9216:
                raise IndexError(f"index {i} is out of bounds for dimension 0 with size 9216")
        return IndexVector([0] * len(indices.values))


def reproduce_rope(source_path, tokenizer_path):
    fixture = json.loads((ROOT / "docs/vram/fixture.json").read_text())
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != fixture["diffusers_source_sha256"]["transformer_qwenimage21.py"]:
        raise ValueError("Upstream source does not match the pinned fixture")
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != fixture["tokenizer_json_sha256"]:
        raise ValueError("Tokenizer does not match the pinned fixture")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt = "a " * 9999 + "a"
    template = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
    verified_layouts = 0
    for resolution in (1024, 2048):
        slots = (resolution // 32) ** 2
        for references in (1, 4, 10):
            images = " ".join(f"<image{i}><|vision_start|>{'<|image_pad|>' * slots}<|vision_end|>"
                              for i in range(1, references + 1))
            for branch in ("positive", "negative"):
                text = fixture["prompt"] if branch == "positive" else fixture["negative_prompt"] or " "
                ids = tokenizer.encode(template.format(images + text)).ids[fixture["system_prefix_tokens_dropped"]:]
                image_count = ids.count(fixture["image_token_id"])
                actual = len(ids) - image_count
                expected = fixture["counts_by_reference_count"][str(references)][branch + "_non_image_tokens"]
                if image_count != references * slots or actual != expected:
                    raise AssertionError(f"Tokenizer fixture mismatch: {resolution}, {references}, {branch}")
                verified_layouts += 1
    retained = len(tokenizer.encode(template.format(prompt)).ids) - fixture["system_prefix_tokens_dropped"]
    tree = ast.parse(source.decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenImage21Rope")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    # Execute the source method unchanged. Substitute CPU list indexing for torch;
    # this proves index validity only, not GPU kernels or floating-point behavior.
    fake_torch = SimpleNamespace(Tensor=IndexVector, device=str, long=int,
                                tensor=lambda values, **kwargs: IndexVector(values),
                                cat=lambda values, **kwargs: IndexVector([]))
    namespace = {"torch": fake_torch}
    module = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)
    request = GenerationRequest.parse({"prompt": prompt, "width": 512, "height": 512})
    spec = json.loads((ROOT / "docs/qwen-image-2.1-spec.json").read_text())
    resident = sum(spec["components"][name]["bf16_parameter_bytes"] for name in ("transformer", "vae"))
    service = ImageService(None, None, Settings(), phase_budgets=lambda: {
        "conditioning": 80 * 2**30 - spec["bf16_parameter_bytes"],
        "denoising": 80 * 2**30 - resident, "decoding": 80 * 2**30 - resident})
    service.check_memory(request, [])
    result = {"prompt_characters": len(prompt), "retained_token_positions": retained,
              "matrix_tokenizer_layouts_verified": verified_layouts,
              "request_parser_accepts": True, "ideal_80gib_memory_guard_accepts": True,
              "source_sha256": hashlib.sha256(source).hexdigest(),
              "tokenizer_sha256": fixture["tokenizer_json_sha256"]}
    try:
        validate_rope_layout([False] * retained + [True] * 1024, [(1, 32, 32)])
    except InputError as exc:
        result["positional_preflight_rejects"] = True
        result["preflight_error"] = str(exc)
    else:
        raise AssertionError("Positional preflight failed to reject the reproduced layout")
    rope = SimpleNamespace(freqs=[FrequencyTable() for _ in range(3)])
    try:
        namespace["forward"](rope, [(1, 32, 32)], IndexVector([False] * retained + [True] * 1024), "cpu")
    except IndexError as exc:
        result["source_forward_error"] = str(exc)
    else:
        raise AssertionError("Expected the admitted long layout to exceed the pinned RoPE table")
    return result


def reproduce_container_payload(destination):
    # Recreate the relevant Docker COPY payload only. No Docker/GPU is required.
    # Keep it under the audit output directory so there is no hidden cleanup.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Container payload verification requires a new empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    ignore_text = (ROOT / ".dockerignore").read_text()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        fields = shlex.split(line)
        if len(fields) != 3 or fields[1].startswith("--"):
            raise ValueError("Review container-payload audit for changed Docker COPY syntax")
        source = (ROOT / fields[1]).resolve()
        source.relative_to(ROOT)
        target = destination / str(PurePosixPath(fields[2]).relative_to("/app"))
        target.resolve().relative_to(destination.resolve())
        if source.is_dir():
            for entry in source.rglob("*"):
                if entry.is_file() and "__pycache__" not in entry.parts and docker_context_includes(entry.relative_to(ROOT).as_posix(), ignore_text):
                    output = target / entry.relative_to(source)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(entry, output)
        else:
            if not docker_context_includes(source.relative_to(ROOT).as_posix(), ignore_text):
                raise ValueError(f"Docker COPY source excluded by .dockerignore: {fields[1]}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    completed = subprocess.run([sys.executable, str(destination / "scripts/benchmark_vram.py"), "--list"],
                               cwd=destination, capture_output=True, text=True, timeout=30)
    matrix_code = generated_cases = None
    if completed.returncode == 0:
        matrix = subprocess.run([sys.executable, str(destination / "scripts/vram_matrix.py"),
                                 "--output-dir", str(destination / "matrix-results")],
                                cwd=destination, capture_output=True, text=True, timeout=30)
        matrix_code = matrix.returncode
        if matrix_code == 0:
            generated_cases = json.loads((destination / "matrix-results/cases.json").read_text())["case_count"]
    return {"method": "CPU simulation of Docker COPY/.dockerignore payload, not a Docker build",
            "exit_code": completed.returncode,
            "listed_cases": len(completed.stdout.splitlines()) if completed.returncode == 0 else 0,
            "matrix_exit_code": matrix_code, "generated_cases": generated_cases,
            "missing_fixture": "FileNotFoundError" in completed.stderr and "fixture.json" in completed.stderr}


def docker_context_includes(relative_path, ignore_text):
    """Evaluate this repo's simple allowlist, including parent-directory matches.

    Deliberately reject unfamiliar glob syntax instead of claiming to emulate
    every Docker ignore feature. Docker also applies patterns to parent paths,
    so an exception for a directory can include its descendants.
    """
    parts = PurePosixPath(relative_path).parts
    included = True
    for line in ignore_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exception = line.startswith("!")
        pattern = line[1:] if exception else line
        pattern = pattern.strip("/")
        if pattern == "**":
            matched = True
        else:
            if any(char in pattern for char in "?[]\\") or "**" in pattern or ".." in pattern.split("/"):
                raise ValueError("Review Docker context audit for changed .dockerignore glob syntax")
            expected = pattern.split("/")
            matched = len(parts) >= len(expected) and all(
                fnmatch.fnmatchcase(part, glob) for part, glob in zip(parts, expected)
            )
        if matched:
            included = exception
    return included


def probe_readonly_cache():
    from qwen_worker import runtime
    # Both CPU fixtures and all write attempts stay in this temporary directory.
    # The real directory-creation failure must occur inside the fallback guard.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        root.relative_to(Path(tempfile.gettempdir()).resolve())
        readonly, fallback = root / "readonly" / "hub", root / "writable" / "hub"
        readonly.relative_to(root)
        fallback.relative_to(root)
        real_mkdir = Path.mkdir
        def mkdir(path, *args, **kwargs):
            if path == readonly:
                raise PermissionError(30, "read-only test cache")
            return real_mkdir(path, *args, **kwargs)
        download = Mock(return_value=str(root / "snapshot"))
        with patch.dict("os.environ", {"HF_HOME": str(readonly.parent), "HF_HUB_CACHE": str(readonly),
                                       "HUGGINGFACE_HUB_CACHE": str(readonly)}), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.object(runtime, "FALLBACK_HUB_CACHE", fallback), \
                patch.object(Path, "mkdir", autospec=True, side_effect=mkdir), \
                patch.object(runtime, "_validate_snapshot"), \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            runtime.download_model(Settings())
        return {"fallback_reached": download.call_args.kwargs["cache_dir"] == str(fallback),
                "download_attempts": download.call_count, "revision": download.call_args.kwargs["revision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/review/fail_closed_audit.json")
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--tokenizer-library", type=Path, help="Optional local tokenizers install directory")
    args = parser.parse_args()
    if bool(args.upstream_source) != bool(args.tokenizer_json):
        parser.error("Supply both --upstream-source and --tokenizer-json")
    if args.tokenizer_library:
        sys.path.insert(0, str(args.tokenizer_library.resolve()))
    logging.disable(logging.CRITICAL)
    raw = png(Image.new("RGB", (1, 1)))
    compressed_text = b"comment\0\0" + zlib.compress(b"x" * (2 * 1024**2))
    chunk = (struct.pack(">I", len(compressed_text)) + b"zTXt" + compressed_text
             + struct.pack(">I", zlib.crc32(b"zTXt" + compressed_text) & 0xffffffff))
    metadata_image = base64.b64encode(raw[:33] + chunk + raw[33:]).decode()
    probes = {
        "oversized_png_metadata": {"base64_characters": len(metadata_image),
                                   **invoke({"prompt": "test", "images": [metadata_image]})},
        "unpaired_unicode_surrogate": invoke(json.loads('{"prompt":"\\ud800"}')),
        "invalid_base64_control": invoke({"prompt": "test", "images": ["%%%"]}),
    }
    # Show that inline serialization can fail after a completed pipeline call.
    # Noise is a deterministic worst-case transport fixture, not model output.
    noise = Image.frombytes("RGBA", (2048, 2048), random.Random(42).randbytes(2048 * 2048 * 4))
    probes["late_png_output_rejection"] = invoke({"prompt": "test", "width": 2048, "height": 2048}, noise)
    probes["risky_four_output_batch"] = invoke({"prompt": "test", "width": 2048, "height": 2048, "num_images": 4}, noise)
    del noise
    reference = base64.b64encode(raw).decode()
    probes["memory_guard_control"] = invoke(
        {"prompt": "test", "images": [reference] * 10, "width": 2048, "height": 2048,
         "output_resolution": 2048, "true_cfg_scale": 2},
        phase_budgets=lambda: {"conditioning": 49 * 2**30, "denoising": 66 * 2**30, "decoding": 66 * 2**30})
    payload_parent = ROOT / "outputs/review"
    payload_parent.mkdir(parents=True, exist_ok=True)
    payload_directory = Path(tempfile.mkdtemp(prefix="container-payload-", dir=payload_parent))
    probes["docker_vram_tool_payload"] = reproduce_container_payload(payload_directory)
    probes["readonly_cache_initialization"] = probe_readonly_cache()
    if args.upstream_source:
        probes["admitted_long_prompt_rope"] = reproduce_rope(args.upstream_source, args.tokenizer_json)
    result = {"kind": "CPU audit; no model execution or paid RunPod jobs", "pillow_version": pillow_version,
              "python_version": sys.version.split()[0], "probes": probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
