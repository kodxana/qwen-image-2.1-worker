"""GPU initialization performed once, before the Runpod worker starts accepting jobs."""

import logging
import math
import hashlib
import errno
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from qwen_worker.config import BASE_MODEL_ID, DEFAULT_STEPS, MODEL_ID, SUPPORTED_MODELS, Settings, configure_cache
from qwen_worker.service import ImageService
from qwen_worker.preflight import RequestPreflight

logger = logging.getLogger("qwen_worker.runtime")
FALLBACK_HUB_CACHE = Path("/cache/huggingface/hub")


def source_fingerprint() -> str:
    """Identify deployed source without a Git checkout or reading model weights."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    try:
        paths = [root / "handler.py", root / "requirements.txt", *sorted((root / "qwen_worker").glob("*.py"))]
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    except OSError:
        return "unavailable"
    return digest.hexdigest()[:16]


def _format_bytes(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid model metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata must contain an object: {path}")
    return value


def _validate_snapshot(path: Path) -> None:
    """Check component metadata and referenced shards without scanning weight contents.

    Hub files may be symlinks into the cache's blob directory. ``is_file`` follows
    those links, so incomplete downloads and broken links do not count as hits.
    The loader still validates tensor contents and component compatibility.
    """
    index = _read_json(path / "model_index.json")
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError(f"Model snapshot is not a QwenImage21Pipeline: {path}")
    for component in ("processor", "scheduler", "text_encoder", "transformer", "vae"):
        entry = index.get(component)
        if not isinstance(entry, list) or len(entry) != 2 or not all(entry):
            raise ValueError(f"Model snapshot has no {component} component: {path}")
        if not (path / component).is_dir():
            raise ValueError(f"Model snapshot is missing {component}: {path}")

    _read_json(path / "scheduler" / "scheduler_config.json")
    # Transformers supports several processor layouts; let its loader validate
    # the exact layout instead of requiring a version-specific filename here.
    if not any((path / "processor").glob("*.json")):
        raise ValueError(f"Model snapshot has no processor metadata: {path}")
    for component in ("text_encoder", "transformer", "vae"):
        directory = path / component
        _read_json(directory / "config.json")
        manifests = list(directory.glob("*.safetensors.index.json"))
        if manifests:
            for manifest in manifests:
                weight_map = _read_json(manifest).get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError(f"Invalid weight map: {manifest}")
                shards = list(weight_map.values())
                if any(not isinstance(shard, str) or Path(shard).name != shard or "/" in shard or "\\" in shard for shard in shards):
                    raise ValueError(f"Invalid weight shard name in {manifest}")
                for shard in set(shards):
                    if not shard:
                        raise ValueError(f"Invalid weight shard name in {manifest}")
                    if not (directory / shard).is_file() or (directory / shard).stat().st_size == 0:
                        raise ValueError(f"Missing or empty weight shard: {directory / shard}")
        elif not any(p.is_file() and p.stat().st_size for p in directory.glob("*.safetensors")):
            raise ValueError(f"Model snapshot has no safetensors weights: {directory}")


def _hub_cache_path(cache: Path) -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or cache / "hub")


def _cache_roots(cache: Path) -> list[Path]:
    """Only known Hub layouts; never recursively choose an unrelated snapshot."""
    return list(dict.fromkeys([
        _hub_cache_path(cache),
        cache / "hub",
        Path("/runpod-volume/huggingface-cache/hub"),
        Path("/runpod-volume/huggingface/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/opt/huggingface/hub"),
        FALLBACK_HUB_CACHE,
    ]))


def _cached_snapshot(base: Path, settings: Settings) -> Path | None:
    model_cache = base / f"models--{settings.model_id.replace('/', '--')}"
    revision = settings.model_revision
    if revision is None:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        # Named revisions must resolve through the Hub's corresponding ref,
        # never through the first directory that happens to contain weights.
        ref = model_cache / "refs" / revision
        try:
            ref.resolve().relative_to((model_cache / "refs").resolve())
            revision = ref.read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            return None
    return model_cache / "snapshots" / revision


def _log_model_cache(settings: Settings) -> None:
    """List cache directories before validation, without reading weight contents."""
    cache = configure_cache()
    paths = [Path("/runpod-volume"), Path("/runpod-volume/huggingface-cache"),
             Path("/runpod-volume/huggingface")]
    for base in _cache_roots(cache):
        paths.append(base)
        for model_id in SUPPORTED_MODELS:
            model = base / f"models--{model_id.replace('/', '--')}"
            paths.extend((model, model / "snapshots"))
    if settings.model_path:
        paths.append(Path(settings.model_path))
    paths = list(dict.fromkeys(paths))
    logger.info("[CACHE LS] Supported models=%s HF_HOME=%s", ", ".join(SUPPORTED_MODELS), cache)
    logger.info("[CACHE LS] ls -la -- %s", " ".join(str(path) for path in paths))
    try:
        result = subprocess.run(
            ["ls", "-la", "--", *(str(path) for path in paths)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("[CACHE LS] Directory listing unavailable: %s", exc)
        return
    output = (result.stderr + result.stdout).strip()
    logger.info("[CACHE LS] exit=%s:\n%s%s", result.returncode, output[:20000],
                "\n[CACHE LS] Output truncated at 20000 characters" if len(output) > 20000 else "")


class ModelCacheError(RuntimeError):
    """Startup cannot proceed without a complete, already provisioned snapshot."""


def resolve_cached_model(settings: Settings) -> str:
    """Read-only resolution. Production startup must never download weights."""
    if settings.model_path:
        model_path = Path(settings.model_path)
        try:
            _validate_snapshot(model_path)
        except (ValueError, OSError) as exc:
            raise ModelCacheError(f"Invalid cached MODEL_PATH {model_path}: {exc}. Downloads are disabled at worker startup.") from exc
        logger.info("[MODEL RESOLVER] Using validated explicit MODEL_PATH: %s", model_path)
        return str(model_path)

    cache = configure_cache()
    for base in _cache_roots(cache):
        snapshot = _cached_snapshot(base, settings)
        if snapshot is None or not snapshot.is_dir():
            continue
        try:
            _validate_snapshot(snapshot)
        except (ValueError, OSError) as exc:
            logger.warning("[MODEL RESOLVER] Incomplete requested snapshot: %s", exc)
            continue
        logger.info("[MODEL RESOLVER] Exact requested revision cache hit: %s", snapshot)
        return str(snapshot)

    searched = ", ".join(str(base) for base in _cache_roots(cache))
    raise ModelCacheError(
        f"Required cached model is missing or incomplete: {settings.model_id}@{settings.model_revision}. "
        f"Searched: {searched}. Downloads are disabled at worker startup. "
        "Check the model/volume mount and exact revision, or provision it beforehand with scripts/download_model.py."
    )


def provisioning_settings(settings: Settings) -> Settings:
    """Only explicit provisioning may select a model using environment variables."""
    model_id = os.environ.get("MODEL_ID", "").strip() or MODEL_ID
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported test/provisioning MODEL_ID {model_id!r}; expected one of {', '.join(SUPPORTED_MODELS)}")
    revision = os.environ.get("MODEL_REVISION", "").strip() or SUPPORTED_MODELS[model_id]
    return replace(settings, model_id=model_id, model_revision=revision)


def _snapshot_identity(path: Path) -> tuple[str, str | None]:
    """Identify an explicit local snapshot without reading weight contents."""
    index = _read_json(path / "model_index.json")
    model_id = next((name for name in SUPPORTED_MODELS
                     if f"models--{name.replace('/', '--')}" in path.parts), None)
    if model_id is None:
        model_id = MODEL_ID if index.get("sample_sigmas") is not None else BASE_MODEL_ID
    _validate_sampling_sigmas(index.get("sample_sigmas"), model_id)
    revision = path.name if re.fullmatch(r"[0-9a-fA-F]{40}", path.name) else None
    return model_id, revision


def detect_cached_model(settings: Settings) -> Settings:
    """Production discovery is read-only and independent of model-selector env."""
    logger.info("[MODEL DETECT] Production: selecting the complete cached model; MODEL_ID/MODEL_REVISION environment overrides are ignored; downloads DISABLED")
    if settings.model_path:
        path = Path(settings.model_path)
        try:
            _validate_snapshot(path)
            model_id, revision = _snapshot_identity(path)
        except (ValueError, OSError) as exc:
            raise ModelCacheError(f"Invalid cached MODEL_PATH {path}: {exc}. Downloads are disabled at worker startup.") from exc
        selected = replace(settings, model_id=model_id, model_revision=revision)
    else:
        roots = _cache_roots(configure_cache())
        candidates = {}
        problems = []
        for base in roots:
            for model_id in SUPPORTED_MODELS:
                directory = base / f"models--{model_id.replace('/', '--')}" / "snapshots"
                try:
                    snapshots = sorted(directory.iterdir())
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    problems.append(f"{directory}: {exc}")
                    continue
                for snapshot in snapshots:
                    if not re.fullmatch(r"[0-9a-fA-F]{40}", snapshot.name):
                        continue
                    try:
                        _validate_snapshot(snapshot)
                        _validate_sampling_sigmas(_read_json(snapshot / "model_index.json").get("sample_sigmas"), model_id)
                    except (ValueError, OSError) as exc:
                        logger.warning("[MODEL DETECT] Incomplete/invalid cached snapshot %s: %s", snapshot, exc)
                        problems.append(f"{snapshot}: {exc}")
                        continue
                    candidates.setdefault((model_id, snapshot.name), snapshot)
        if len(candidates) > 1:
            found = "; ".join(f"{model}@{revision}: {path}" for (model, revision), path in candidates.items())
            raise ModelCacheError(f"Ambiguous model cache: {found}. Mount only the intended model/revision or set MODEL_PATH to its complete snapshot. Downloads are disabled at worker startup.")
        if not candidates:
            details = "; ".join(problems[:10]) or "no supported complete snapshots found"
            raise ModelCacheError(
                f"No complete supported cached model found ({', '.join(SUPPORTED_MODELS)}). "
                f"Searched: {', '.join(map(str, roots))}. Details: {details}. "
                "Downloads are disabled at worker startup. Check Runpod's cached Model setting and cache mount."
            )
        (model_id, revision), path = next(iter(candidates.items()))
        selected = replace(settings, model_id=model_id, model_revision=revision, model_path=str(path))
    logger.info("[MODEL DETECT] Selected model=%s revision=%s snapshot=%s", selected.model_id,
                selected.model_revision or "unversioned local snapshot", selected.model_path)
    return selected


def prepare_startup_model(settings: Settings) -> Settings:
    """Select before GPU imports or queue registration, with a clear fatal log."""
    mode = "Hub test provisioning" if os.environ.get("QWEN_HUB_TEST_PROVISION", "false").lower() == "true" else "production cache detection"
    try:
        _log_model_cache(settings)
        prepared = prepare_hub_test_model(settings)
        return prepared if mode == "Hub test provisioning" else detect_cached_model(prepared)
    except Exception:
        logger.critical("[STARTUP FATAL] %s failed; refusing to accept jobs", mode, exc_info=True)
        raise


def download_model(settings: Settings) -> str:
    """Explicit provisioning command only; never called by the serving runtime."""
    try:
        return resolve_cached_model(settings)
    except ModelCacheError:
        if settings.model_path:
            raise
    cache = configure_cache()
    from huggingface_hub import snapshot_download

    logger.info("[MODEL RESOLVER] Resolving requested Hub snapshot: %s (%s)", settings.model_id, settings.model_revision)
    started = time.perf_counter()
    download_args = dict(
        repo_id=settings.model_id, revision=settings.model_revision,
        allow_patterns=[
            "model_index.json", "processor/*", "scheduler/*",
            "text_encoder/*", "transformer/*", "vae/*",
        ],
    )
    download_cache = _hub_cache_path(cache)
    try:
        download_cache.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(cache_dir=str(download_cache), **download_args)
    except OSError as exc:
        # Runpod managed model mounts may be read-only. A missing pinned
        # revision must download to container storage, never select another
        # cached revision. Do not disguise network/disk-full errors as this case.
        fallback = FALLBACK_HUB_CACHE
        if exc.errno not in {errno.EACCES, errno.EPERM, errno.EROFS} or download_cache == fallback:
            raise
        logger.warning("[MODEL RESOLVER] Cache is read-only; downloading exact revision to %s", fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(cache_dir=str(fallback), **download_args)
    _validate_snapshot(Path(downloaded_path))
    logger.info("[MODEL RESOLVER] Snapshot ready in %.2fs: %s", time.perf_counter() - started, downloaded_path)
    return downloaded_path


def prepare_hub_test_model(settings: Settings) -> Settings:
    """Explicit Hub-test provisioning before the unchanged cache-only loader.

    Only .runpod/tests.json enables this exception. Never infer permission from
    a missing cache, a job input, or the SDK's local/test endpoint identifiers.
    """
    enabled = os.environ.get("QWEN_HUB_TEST_PROVISION", "false").lower()
    if enabled not in {"true", "false"}:
        raise ValueError("QWEN_HUB_TEST_PROVISION must be true or false")
    if enabled == "false":
        return settings
    settings = provisioning_settings(settings)
    source = "MODEL_ID environment" if os.environ.get("MODEL_ID", "").strip() else "Turbo default (MODEL_ID unset)"
    logger.info("[HUB TEST MODEL] selection=%s model=%s revision=%s", source, settings.model_id, settings.model_revision)
    logger.warning(
        "[HUB TEST PROVISION] Explicit test-only downloads ENABLED for %s@%s. "
        "Do not enable QWEN_HUB_TEST_PROVISION on production endpoints. "
        "Provisioning may take minutes on the allocated test GPU; jobs are not yet accepted.",
        settings.model_id, settings.model_revision,
    )
    started = time.perf_counter()
    source = download_model(settings)
    _validate_sampling_sigmas(_read_json(Path(source) / "model_index.json").get("sample_sigmas"), settings.model_id)
    logger.info("[HUB TEST PROVISION] Complete in %.2fs; entering cache-only model loading", time.perf_counter() - started)
    return replace(settings, model_path=source)


def _validate_architecture(pipe) -> None:
    """The memory estimator is specific to this architecture and BF16 loading."""
    config = pipe.transformer.config
    expected = {"num_layers": 32, "num_attention_heads": 32, "attention_head_dim": 128,
                "patch_size": 1, "in_channels": 64}
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError(f"Unsupported transformer {name}; expected {value} for the memory guard")
    if getattr(pipe.vae.config, "z_dim", None) != 64 or getattr(pipe.vae, "spatial_compression_ratio", None) != 16:
        raise ValueError("Unsupported VAE architecture; expected 64 latent channels and 16x spatial compression")


def _validate_sampling_sigmas(sigmas, model_id: str) -> int | None:
    if sigmas is None:
        if model_id == MODEL_ID:
            raise ValueError("Turbo checkpoint is missing sample_sigmas; check the cached checkpoint and Diffusers pin")
        return None
    if (not isinstance(sigmas, (list, tuple)) or not sigmas
            or any(type(sigma) not in {int, float} or not math.isfinite(sigma)
                   or not 0 < sigma <= 1 for sigma in sigmas)
            or any(a <= b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError("Invalid checkpoint sample_sigmas")
    return len(sigmas)


def _configured_sampling_steps(pipe, settings: Settings) -> int | None:
    steps = _validate_sampling_sigmas(getattr(pipe.config, "sample_sigmas", None), settings.model_id)
    if steps is not None:
        logger.info("[SAMPLING] Checkpoint defines a %d-step sigma schedule; reporting native pipeline steps without rewriting request inputs", steps)
    else:
        logger.info("[SAMPLING] No checkpoint sigma schedule; using requested num_inference_steps")
    return steps


def _install_conditioning_encoder_hook(encoder):
    """Limit unused LM outputs while retaining upstream hidden-state semantics.

    The pipeline owns its existing pre-final-RMSNorm hook. This pre-hook changes
    only language logits/cache flags, never hidden states or multimodal inputs.
    """
    if "logits_to_keep" not in inspect.signature(encoder.forward).parameters:
        raise RuntimeError("The conditioning encoder must support logits_to_keep")

    def conditioning_forward(_module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["logits_to_keep"] = 1
        kwargs["use_cache"] = False
        return args, kwargs

    return encoder.register_forward_pre_hook(conditioning_forward, with_kwargs=True)


def _module_bytes(module, device_type: str | None = None) -> int:
    seen = set()
    total = 0
    for tensor in (*module.parameters(), *module.buffers()):
        if id(tensor) in seen or (device_type is not None and tensor.device.type != device_type):
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def _phase_tensor_budgets(pipe, torch, offload: str, component_bytes: dict[str, int]) -> dict[str, int]:
    """Reserve future weight placements before estimating request tensors.

    Standard model offload can keep the VAE resident while denoising and the
    encoder resident during reference encoding. Sequential offload uses the same
    conservative whole-component reservation; its true submodule peak needs GPU
    measurement. Allocator cache is reusable but live tensors are not.
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    reusable = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    available = free_bytes + reusable
    if offload == "text_encoder":
        phases = {"conditioning": ("text_encoder",), "denoising": (), "decoding": ()}
    elif offload in {"model", "sequential"}:
        phases = {"conditioning": ("text_encoder", "vae"),
                  "denoising": ("transformer", "vae"), "decoding": ("vae",)}
    else:
        phases = {"conditioning": (), "denoising": (), "decoding": ()}
    resident = {name: _module_bytes(getattr(pipe, name), "cuda") for name in component_bytes}
    return {
        phase: max(0, available - sum(max(0, component_bytes[name] - resident[name]) for name in components))
        for phase, components in phases.items()
    }


def load_service(settings: Settings) -> ImageService:
    t_resolve = time.perf_counter()
    logger.info("[STARTUP] Checking required cached model; automatic downloads DISABLED")
    try:
        source = resolve_cached_model(settings)
    except ModelCacheError:
        logger.critical("[STARTUP FATAL] Cached model unavailable; refusing to accept jobs", exc_info=True)
        raise
    cache_seconds = time.perf_counter() - t_resolve
    logger.info("[STARTUP] Cached snapshot validated in %.3fs: %s", cache_seconds, source)
    # Belt and suspenders: component loaders are local-only and Hub clients are
    # offline even if an upstream loader attempts an incidental metadata fetch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    t_import = time.perf_counter()
    import torch
    from diffusers import QwenImage21Pipeline
    from qwen_worker.conditioning import make_pipeline_class
    import_seconds = time.perf_counter() - t_import
    logger.info("[STARTUP] Torch/Diffusers/adapter imports=%.3fs", import_seconds)
    t_device = time.perf_counter()

    # =========================================================================
    # SYSTEM DIAGNOSTICS BANNER
    # =========================================================================
    logger.info("=" * 80)
    logger.info("     QWEN-IMAGE-2.1 RUNPOD SERVERLESS WORKER - INITIALIZATION BANNER")
    logger.info("=" * 80)
    logger.info(" [SYSTEM & PLATFORM]")
    logger.info("   * Platform: %s (%s %s)", platform.system(), platform.release(), platform.machine())
    logger.info("   * Python: %s | Executable: %s", sys.version.split()[0], sys.executable)
    logger.info("   * Working Directory: %s", os.getcwd())
    logger.info("   * PyTorch: %s | CUDA Runtime: %s | cuDNN: %s",
                torch.__version__, torch.version.cuda or "N/A", torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A")

    if not torch.cuda.is_available():
        logger.error("<!> FATAL: CUDA is not available. This worker requires an NVIDIA GPU.")
        raise RuntimeError("A CUDA GPU is required to run Qwen-Image-2.1 inference")
    if not torch.cuda.is_bf16_supported():
        logger.error("<!> FATAL: BF16 is not supported on this device. Ampere (RTX 30xx/Axx) or newer required.")
        raise RuntimeError("This worker requires a GPU with BF16 support (Ampere or newer)")

    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_cap = torch.cuda.get_device_capability(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    logger.info(" [GPU COMPUTE & VRAM]")
    logger.info("   * CUDA Devices: %d (Active: 0 - %s, Compute %d.%d)", device_count, gpu_name, gpu_cap[0], gpu_cap[1])
    logger.info("   * Initial VRAM: %.2f GiB Free / %.2f GiB Total", free_mem / (1024**3), total_mem / (1024**3))
    logger.info("   * Native BF16 Support: %s", torch.cuda.is_bf16_supported())

    # Host Memory and CPU
    cpu_cores = os.cpu_count() or 4
    num_threads = min(8, max(2, cpu_cores // 2))
    torch.set_num_threads(num_threads)

    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_info = f"{vm.available / (1024**3):.1f} GiB Available / {vm.total / (1024**3):.1f} GiB Total"
    except ImportError:
        ram_info = "psutil not available"

    logger.info(" [CPU & HOST SYSTEM RAM]")
    logger.info("   * CPU Logical Cores: %d | PyTorch CPU Threads: %d", cpu_cores, num_threads)
    logger.info("   * Host System RAM: %s", ram_info)

    # Storage & Volumes
    logger.info(" [STORAGE & VOLUMES]")
    for path_str in ["/runpod-volume", "/opt/huggingface", "/tmp", str(Path.home() / ".cache")]:
        p = Path(path_str)
        if p.exists():
            try:
                du = shutil.disk_usage(p)
                logger.info("   * %s: directory available, mount point=%s (Free: %.1f GB, Total: %.1f GB)", path_str, p.is_mount(), du.free / (1000**3), du.total / (1000**3))
            except OSError:
                logger.info("   * %s: Exists", path_str)
        else:
            logger.info("   * %s: [Not Found]", path_str)

    # Worker Settings
    logger.info(" [WORKER SETTINGS]")
    logger.info("   * MODEL_ID: %s", settings.model_id)
    logger.info("   * MODEL_REVISION: %s", settings.model_revision)
    logger.info("   * MODEL_PATH: %s", settings.model_path or "(Exact requested Hub revision)")
    logger.info("   * CPU_OFFLOAD: %s", settings.offload)
    logger.info("   * VAE_TILING: %s", settings.vae_tiling)
    logger.info("   * MEMORY_GUARD: %s", settings.memory_guard)
    logger.info("=" * 80)

    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    logger.info("<<< Step 1/3: Required cached snapshot already validated: %s (offline)", source)

    device_seconds = time.perf_counter() - t_device
    logger.info(">>> Step 2/3: Loading Diffusers QwenImage21Pipeline (torch_dtype=bfloat16)...")
    t_pipe = time.perf_counter()
    pipeline_class = make_pipeline_class(QwenImage21Pipeline)
    pipe = pipeline_class.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
    )
    _validate_architecture(pipe)
    sampling_steps = _configured_sampling_steps(pipe, settings)
    preflight = RequestPreflight(pipe)
    _install_conditioning_encoder_hook(pipe.text_encoder)
    component_bytes = {name: _module_bytes(getattr(pipe, name)) for name in ("text_encoder", "transformer", "vae")}
    for name, size in component_bytes.items():
        logger.info("   * Loaded %s parameters and buffers: %s", name, _format_bytes(size))
    logger.info("<<< Step 2/3 Completed: Pipeline components instantiated in %.2fs", time.perf_counter() - t_pipe)
    load_seconds = time.perf_counter() - t_pipe

    logger.info(">>> Step 3/3: Applying hardware placement & memory offload strategy...")
    t_placement = time.perf_counter()
    if settings.vae_tiling:
        pipe.vae.enable_tiling()
        logger.info("   * VAE tiling enabled for large-resolution decoding memory optimization.")

    pipe.vae.enable_slicing()
    logger.info("   * VAE batch slicing enabled; samples decode one at a time.")

    # Device placement based on offload mode
    if settings.offload == "model":
        pipe.enable_model_cpu_offload()
        logger.info("   * Enabled standard Diffusers model CPU offload.")
    elif settings.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        logger.info("   * Enabled sequential CPU offload.")
    elif settings.offload == "text_encoder":
        # Keep the transformer and VAE resident; move the encoder only on a
        # conditioning cache miss and evict it before reference VAE encoding.
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to("cpu")
        logger.info("   * Transformer and VAE resident in CUDA; text encoder staged for conditioning.")

        def _load_text_encoder():
            if next(pipe.text_encoder.parameters()).device.type != "cuda":
                pipe.text_encoder.to("cuda")

        def _offload_text_encoder():
            if next(pipe.text_encoder.parameters()).device.type == "cuda":
                pipe.text_encoder.to("cpu")
                logger.debug("   * Text encoder offloaded before reference VAE encoding.")

        pipe.before_prompt_encode = _load_text_encoder
        pipe.before_prepare_latents = _offload_text_encoder
    else:
        pipe.to("cuda")
        logger.info("   * Placed full pipeline (all components) resident on CUDA GPU.")

    alloc_mem = torch.cuda.memory_allocated() / (1024**3)
    reserved_mem = torch.cuda.memory_reserved() / (1024**3)
    free_mem_after, _ = torch.cuda.mem_get_info(0)
    free_gib_after = free_mem_after / (1024**3)

    logger.info("=" * 80)
    logger.info(">>> QWEN-IMAGE-2.1 SERVICE READY TO SERVE RUNPOD SERVERLESS JOBS <<<")
    logger.info("    Baseline GPU VRAM: %.2f GiB Allocated | %.2f GiB Reserved | %.2f GiB Free", alloc_mem, reserved_mem, free_gib_after)
    logger.info("[STARTUP SUMMARY] cache=%.3fs, imports=%.3fs, device/diagnostics=%.3fs, "
                "model_load=%.3fs, placement=%.3fs, service_total=%.3fs; source=%s",
                cache_seconds, import_seconds, device_seconds, load_seconds,
                time.perf_counter() - t_placement, time.perf_counter() - t_resolve, source_fingerprint())
    logger.info("=" * 80)

    pipe.set_progress_bar_config(disable=True)

    def infer(**kwargs):
        started = time.perf_counter()
        last_logged = started
        total_steps = sampling_steps or kwargs.get("num_inference_steps", DEFAULT_STEPS)

        def log_step(_pipe, step, _timestep, callback_kwargs):
            nonlocal last_logged
            now = time.perf_counter()
            done = step + 1
            if done == 1 or done == total_steps or now - last_logged >= 5:
                elapsed = now - started
                logger.info("[DENOISING] step %d/%d (%.0f%%), pipeline elapsed=%.2fs, estimated remaining=%.2fs",
                            done, total_steps, 100 * done / total_steps,
                            elapsed, elapsed / done * max(0, total_steps - done))
                last_logged = now
            if done == total_steps:
                logger.info("[DECODING] Denoising complete; decoding %d generated image(s)",
                            kwargs.get("num_images_per_prompt", 1))
            return callback_kwargs

        # Empty tensor inputs keep logging off the tensor/CPU transfer path.
        kwargs["callback_on_step_end"] = log_step
        kwargs["callback_on_step_end_tensor_inputs"] = []
        try:
            with torch.inference_mode():
                return pipe(**kwargs)
        except Exception:
            # Failed calls bypass the pipeline's normal hook/cache cleanup.
            for cleanup in (pipe.vae.clear_cache, pipe.maybe_free_model_hooks):
                try:
                    cleanup()
                except Exception:
                    logger.exception("Pipeline cleanup failed during error handling")
            raise
        finally:
            try:
                logger.info("Job CUDA peaks so far: allocated %.2f GiB, reserved %.2f GiB",
                            torch.cuda.max_memory_allocated() / (1024**3),
                            torch.cuda.max_memory_reserved() / (1024**3))
            except Exception:
                logger.warning("Could not read CUDA memory peak counters", exc_info=True)
            if settings.offload == "text_encoder":
                if next(pipe.text_encoder.parameters()).device.type == "cuda":
                    pipe.text_encoder.to("cpu")

    def generator(seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    @contextmanager
    def job_context():
        stats = {}
        reset_ok = False
        # Reset once per job, so sequential calls retain the highest peak.
        try:
            torch.cuda.reset_peak_memory_stats()
            reset_ok = True
        except Exception:
            logger.warning("Could not reset CUDA memory peak counters", exc_info=True)
        try:
            with pipe.conditioning_context():
                yield stats
        finally:
            try:
                if reset_ok:
                    stats["allocated"] = torch.cuda.max_memory_allocated()
                    stats["reserved"] = torch.cuda.max_memory_reserved()
            except Exception:
                logger.warning("Could not read CUDA memory peak counters", exc_info=True)

    def get_memory_budget() -> int:
        if not torch.cuda.is_available() or not settings.memory_guard:
            return 1024**4  # Effectively unlimited if guard is disabled
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes + max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())

    def get_phase_budgets() -> dict[str, int]:
        return _phase_tensor_budgets(pipe, torch, settings.offload, component_bytes)

    def release_memory() -> None:
        if torch.cuda.is_available():
            if settings.offload == "text_encoder":
                if next(pipe.text_encoder.parameters()).device.type == "cuda":
                    pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return ImageService(
        infer,
        generator,
        settings,
        memory_budget=get_memory_budget if settings.memory_guard else None,
        phase_budgets=get_phase_budgets if settings.memory_guard else None,
        release_memory=release_memory,
        job_context=job_context,
        preflight=preflight,
        sampling_steps=sampling_steps,
    )


