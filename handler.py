"""Runpod queue entrypoint; load a single pipeline before accepting jobs."""

import time

_PROCESS_STARTED = time.perf_counter()

import argparse
import base64
import json
import logging
import os
import sys
import threading
from pathlib import Path

# Runtime resolves the required local snapshot before importing GPU libraries.
torch = None


class FlushStreamHandler(logging.StreamHandler):
    """Guarantees immediate, unbuffered stdout flush for RunPod serverless log streaming."""
    def emit(self, record):
        super().emit(record)
        self.flush()


# Configure root logger to output unbuffered to stdout immediately
_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler = FlushStreamHandler(sys.stdout)
_handler.setFormatter(_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler],
    force=True,
)
logger = logging.getLogger("qwen_worker.handler")
logger.info("[STARTUP] Python process started; measuring container-side initialization")
_sdk_started = time.perf_counter()
try:
    import runpod
except ImportError:
    runpod = None
logger.info("[STARTUP] RunPod SDK import=%.3fs; process elapsed=%.3fs",
            time.perf_counter() - _sdk_started, time.perf_counter() - _PROCESS_STARTED)

from qwen_worker.config import Settings
from qwen_worker.requests import InputError
from qwen_worker.runtime import load_service, prepare_startup_model, source_fingerprint
logger.info("[STARTUP] Worker source fingerprint=%s", source_fingerprint())

_service = None
_service_lock = threading.Lock()


def get_service():
    global _service, torch
    if _service is None:
        with _service_lock:
            if _service is None:
                logger.info("[INIT] Initializing Qwen-Image-2.1 service runtime...")
                t0 = time.perf_counter()
                settings = prepare_startup_model(Settings.from_env())
                _service = load_service(settings)
                torch = sys.modules.get("torch")
                logger.info("[INIT] Service initialization completed in %.2fs. Listening for RunPod serverless jobs.", time.perf_counter() - t0)
                logger.info("[STARTUP] Process-to-service-ready=%.3fs. RunPod queue/allocation time before process launch is not included.",
                            time.perf_counter() - _PROCESS_STARTED)
    return _service


def handler(job):
    job_id = job.get("id", "unknown_job") if isinstance(job, dict) else "raw_input"
    logger.info("=" * 80)
    logger.info(">>> [RUNPOD JOB RECEIVED] ID: %s", job_id)
    logger.info("-" * 80)
    t0 = time.perf_counter()

    try:
        service = get_service()
        result = service.handle(job)
        elapsed = time.perf_counter() - t0
        logger.info("-" * 80)
        logger.info("<<< [RUNPOD JOB COMPLETED] ID: %s in %.2fs", job_id, elapsed)
        logger.info("=" * 80)
        return result
    except InputError as exc:
        elapsed = time.perf_counter() - t0
        logger.warning("-" * 80)
        logger.warning("<!> [INPUT VALIDATION REJECTED] Job %s rejected in %.2fs: %s", job_id, elapsed, exc)
        logger.warning("=" * 80)
        return {"error": str(exc)}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("-" * 80)
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**3)
                res = torch.cuda.memory_reserved() / (1024**3)
                logger.error("<X> [CUDA OOM FATAL] Job %s failed after %.2fs. VRAM Allocated: %.2f GiB, Reserved: %.2f GiB", job_id, elapsed, alloc, res)
                try:
                    logger.error("    VRAM Summary:\n%s", torch.cuda.memory_summary(abbreviated=True))
                except Exception:
                    pass
            else:
                logger.error("<X> [CUDA OOM FATAL] Job %s failed after %.2fs", job_id, elapsed)
            logger.error("=" * 80)
            return {
                "error": "CUDA out of memory. Set CPU_OFFLOAD=text_encoder (or model), reduce dimensions or reference images.",
                "refresh_worker": True,
            }
        logger.exception("<X> [INFERENCE ERROR] Job %s failed after %.2fs: %s", job_id, elapsed, exc)
        logger.error("=" * 80)
        return {"error": f"Inference error: {str(exc)}", "refresh_worker": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, help="Run one input JSON on a CUDA GPU and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args, sdk_args = parser.parse_known_args()

    if args.local:
        if sdk_args:
            parser.error(f"Unrecognized local arguments: {' '.join(sdk_args)}")
        result = handler(json.loads(args.local.read_text(encoding="utf-8")))
        if "error" in result:
            raise SystemExit(result["error"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(result["images"]):
            extension = image["mime_type"].split("/")[1]
            destination = args.output_dir / f"image-{index}-{image['seed']}.{extension}"
            destination.write_bytes(base64.b64decode(image.pop("image_base64")))
            image["path"] = str(destination)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        # Pre-warm service before registering with RunPod serverless queue
        get_service()
        runpod.serverless.start({"handler": handler})

