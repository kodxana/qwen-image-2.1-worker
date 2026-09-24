"""Runpod queue entrypoint; load a single pipeline before accepting jobs."""

import argparse
import base64
import json
import logging
from pathlib import Path

from qwen_worker.config import Settings
from qwen_worker.requests import InputError

logger = logging.getLogger(__name__)


def make_handler(service):
    def handler(job):
        try:
            return service.handle(job)
        except InputError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            # Recoverable validation errors never reset the loaded model. CUDA failures do.
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                logger.exception("CUDA memory exhausted")
                return {
                    "error": "CUDA out of memory. Reduce dimensions/reference images or enable "
                    "CPU_OFFLOAD=model; use a larger GPU if necessary.",
                    "refresh_worker": True,
                }
            logger.exception("Inference failed")
            return {"error": "Inference failed; inspect worker logs", "refresh_worker": True}

    return handler


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, help="Run one input JSON on a CUDA GPU and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args, sdk_args = parser.parse_known_args()
    if args.local and sdk_args:
        parser.error(f"Unrecognized local arguments: {' '.join(sdk_args)}")
    from qwen_worker.runtime import load_service

    service = load_service(Settings.from_env())
    handler = make_handler(service)
    if args.local:
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
        import runpod

        # The service lock also protects against accidental future SDK concurrency changes.
        runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
