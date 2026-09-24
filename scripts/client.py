"""Submit a job, wait for completion, and save images without printing base64 or API keys."""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_ROOT = "https://api.runpod.ai/v2"


def api_call(url: str, key: str, body: dict | None = None) -> dict:
    payload = json.dumps(body).encode() if body is not None else None
    request = Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        # Avoid logging headers, the auth key, or a server body that could echo input images.
        raise RuntimeError(f"Runpod returned HTTP {exc.code}") from None


def save_images(result: dict, destination: Path) -> dict:
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    output = result.get("output", result)
    if not isinstance(output, dict) or not output.get("images"):
        raise RuntimeError("The response contains no generated images")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in output.items() if key != "images"}
    metadata["images"] = []
    extensions = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp"}
    for index, image in enumerate(output["images"]):
        extension = extensions.get(image.get("mime_type"))
        if extension is None:
            raise RuntimeError("Unexpected image MIME type in worker response")
        path = destination / f"image-{index}.{extension}"
        path.write_bytes(base64.b64decode(image["image_base64"], validate=True))
        metadata["images"].append(
            {
                **{key: value for key, value in image.items() if key != "image_base64"},
                "path": str(path),
            }
        )
    (destination / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint_id", nargs="?")
    parser.add_argument("--input", type=Path, default=Path("examples/generate.json"))
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Local editing reference; repeat for multiple images",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--decode", type=Path, help="Decode an existing /status or worker result JSON"
    )
    args = parser.parse_args()
    if args.decode:
        result = json.loads(args.decode.read_text(encoding="utf-8-sig"))
        print(json.dumps(save_images(result, args.output_dir), indent=2))
        return
    if not args.endpoint_id or not re.fullmatch(r"[A-Za-z0-9_-]+", args.endpoint_id):
        parser.error("Supply a valid endpoint id")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        parser.error("Set RUNPOD_API_KEY in the environment")
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), dict):
        parser.error("The input file must contain an input object")
    if args.image:
        if len(args.image) > 10:
            parser.error("At most 10 reference images are supported")
        sizes = [path.stat().st_size for path in args.image]
        if sum(4 * ((size + 2) // 3) for size in sizes) > 8_000_000:
            parser.error("Reference images exceed the worker's base64 input limit")
        payload["input"]["images"] = [
            base64.b64encode(path.read_bytes()).decode("ascii") for path in args.image
        ]
    if len(json.dumps(payload).encode()) > 9_000_000:
        parser.error("Request exceeds 9000000 bytes")
    endpoint = f"{API_ROOT}/{args.endpoint_id}"
    result = api_call(f"{endpoint}/run", key, payload)
    job_id = result.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise RuntimeError("Runpod did not return a valid job id")
    print(f"Submitted job {job_id}", flush=True)
    deadline = time.monotonic() + args.timeout
    while result.get("status") not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Client wait expired; job {job_id} may still be running. "
                f"Check {endpoint}/status/{job_id}; this client did not cancel it."
            )
        time.sleep(min(5, max(0, deadline - time.monotonic())))
        result = api_call(f"{endpoint}/status/{job_id}", key)
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"Job {job_id}: {result.get('status')}: {result.get('error', '')}")
    print(json.dumps(save_images(result, args.output_dir / job_id), indent=2))


if __name__ == "__main__":
    main()
