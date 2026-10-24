# API-frozen failure fixes

Implemented in the working tree on 2026-10-17, following
[the proposal](API_FROZEN_FIX_PROPOSAL.md). The original failure observations
remain in [the historical audit](vram/fail_closed_audit.json); the new
[after-fix observations](vram/fail_closed_after_fixes.json) are separate.

| Issue | Implementation | Verification |
|---|---|---|
| Unsupported positional layout | `RequestPreflight` uses the pinned processor's size/token helpers and loaded pipeline templates before conditioning. Both CFG branches, transparency text, reference-grid advances and target positions are checked. It runs independently of `MEMORY_GUARD`. | The real 10,008-position positive/negative prompts are rejected. Twenty actual upstream CPU masks match the validator. Four boundaries, 200 upstream index comparisons and 1,000 resize comparisons pass. |
| Malformed PNG/Unicode refreshes | Parsing rejects invalid UTF-8 strings with `InputError`. Decoder-specific `ValueError` becomes an input error, preserving existing explicit validation messages. | The reproduced inputs make zero pipeline calls and return no refresh flag. A valid job succeeds on the same service afterward. Model errors and CUDA OOM still request refresh. |
| Discarded output batches | A transport budget permits parallel generation only when its conservative bound fits. Otherwise outputs are generated/encoded sequentially; an actual overflow stops remaining calls immediately. | Four compressible 1024-square outputs still complete, with seed order/wrap and alpha preserved. A first-image overflow in a four-output job now makes one batch-one call. Exact JSON-limit boundary checks pass. |
| Missing container inputs | Docker now copies `docs`, the `.dockerignore` allowlist includes the required inputs/tools, and the build runs `benchmark_vram.py --list`. | A fresh simulation of the Docker COPY/ignore payload lists all 48 cases successfully. Matrix generation also succeeds without CUDA. |
| Read-only cache initialization | Cache selection configures environment paths without creating directories. Existing snapshots are searched first. Directory creation and downloading share the permission-only fallback. | Complete read-only and explicit snapshots need no writes/network. Missing read-only directories fall back once; unwritable destinations terminate, and disk-full/nonpermission failures do not become permission retries. |

Public request/response fields, defaults, seeds, output count, dimensions,
format, quality and alpha behavior are preserved. The handler's error/refresh
schema is unchanged. Model pins, precision, prompt conditioning, denoising and
VAE code remain unchanged; positional validation uses CPU metadata only.
No prompt truncation, automatic quality/format changes, partial success or
external-image URLs were introduced. `notes.md` was left untouched.

## Output-size tradeoff

The transport assessment is a routing policy, not a new rejection policy. An
upper bound above the cap means uncertain fit; compressible batches remain
accepted. Sequential calls reuse existing job-local conditioning/reference
latents. GPU numerical results can differ between physical batch sizes, as
with the existing memory-dependent routing; no bit-identical batch/individual
equivalence is claimed.

For metadata-free RGB(A) outputs from the pinned pipeline, PNG routing includes
four channels, scanline filters, a general zlib/zlib-ng compression bound, PNG
chunk overhead, base64 expansion and JSON/envelope allowance. It deliberately
does not use zlib's tighter default-memory bound: Pillow's PNG encoder uses
memory level 9. Other Pillow/buffer settings and JPEG/WebP have no claimed
bound here and route sequentially for multiple outputs. Exact serialized-byte
checks remain authoritative for every format.

For the fixed 48-case comparison, memory admission alone still yields thirteen
parallel, six sequential and five rejected distinct requests on an idealized
80 GiB device. The output assessment changes all thirteen parallel candidates
to sequential, giving **nineteen sequential and five rejected**. Each distinct
request appears in both forced-route benchmark rows. The regenerated
[VRAM comparison](VRAM_COMPARISON.md) now separates these routing decisions.

A single image can still exceed the inline cap after generation, and aggregate
overflow can still discard earlier outputs in that job. Preventing every such
failure would require tighter acceptance limits or a delivery-contract change.
The approved implementation preserves accepted request combinations. It also
preserves encoded-file behavior by leaving compression settings unchanged;
the proposal's optional stronger-compression retry was not enabled.

## Verification commands

**78 CPU regression tests passed**, both in the minimal local environment and
with the pinned worker packages. No model weights or network are needed by the
regression suite:

```powershell
python -m unittest discover -s tests -v
python scripts/vram_matrix.py
python scripts/benchmark_vram.py --list
```

The pinned packages were additionally installed in the ignored
`.local-dev/verify-env` environment, using PyTorch 2.8.0 CPU instead of CUDA.
`scripts/check_runtime.py` passed and `pip check` reported no broken
requirements. No global Python installation was modified.

The real processor/RoPE checks use only checkpoint processor files, downloaded
under `outputs/processor-cache`; no model weights are needed:

```powershell
.local-dev/verify-env/Scripts/python.exe scripts/check_preflight.py --processor-path outputs/processor-cache/models--Qwen--Qwen-Image-2.1/snapshots/790c92633540aa0cb11d9abf19eb46d861714758/processor --output docs/vram/preflight_verification.json
```

The script verifies pinned source/tokenizer hashes, compares actual upstream
preprocessing/masks and executes the upstream RoPE code on CPU. It uses a tiny
fake encoder solely to expose masks; this is not full model inference or an
image-quality test. [Recorded verification](vram/preflight_verification.json).

The updated failure reproductions can be rerun with:

```powershell
python scripts/audit_fail_closed.py --output outputs/review/fail_closed_after_fixes.json
```

Its optional source/tokenizer arguments also demonstrate that upstream still
cannot handle the reproduced long layout while the new preflight rejects it.

The local Docker daemon is unavailable, so a full Docker build was not run.
The local 6 GiB GTX 1660 Ti cannot run this BF16 worker; full CUDA image/throughput
verification and peak-VRAM measurements remain deployment checks. No RunPod
jobs were submitted or paid GPU inference performed. The 48 measured-VRAM
fields remain missing rather than being replaced with estimates.

Encoder-bound references: [Pillow 12.3.0 ZIP encoder](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/ZipEncode.c),
[PNG writer](https://github.com/python-pillow/Pillow/blob/12.3.0/src/PIL/PngImagePlugin.py),
[zlib general bound](https://github.com/madler/zlib/blob/v1.3.1/deflate.c),
[zlib-ng general bound](https://github.com/zlib-ng/zlib-ng/blob/2.2.5/deflate.c).
