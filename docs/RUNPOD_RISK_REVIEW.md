# API-frozen code, cost and failure review

Implementation update: see [the fixes and verification](FIXES_IMPLEMENTED.md)
for the current behavior. The findings and original observations below describe
the pre-fix worker. The VRAM table now also includes the final output-aware route.

Reviewed 2026-10-17 at worker commit `de63e15`. No handler, request schema,
runtime placement, inference implementation or response fields were changed.
Changes from this review are analysis tools, tests and reports.

**Main result:** malformed inputs can unnecessarily restart the worker, an
admitted long prompt can fail after conditioning, and a completed inference can
be discarded for oversized inline output. The 48-case memory arithmetic is
reproducible, but total GPU peaks remain unmeasured.

## Findings ordered by avoidable cost

| Priority | Finding | Evidence and cost | API-preserving correction |
|---|---|---|---|
| P1 | Missing positional-layout admission | A 19,999-character prompt passes parsing and the idealized 80 GiB memory guard, but its 10,008 retained positions exceed the pinned DiT RoPE tables. Source-method indexing reproduces an exception at index 9,216. In real execution this method is reached after conditioning/reference preparation; the handler's generic error path requests refresh. | Tokenize and validate positive/negative layouts, including reference-grid position advances, before loading the encoder to CUDA. Reject with existing `InputError`; preserve the prompt rather than truncating it. |
| P2 | PNG metadata errors restart a healthy worker | A **2,860-character** base64 PNG with oversized compressed text raises Pillow `ValueError`, outside `decode_images`' exception list. Calling the actual handler with a mocked inference service returns `refresh_worker: true` with **zero pipeline calls**. | Normalize documented decoder failures, including this bounded-metadata `ValueError`, to `InputError` around decoding. Keep genuine CUDA/model errors distinct. |
| P2 | Invalid Unicode restarts a healthy worker | JSON `{"prompt":"\ud800"}` passes parsing. `text_token_bound` raises `UnicodeEncodeError`; the actual handler requests refresh with **zero pipeline calls**. Negative prompts have the same UTF-8 conversion path. | Validate UTF-8 encodability for both prompts during parsing and use the existing input-error response. |
| P2 | Inline output fails after completed computation | `_append_images` checks the serialized size only after a pipeline call and encoding. A deterministic noisy 2048-square RGBA PNG produces an input error **after one completed mocked call**; no generated image is returned. Parallel mode can already have computed all four outputs. | Keep incremental size enforcement. Add conservative format/dimension/batch transport admission where justified, and calibrate with real encoded outputs. Exact PNG/WebP size cannot be known before generation. Avoid silently changing format/quality or the API. |
| P2 | Shipped image omits VRAM tooling inputs | Docker copies `scripts` but not `docs/vram/fixture.json` or `docs/qwen-image-2.1-spec.json`. Reproducing that file payload makes even `benchmark_vram.py --list` fail with `FileNotFoundError`. This blocks verification with the shipped image. | Package the two inputs with the benchmark, and copy the comparison documentation if the matrix writer is to run in-image. Test `--list` during build without CUDA. |
| P2, conditional | Read-only cache initialization bypasses fallback | `configure_cache()` runs before snapshot lookup/download fallback and unconditionally creates `HF_HOME`. If that directory needs creation under a read-only volume, it raises before the writable fallback is considered. A mocked permission failure reproduces this control flow. An existing readable directory generally does not trigger it. | Discover readable caches without creating them; create the writable download destination only when necessary. Verify this on the real mount layout. |

Sources in this repository: [request validation](../qwen_worker/requests.py),
[decoding](../qwen_worker/images.py), [admission and serialization](../qwen_worker/service.py),
[startup/placement](../qwen_worker/runtime.py), [error handling](../handler.py),
and [container packaging](../Dockerfile).

The pinned DiT has 8,192 nonnegative position rows followed by 1,024 negative
rows. Positive indices from 8,192 therefore select the wrong frequencies even
before they run out of bounds. Image grids advance the shared position cursor,
so a prompt-only character/token limit is insufficient.
[Pinned implementation](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/models/transformers/transformer_qwenimage21.py).

Returning `refresh_worker` is consequential: the pinned RunPod SDK removes it
from user output and sets `stopPod`. These malformed-input bugs request a
worker replacement, potentially adding another cold start, rather than merely
returning an error to the client.
[Pinned RunPod result handling](https://github.com/runpod/runpod-python/blob/v1.12.0/runpod/serverless/modules/rp_job.py),
[worker-refresh documentation](https://docs.runpod.io/serverless/workers/handler-functions#worker-refresh).
No deployed endpoint, restart duration, retry policy or GPU price was inspected;
an actual dollar loss is not established by these CPU reproductions.

## What already fails closed

Here, *cheap rejection* means stopping before a pipeline call and keeping the
worker warm. It is separate from restarting after a CUDA/model error.

- Invalid ordinary base64 becomes `InputError`, with zero pipeline calls and
  no refresh in the handler probe.
- Oversized estimated memory becomes `InputError` before inference. The probe
  rejects ten 2K references with CFG enabled and caching on, with no refresh.
- Parsing rejects unknown fields, invalid sizes, unpaired width/height fields,
  more than ten references, inappropriate JPEG transparency and invalid seeds.
- Snapshot lookup checks the configured revision and shard presence; it does
  not silently load an unrelated cached checkpoint.
- A batch that exceeds estimates falls back to single-output calls when those
  estimates fit. OOM does not cause an in-process retry that retains the failed
  inference's tensors. Conditioning caches clear in a `finally` block.

The memory guard is an uncalibrated heuristic, not a guaranteed maximum. It does
not check the RoPE layout and gives the same estimate with VAE tiling enabled
or disabled. An untiled decode, a different attention backend or fragmentation
can therefore behave differently from the fixed matrix. Disabling the guard
also permits parallel execution without admission; it is inappropriate as a
general production workaround for a rejection.

## All 48 cases and production routing

The full table and byte-level exports are:

- [48-case VRAM table](VRAM_COMPARISON.md)
- [CSV](vram/cases.csv), [JSON](vram/cases.json)
- [Equations, assumptions and GPU measurement procedure](VRAM_MEASUREMENT.md)

The fixed workload is four outputs, CFG 1/2, 1/4/10 references, prefix cache
on/off, forced sequential/parallel execution, and 1024/2048-square outputs and
reference processing. That is `2 * 3 * 2 * 2 * 2 = 48` cases. All use BF16,
40 steps, the saved prompt, and default `text_encoder` offload with VAE tiling
and slicing. This follows the repository's existing explicit fixture.

Persistent prefix K/V uses **512 KiB per token per physical output per branch**.
Parallel four-output execution has four physical copies. VAE decoding has no
DiT prefix-cache allocation; batch slicing reduces VAE workspaces but does not
remove those prefix copies. The pinned pipeline runs CFG branches sequentially;
their caches coexist when enabled.
[Pinned cache and attention implementation](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/models/transformers/transformer_qwenimage21.py),
[pipeline lifecycle](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).

The matrix separates exact prefix K/V payload, known-tensor lower bounds,
worker estimates, and missing measured peaks. A worker estimate is **not an
exact required-VRAM figure or a guaranteed upper bound**. Default placement has
30.212 GiB of parameter payload during conditioning and 13.882 GiB during
denoising after encoder eviction, before temporary allocations.

For a hypothetical wholly available **80 GiB** device, using parameter payloads
and the actual service admission formulas:

| Distinct four-output request | Cache | Parallel projection, GiB | Sequential projection, GiB | Guard decision |
|---|---|---:|---:|---|
| 1024, CFG 1, 10 refs | on | 121.46 | 53.42 | Sequential fallback |
| 1024, CFG 2, 4 refs | on | 103.69 | 41.37 | Sequential fallback |
| 1024, CFG 2, 10 refs | on | 224.16 | 68.33 | Sequential fallback |
| 2048, CFG 1, 4 refs | on | 187.41 | 65.37 | Sequential fallback |
| 2048, CFG 2, 1 ref | on | 115.65 | 41.34 | Sequential fallback |
| 2048, CFG 2, 4 refs | off | 98.88 | 65.37 | Sequential fallback |
| 2048, CFG 1, 10 refs | on | 427.65 | 119.20 | Reject before inference |
| 2048, CFG 1, 10 refs | off | 113.42 | 113.42 | Reject before inference |
| 2048, CFG 2, 4 refs | on | 355.88 | 101.26 | Reject before inference |
| 2048, CFG 2, 10 refs | on | 836.35 | 221.37 | Reject before inference |
| 2048, CFG 2, 10 refs | off | 194.98 | 113.42 | Reject before inference |

The other thirteen distinct requests receive parallel admission under those
assumptions. Thus the 24 distinct requests route as **13 parallel, six
sequential, five rejected**. Each appears twice in the 48-case forced-route
matrix. Production has no request field to force the route; the benchmark does
that internally.

**Eleven forced rows have known-tensor lower bounds above 80 GiB. All eleven
are rejected or rerouted by this guard calculation.** It would be incorrect to
say all eleven reach inference and waste credit with the default guard enabled.
Passing admission for another row still does not demonstrate successful GPU
execution or a successful inline response.

**All 48 fixture rows request four PNG outputs, so none has a guaranteed inline
transport fit from these VRAM calculations.** Four 1024-square RGBA outputs have
16 MiB of raw pixels, or about 21.33 MiB after base64 without compression; four
2048-square outputs have 64 MiB raw / 85.33 MiB base64. PNG compression may bring
real outputs below the 8,000,000-byte cap, but image content determines that.
The audit's single 2048-square noise image already exceeds it. These are raw
payload calculations and a synthetic encoding reproduction, not measured model
output sizes. JPEG and WebP also need size checks; RGBA WebP is lossless here.

This scenario excludes CUDA context/library memory, runtime buffers, other
processes and fragmentation; real free tensor budgets are lower. Use actual
device-reported memory, not a marketed capacity label. The route numbers are
covered by a test calling `ImageService.check_memory` for every row.

## Performance improvements, without changing the public API

| Order | Opportunity | Why it may help | Required validation |
|---|---|---|---|
| 1 | Reject malformed text/images and unsupported positional layouts before GPU placement | Prevents avoidable failures and cold starts; the reproductions identify concrete triggers. | Check existing error schema and verify healthy workers stay warm. |
| 2 | Measure phase peaks and calibrate conditioning estimates | The guard uses expanded DiT reference tokens; the VLM uses four-times-fewer image slots. Its large conditioning allowance can reject cache-off jobs that might fit. | Successful peaks on the exact deployment GPU/runtime, including cold and warm jobs. Never infer fit from a lower bound. |
| 3 | Capture only the required final pre-normalization encoder representation | The current encoder hook removes unused autoregressive cache/logits but upstream still asks for all hidden layers. Retaining only the needed representation could reduce conditioning peak. | Preserve pinned pre-final-RMSNorm semantics and multimodal inputs; compare embeddings and image quality. This needs an internal adapter, not just flipping `output_hidden_states` to false. |
| 4 | Reuse fixed token layouts/RoPE and reference preprocessing within a job | Cached steps still construct prefix-related layouts; four sequential calls also repeat CPU reference preprocessing. | Key by complete prompt/layout/reference configuration; compare masks, positions and outputs. |
| 5 | Reuse immutable DiT prefix K/V across sequential outputs | Current job-local reuse covers prompt representations/reference latents, but each output repeats DiT prefill. | Explicit cache lifecycle and isolation, both CFG branches, physical batch/dtype/device checks, and reduced-precision correctness checks. |
| 6 | Benchmark compiled attention/fused projections or MLP work | Can reduce eager launch/intermediate overhead, especially for long prefixes. | Include compile/cold-start cost, dynamic sizes, memory and quality. Pinned Flex attention warns that uncompiled execution materializes dense scores; do not enable it without compilation. |
| 7 | Measure allocator cleanup overhead before making it conditional | Every job calls `empty_cache()` and `ipc_collect()`. Released cache blocks must be reacquired for subsequent work. | Compare warmed throughput, fragmentation and OOM recovery on the real worker. Preserve exceptional cleanup. |

The existing VAE slicing, prompt/reference-latent reuse, final-position-only LM
logits and disabled encoder autoregressive cache are already present. Adding
those again is not a new optimization. No speedup percentage is claimed without
measurements. The pinned pipeline and DiT source links above establish the
candidate locations; these rows are proposals, not implemented changes.

## Reproduction and limits

[Machine-readable CPU observations](vram/fail_closed_audit.json) were generated by
[the audit script](../scripts/audit_fail_closed.py). Most probes need only
Python and Pillow:

```powershell
python scripts/audit_fail_closed.py
python scripts/vram_matrix.py
python -m unittest discover -s tests -v
```

For the optional real-source/tokenizer check, provide the pinned transformer
source and checkpoint `processor/tokenizer.json`. The script verifies both
SHA-256 hashes before executing the source RoPE method with CPU indexing
substitutes. It uses `tokenizers` only for this optional check. In this review:

```powershell
python scripts/audit_fail_closed.py --output docs/vram/fail_closed_audit.json --upstream-source outputs/review-sources/transformer_qwenimage21.py --tokenizer-json outputs/review-sources/tokenizer.json --tokenizer-library .local-dev/audit-deps
```

All twelve expanded positive/negative tokenizer layouts matched the matrix
fixture. The long prompt retained 10,008 positions. The unchanged upstream
RoPE source method reproduced out-of-range indexing; this is not full model
execution. The local audit used Python 3.10.11, Pillow 12.3.0 and an isolated
`tokenizers` 0.22.1 install; the production Transformers stack was not installed.

**41 CPU tests passed**, including the existing 39 and two new tests for actual
service routing of all 48 rows and protection of impossible forced rows. No
production bug was fixed by this review, and passing these tests does not mean
the failure findings are resolved.

The current computer has a **GTX 1660 Ti with 6,144 MiB**, without the worker's
required native BF16 support or capacity; PyTorch/Transformers are not installed.
It cannot run this matrix's full BF16 pipeline. Earlier documents' references
to an 8 GiB Linux cgroup describe a previous research environment, not this
verification. **Zero of the 48 full GPU peaks is measured here.** No RunPod jobs
were submitted and no RunPod inference credit was spent by this review.

The current Docker image needs the packaging correction above before using its
bundled benchmark. On a properly prepared CUDA host, first run one selected case
using the procedure in [VRAM_MEASUREMENT.md](VRAM_MEASUREMENT.md). The benchmark
deliberately bypasses the memory guard and raises the inline cap to isolate GPU
work. It skips cases whose lower bounds exceed device capacity unless explicitly
overridden. Its default full sweep can request 192 jobs / 768 outputs and starts
a fresh model process per case; do not confuse it with a cheap arithmetic run.

The client intentionally does not cancel a remote job when its own wait timeout
expires. Transport/network failure after submission can therefore leave a paid
job running. Persist the returned job ID immediately and check its status before
submitting a replacement; an explicit cancel-on-timeout CLI policy could be
added without changing the worker API. A timeout does not establish that remote
computation stopped. [Existing client behavior](../scripts/client.py).

Finally, the untracked `notes.md` was left untouched. Its claims that parallel
prefix K/V is physically shared, that VAE slicing is absent, that the latent
width is 16 channels, and that WebP/JPEG guarantees bounded response size do not
match this worker. The configured latent channel count is 64, slicing is enabled,
and alpha WebP is saved losslessly. Its VAE-versus-DiT-cache distinction remains
useful, but those memory/transport conclusions should not guide deployment.
