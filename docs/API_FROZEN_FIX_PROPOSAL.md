# Draft: fixing avoidable failures with the public API frozen

Implementation status: the compatible fixes and output-routing mitigation are
now implemented; see [changes and verification](FIXES_IMPLEMENTED.md). This
document preserves the proposal and its compatibility rationale, following the
[failure and cost review](RUNPOD_RISK_REVIEW.md). Strict output admission remains
a separate policy option because it rejects some previously successful jobs.

Preserve request fields and defaults, response fields, inline base64 delivery,
seed assignment, requested image count/dimensions, output format/quality and
alpha behavior. Use the existing `InputError` response for correctable input.
Keep model precision, source pins and inference semantics unchanged.

The first, second and fourth fixes move failures earlier or repair internal
behavior. Output admission requires a separate distinction: rejecting requests
using worst-case compressed size can reject requests that previously succeeded.
That preserves the schema but changes which requests are accepted.

1. **Validate the actual DiT layout before conditioning.**

   Add a small internal CPU preflight component, supplied by `runtime.py` to
   `ImageService`. Run it under the existing pipeline lock, before staging the
   text encoder or executing any request-related GPU forward pass. This check
   must run even when `MEMORY_GUARD=false`: positional validity is independent
   of estimated free VRAM.

   Build the same raw prompt template, resized reference grids and image-slot
   masks as the pinned pipeline. Use the loaded processor/tokenizer and its
   actual system-prefix removal count. Include `effective_prompt`'s transparency
   text, the empty-negative-prompt space substitution, and both CFG branches.
   Validate both branches before starting the positive encoder pass.

   Expand image slots according to the pinned layout and include the target
   grid. Check the position indices the DiT will actually use: nonnegative
   positions must be **0 through 8191**, and negative spatial positions must
   be **-1024 through -1**. Count reference-grid position advances as well as
   text positions. Validate emitted positions, rather than rejecting solely
   because an unused final cursor has advanced beyond the range.

   This avoids the weaker checks `characters <= 20000`, `tokens <= 8192`, or
   `index < 9216`: none fully validates the supported layout. Also verify that
   image-slot counts agree with the reference grids so malformed special-token
   layouts fail before encoding.

   Return the existing input-error shape with an actionable message, for example:

   ```json
   {"error": "Prompt and reference layout exceed the model's positional limits. Shorten the prompt or reduce reference processing resolution/count."}
   ```

   Reject the unsupported layout without truncating text or extending RoPE
   tables. An extension needs its own model/correctness validation. Keep CPU
   preflight results local to the job; reuse them where practical and clear
   them on every exit. Prefer reusing the pipeline's preprocessing helpers;
   cover any necessary adapter logic with parity checks against the pinned
   source so validation cannot drift from inference.

   Primary files: `qwen_worker/service.py`, `runtime.py`, `conditioning.py`,
   and a proposed `qwen_worker/preflight.py`. These are internal interfaces.

2. **Classify malformed text and image data as input errors.**

   In `GenerationRequest.parse`, check that positive and negative strings can
   be encoded as UTF-8. Convert `UnicodeEncodeError` to `InputError`. Preserve
   valid Unicode and existing whitespace behavior; do not repair invalid text
   by dropping/replacing code points.

   In `decode_images`, preserve existing explicit `InputError` messages, and
   catch the reproduced Pillow metadata `ValueError` inside the image-decoding
   boundary. Convert it to the existing invalid-image `InputError`. Retain
   metadata, decompression-bomb, pixel-count, frame-count and aspect-ratio
   limits. Add other codec exception types only when their input-triggered
   behavior is established by a reproduction or documentation.

   Keep exception conversion close to these CPU operations. A global
   `except ValueError: InputError(...)` could hide a genuine model failure.
   CUDA OOM and unexpected inference failures retain their existing refresh
   handling; `MemoryError` is not treated as harmless malformed image data.

   Expected result for both reproduced malformed inputs: zero pipeline calls,
   an `error` response, and no `refresh_worker: true`. Then send a valid request
   through the same service instance to verify it remains usable.

   Primary files: `qwen_worker/requests.py`, `images.py`; preserve the handler's
   existing distinction between `InputError` and inference failures.

3. **Reduce late output-size losses without silently changing results.**

   Exact compressed size is unknown until pixels are generated. A finite
   inline cap cannot guarantee delivery for every currently accepted image
   size/count/content combination. Sequential generation reduces wasted work
   after an overflow; it cannot guarantee a successful response.

   Recommended initial patch, preserving accepted request combinations:

   - Add an internal output-budget assessment using count, dimensions, format,
     alpha mode, base64 expansion and JSON/envelope overhead. A verified upper
     bound can identify guaranteed-fit combinations. Exceeding that bound
     means uncertainty, not proof that the actual compressed output will fail.
   - For batches without a guaranteed transport fit, use sequential output
     calls, reusing the existing job-local conditioning cache. Encode and check
     each image before computing the next. This changes internal scheduling;
     preserve seeds/order and check batch-versus-sequential numerical behavior.
   - Keep the exact serialized-byte limit authoritative. Preserve its final
     check and timing/envelope allowance. On overflow, stop remaining calls
     immediately and return the existing error shape without refreshing a
     healthy worker. Do not return a partial image list as success.
   - Consider one bounded encoding retry for lossless formats when near the
     limit: stronger PNG compression or greater lossless WebP compression
     effort. Preserve decoded pixels, alpha, dimensions and MIME type. Treat
     this as a measured CPU-cost tradeoff, not a guaranteed rescue. Encoding
     bytes can change even though the returned image is identical; omit this
     optimization if the freeze includes byte-identical encoded files.

   Derive any guaranteed-fit bound from the actual pinned encoders: raw pixel
   bytes alone omit PNG filter bytes, compression/chunk overhead and metadata.
   JPEG/WebP need their own justified bounds. Never infer a safe compressed
   size from a mean compression ratio, format name or requested quality.

   This approach can avoid computing outputs two through four once output one
   has already overflowed. It cannot recover the GPU cost of the first image
   or guarantee delivery if a later image makes the aggregate exceed the cap.
   It can also reduce throughput versus parallel generation; compare both
   successful-job latency and discarded GPU work before broad rollout.

   A stronger **strict admission policy** would reject before inference unless
   an encoder-specific worst-case bound fits the inline cap. That can eliminate
   size-related late failures for admitted jobs once the bound is established,
   but rejects some compressible jobs that currently succeed. Keep this as an
   explicit acceptance-policy decision, not an automatic part of the first
   patch. URLs/external storage, partial results, fewer/smaller outputs,
   automatic format changes and lower JPEG quality would change the frozen
   delivery contract or requested result.

   Primary files: `qwen_worker/service.py`, `images.py`; use the existing
   configured output cap. No new public request or response fields are needed.

4. **Package benchmark inputs and separate cache discovery from writes.**

   Include the existing documentation/data directory in the image. A simple
   packaging change is:

   ```dockerfile
   COPY docs /app/docs
   RUN python /app/scripts/benchmark_vram.py --list
   ```

   Place these after `COPY scripts /app/scripts`. This includes the fixture,
   tensor specification and generated report directory expected by the tools.
   Listing cases does not load the model or require CUDA. Keep the production
   handler command unchanged. For a smaller payload, copy just the required
   JSON inputs and ensure the report writer's destination directory exists.
   Test matrix generation into a temporary output location separately.

   Split cache path selection/environment setup from directory creation.
   Configure Hub environment paths before importing `huggingface_hub`, but
   do not create directories just to discover an existing model. Validate an
   explicit `MODEL_PATH` and search the existing known cache roots/revision
   before any download writes. Update both `load_service`'s early setup and
   `download_model`; fixing only the latter leaves the early failure in place.

   If a download is necessary, create/use the selected write destination.
   Apply the existing permission-only fallback to both directory creation and
   download operations. On `EACCES`, `EPERM` or `EROFS`, try the existing
   `/cache/huggingface/hub` fallback once, unless already there. Pass an explicit
   `cache_dir` so prior Hub imports cannot redirect that download. Always use
   the configured model and revision and validate the completed snapshot.

   Disk-full, authentication, network and corrupt-snapshot failures should
   retain accurate diagnostics instead of becoming permission fallbacks or
   selecting another revision. If neither destination works, fail startup
   clearly rather than retrying indefinitely. A complete readable cache hit
   should require neither network access nor write permission.

   Primary files: `Dockerfile`, `qwen_worker/config.py`, `runtime.py`.

Recommended patch order: malformed-input handling first; positional preflight
second; Docker/cache fixes third; output-budget mitigation fourth. Keep each
patch independently reviewable. Treat strict output admission as a separate
policy change because of its acceptance tradeoff.

| Fix | Acceptance checks before deployment |
|---|---|
| Positional preflight | Reproduce the 10,008-position prompt; reject before encoder staging/forward. Check positive index 8191/8192 and negative spatial index -1024/-1025, both CFG branches, transparency text, long/thin references and malformed slot layouts. Verify allowed layouts match upstream positions. |
| Malformed inputs | Re-run both existing reproductions; assert no refresh and zero pipeline calls. Follow each with a valid job. Keep normal Unicode, RGB/RGBA and existing invalid-base64 behavior. Inject a genuine inference error to confirm it still requests refresh. |
| Output handling | Use compressible and high-entropy RGB/RGBA fixtures across formats, boundary sizes and four outputs. Verify exact JSON cap, format/quality/pixels/alpha, seed order, stopping before remaining calls after overflow, and no partial success. Measure the sequential throughput tradeoff on GPU. |
| Container/cache | Built image lists exactly 48 cases and generates the matrix without CUDA. Test a complete read-only cache hit, absent read-only cache directory, permission fallback, explicit local snapshot, both destinations unwritable, disk-full without permission retry, and no unrelated revision selection. |
| Contract | Run existing tests and compare accepted-job request/response schemas. Check ordinary prompt embeddings/image behavior on the pinned GPU runtime. Report new early rejection separately from successful-output compatibility. |

Production GPU verification remains necessary for inference parity and the
output-routing throughput tradeoff. The malformed-input, layout-indexing,
packaging and cache-control checks can mostly run on CPU before renting a GPU.
