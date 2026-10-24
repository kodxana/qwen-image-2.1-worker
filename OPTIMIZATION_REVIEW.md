# Optimization and memory-accounting review

The follow-up fixes are now implemented: see
[current changes and verification](docs/FIXES_IMPLEMENTED.md). Earlier
verification at worker commit `de63e15` is in the
[API-frozen failure/cost review](docs/RUNPOD_RISK_REVIEW.md) for additional
reproduced malformed-input refreshes, late output rejection, packaging findings,
and all-48-case production routing checks. The historical environment and test
counts below describe the earlier review, which passed 41 CPU tests. The
implementation verification now passes 78 CPU tests.

Reviewed 2026-10-17 against worker commit
`54ce8e2bedbc239faf4944757112ce81659e2e66`, model revision
`790c92633540aa0cb11d9abf19eb46d861714758`, and the pinned Diffusers source.

**Assessment:** the core parameter-byte and DiT prefix-cache arithmetic matches
the checkpoint. The conditioning reuse and encoder-output reductions are
consistent with the pinned pipeline. One input-validation bug was reproduced.
Peak-memory admission remains an uncalibrated heuristic; this review does not
establish GPU performance or freedom from OOM. The follow-up research below now
provides all **48 requested cases**, with exact cache bytes, source-derived
live-tensor lower bounds, and a reproducible CUDA measurement harness.

## Confirmed finding: validate the actual DiT positional layout

**Priority: P2.** [requests.py](qwen_worker/requests.py) validates positive and
negative prompts by character count, permitting up to 20,000 characters.
[service.py](qwen_worker/service.py) checks estimated memory, but neither checks
the resulting DiT positional indices.

The pinned `QwenImage21Rope` implementation builds lookup tensors containing
8,192 nonnegative positions followed by 1,024 negative positions. Its forward
method directly indexes those tensors using the text/image position layout.
The conditioner's 262,144-position config does not extend these DiT tables.

Reproduction:

```python
request = {
    "prompt": "a " * 9999 + "a",  # 19,999 characters
    "width": 512,
    "height": 512,
}
```

- The actual worker parser accepts this request with default prefix caching.
- The official pinned tokenizer produces approximately 10,008 retained sequence
  positions after the pipeline's system-prefix removal.
- The actual `ImageService.check_memory` accepts it with phase tensor budgets of
  49 GiB for conditioning and 66 GiB for denoising/decoding. Those are plausible
  available tensor budgets on the intended large GPU, rather than an artificially
  unlimited budget.
- The guard estimates 12.828 GiB for conditioning, 15.410 GiB for denoising, and
  15.413 GiB for decoding.
- Executing the pinned RoPE `forward` method with NumPy-backed equivalents for
  its tensor construction/indexing reproduces:
  `IndexError: index 9216 is out of bounds for axis 0 with size 9216`.

This reproduction uses the real tokenizer and source method, not full model
inference. The failure is an indexing issue, independent of weight values or GPU
speed. Positions starting at 8,192 are also problematic before an out-of-bounds
index: those rows hold the negative-position frequencies, not higher positive
positions.

Impact: an otherwise admitted request can fail after expensive conditioning and
trigger the handler's generic inference-error worker refresh. Large reference
layouts also advance the position cursor, so a simple prompt-token limit alone
would not cover every case.

Recommended correction: validate tokenized positive and negative layouts,
including reference-grid position advances, before inference. Reject unsupported
layouts with `InputError`. Alternatively, implement and validate an upstream
compatible extension of the RoPE tables. Do not silently truncate user prompts
or assume the VLM context limit applies to the DiT.

## Correct calculations and optimizations

| Area | What was checked | Result |
|---|---|---|
| Component weight bytes | `_module_bytes` sums `numel() * element_size()` over parameters and buffers, deduplicating repeated tensor objects | Correct for the ordinary independently stored tensors in this checkpoint; measures loaded dtype rather than assuming the on-disk VAE precision |
| Complete model size | All checkpoint tensor headers | 7,115,124,736 DiT + 8,767,123,696 conditioner + 337,740,404 VAE elements; about 30.212 GiB parameter payload if entirely BF16 |
| Prefix K/V coefficient | 32 layers × K/V × 32 heads × 128 head dimensions × 2 bytes | Correct: 524,288 bytes, or 512 KiB, per prefix token per output per branch |
| Reference token count | 16-pixel latent stride and the pinned area-preserving, 32-pixel-rounded resize | Correct; 1,000 sampled resize calculations matched the pinned formula |
| CFG caches | Positive and negative prefix lengths are added | Correct; includes references in both branch caches |
| Parallel output caches | Prefix storage multiplied by physical output batch size | Correct for the pinned implementation; it does not share a single physical prefix across batch items |
| Sequential output routing | Admits a single output first, then checks batch estimates | The control flow supports sequential fallback when only the batch exceeds estimates |
| Weight placement reservations | Future component placements are deducted from available tensor budgets; resident component bytes are not reserved twice | Consistent with the intended conservative placement policy; covered by mocked tests |
| Prompt reuse | Job-local cache of full prompt return tuples, including masks | Correct scope for the service's immutable prompt/reference set and inference lock |
| Reference latent reuse | Job-local cache of normalized posterior-mode reference latents | Consistent with deterministic upstream encoding; does not reuse target noise |
| Encoder output reduction | `use_cache=False`, `logits_to_keep=1`; hidden-state output retained | Correctly removes unused autoregressive K/V and most unused language logits while preserving the upstream conditioning representation |
| Encoder placement, default mode | Encoder staged for conditioning and evicted before reference VAE encoding | Implemented; mocked regression coverage verifies ordering and cache-hit behavior |
| VAE tiling/slicing | Tiling controlled by configuration; slicing always enabled | Applied using upstream APIs; these reduce VAE workspaces, not DiT prefix-cache size |
| Failure cleanup | Job conditioning cache cleared in `finally`; failed pipeline calls attempt VAE/offload-hook cleanup | Present; mocked failure coverage passes |

Example verified cache totals, including 512 retained text positions:

| Request | BF16 persistent prefix K/V |
|---|---:|
| One 1024-square reference, one output, CFG off | 2.25 GiB |
| Same, positive and negative prefixes each 512 text positions | 4.5 GiB |
| Same CFG request, four parallel outputs | 18 GiB |
| Ten 2048-square references, one output, CFG off | 80.25 GiB |

The model parameter payload and cache payload are separate live allocations.
Neither table is a complete peak-VRAM requirement.

## Limitations and optimization opportunities, not additional confirmed bugs

1. **Activation and decode estimates need calibration.**
   [memory.py](qwen_worker/memory.py) uses a fixed activation coefficient,
   3 GiB overhead, and a 64-layer-width conditioning allowance. These are
   explicitly heuristic. The calculation does not take VAE tiling state or the
   actual attention backend as inputs. An untiled decode and a tiled decode
   therefore receive the same estimate. Treat batch admission as provisional
   until representative phase peaks are measured.

2. **Default encoder offload still has a full-weight conditioning peak.**
   In `text_encoder` mode the DiT and VAE stay resident while the conditioner
   moves onto the GPU. Parameter payload during conditioning is therefore about
   30.212 GiB, versus about 13.882 GiB during denoising after encoder eviction.
   The reservation calculation accounts for this. This mode is not a method of
   fitting the full workflow on a 24 GiB GPU; smaller devices require a different
   placement strategy. The code already describes it as a large-GPU policy.

3. **Sequential outputs repeat DiT prefix prefill.**
   [conditioning.py](qwen_worker/conditioning.py) reuses prompt representations
   and VAE reference latents, but the upstream pipeline creates fresh DiT caches
   for every output call. For identical conditioning, same-job prefix K/V reuse
   across sequential outputs is a potential speed improvement. It would need an
   explicit upstream-compatible cache interface and correctness checks; the
   current implementation is safer than attempting reuse with an invalid key.

4. **Some memory estimates are deliberately pessimistic.**
   The byte-based prompt bound can exceed actual tokenizer length considerably.
   The activation allowance includes both CFG branches even though the main
   transformer calls run sequentially, and sequential offload reserves whole
   components. These choices can reject requests that would fit. They should be
   tuned from measurements, not replaced with optimistic constants.

5. **Existing tests do not establish real GPU behavior.**
   All 24 tests in [test_runtime.py](tests/test_runtime.py) pass, but model and
   CUDA behavior are mocked. They do not execute real attention, VAE kernels,
   image-quality comparisons, or measured VRAM admission. The Docker import
   check is also not an inference test. This environment does not have PyTorch,
   Transformers, or a tested CUDA runtime installed, so full model execution was
   not attempted.

## Validation performed

- Original runtime checks: **24 passed**. Follow-up matrix/measurement-bookkeeping
  checks are recorded below; these are CPU checks, not GPU measurements.
- Independent comparison of cache coefficient and tensor counts with the saved
  official checkpoint specification.
- Cache examples for reference resolution, CFG, and output batches.
- 1,000 sampled reference-resize comparisons against the pinned formula.
- Actual tokenizer and request/memory-admission reproduction of the long-prompt
  issue; direct execution of the upstream RoPE forward indexing with a NumPy shim.
- Static inspection of the pinned pipeline, DiT, VAE, and conditioner source.

Runtime implementation was not changed by this review. Architecture sources and
the complete tensor inventory are linked from
[QWEN_IMAGE_2_1_ARCHITECTURE.md](QWEN_IMAGE_2_1_ARCHITECTURE.md).


## Follow-up: complete 48-case quantitative comparison

The user selected four outputs per job and matching reference/output resolutions.
The complete matrix varies CFG=1/2, references=1/4/10, prefix caching on/off,
sequential/parallel execution, and 1024/2048-square resolution: **48 cases**.

- [All 48 rows, with separate exact/bounded/estimated/measured columns](docs/VRAM_COMPARISON.md)
- [CSV with integer-byte values](docs/vram/cases.csv)
- [JSON cases and fixed workload](docs/vram/cases.json)
- [Research, equations, runtime controls, and measurement protocol](docs/VRAM_MEASUREMENT.md)
- [Analytical matrix generator](scripts/vram_matrix.py)
- [Isolated real-worker CUDA benchmark](scripts/benchmark_vram.py)

### Exact persistent K/V: all cache-on combinations

All entries below are GiB of **persistent K/V only**, with actual tokenizer counts
for the fixed fixture. Sequential mode keeps one output's cache at a time;
parallel mode holds four. Each case still produces four outputs in total.

| Resolution | Inputs | CFG=1 sequential | CFG=1 parallel | CFG=2 sequential | CFG=2 parallel |
|---|---:|---:|---:|---:|---:|
| 1024 square | 1 | 2.018066 | 8.072266 | 4.025391 | 16.101562 |
| 1024 square | 4 | 8.026855 | 32.107422 | 16.042969 | 64.171875 |
| 1024 square | 10 | 20.044922 | 80.179688 | 40.079102 | 160.316406 |
| 2048 square | 1 | 8.018066 | 32.072266 | 16.025391 | 64.101562 |
| 2048 square | 4 | 32.026855 | 128.107422 | 64.042969 | 256.171875 |
| 2048 square | 10 | 80.044922 | 320.179688 | 160.079102 | 640.316406 |

For **all 24 cache-off cases**, persistent DiT prefix K/V is zero. Total VRAM is
not zero and is not obtained by subtracting K/V from a cache-on measurement:
full-prefix attention/MLP computation repeats at every denoising step.

### More precise conclusions from the pinned eager implementation

The default SwiGLU multiplication has three simultaneous `[B,S,12288]` tensors:
activated gate, up-projection, and product. Together with cache lifetimes and
resident parameters, this produces stronger lower bounds than weights/KV alone.
These are minimum known live-payload figures, **not total peak predictions**.

- One 2K reference, CFG=2, cache on, four parallel outputs: **64.1015625 GiB**
  persistent K/V and a stronger **86.987725474 GiB** eager tensor floor. An 80 GiB
  device cannot complete this exact policy/configuration.
- Ten 2K references, CFG=2, cache on, four parallel outputs: **640.31640625 GiB**
  persistent K/V alone.
- Ten 2K references, cache off, four parallel outputs: zero persistent prefix
  cache but a **63.407311656 GiB** known eager tensor floor. This does not prove
  the case fits an 80 GiB GPU; other activations and runtime allocations remain.
- CFG=2 executes two branches sequentially. Their persistent caches coexist when
  enabled, but two copies of all MLP workspaces do not. The current activation
  formula's branch multiplier is conservative rather than exact liveness.
- The worker's conditioning allowance uses expanded DiT reference tokens rather
  than the VLM's four-times-smaller merged image-slot count. For ten 2K inputs,
  its projected parameter-plus-conditioning figure is about **113.415 GiB**, even
  with caching disabled. This number is an intentionally uncalibrated admission
  projection, not evidence that the actual conditioning peak is 113.415 GiB.

All 48 worker projections exceed the derived known-tensor floors for this
fixture. That consistency check does not establish a guaranteed upper bound.
**Eleven of the 48 rows have derived eager tensor floors above 80 GiB** under
this exact placement policy. Use the actual device-reported capacity rather
than a marketed GPU memory label when applying that comparison to hardware.

### Required measurement correction

The existing runtime resets peaks **per pipeline call**. A benchmark that reads
only the counter after a four-output sequential job can miss the first output's
conditioning peak. The new harness preserves each call's maximum, distinguishes
cold from warmed jobs, and aggregates successful repeated measurements.

Primary comparison runs avoid per-step/per-phase GPU barriers. Optional phase
profiling runs separately so synchronization effects are not mixed into the
primary numbers. Both allocated and reserved peaks are retained; sampled
`nvidia-smi` process usage is separate and explicitly subject to sampling/PID
attribution limits. Different GPU/runtime fingerprints cannot be silently joined.
OOM, analytical impossibility, interruption, and missing results remain distinct.

No NVIDIA device is present in this chat environment, and its host-memory cgroup
limit is 8 GiB. Consequently **none of the 48 total-VRAM peaks has been measured
here**. The tables intentionally keep those cells missing. The CUDA benchmark
must be run on the intended GPU to populate proper measured peak figures.

Validation includes the original 24 CPU tests and 15 additional matrix/peak-
bookkeeping tests (**39 total**). It checks case coverage, unequal CFG prefix
lengths, physical batch multipliers, eager liveness floors, preservation of early
peaks across resets, primary-pass absence of phase barriers, and honest missing
measurement handling. It does not replace real model execution.
All 12 expanded positive/negative tokenizer layouts (two resolutions times three
reference counts times two branches) were additionally checked against the
official tokenizer; their retained non-image counts match the fixture.
