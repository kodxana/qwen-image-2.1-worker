# Reproducible VRAM comparison for all 48 cases

Researched 2026-10-17. The complete numerical comparison is in
[VRAM_COMPARISON.md](VRAM_COMPARISON.md), with integer-byte data in
[cases.csv](vram/cases.csv) and [cases.json](vram/cases.json).

## Workload definition

The user selected **four outputs per job** and **reference processing resolution
matching output resolution**. The Cartesian product is:

```text
CFG             {1, 2}
input images    {1, 4, 10}
prefix KV       {on, off}
execution       {4 sequential outputs, 4 parallel outputs}
output/reference resolution  {1024×1024, 2048×2048}

2 × 3 × 2 × 2 × 2 = 48 cases
```

Reference count and output batch size are independent: ten reference images and
four parallel outputs means the same ten references condition each of four
targets. It does not mean ten outputs.

All cases fix the original base model revision, BF16, the repository's pinned
PyTorch/Diffusers/Transformers runtime, eager default attention processors, 40
steps, seeds 42–45, `text_encoder` offload, VAE tiling/slicing on, and PNG output.
No FP8, quantization, model/sequential offload, compiled Flex attention, or
approximate step-skipping cache is mixed into this comparison.

The explicit fixture is [fixture.json](vram/fixture.json). It includes the actual
prompt, empty negative prompt, tokenizer and source hashes, system-prefix
removal count, and tokenized templates. The pinned pipeline substitutes a space
for an empty negative prompt. Non-image prefix counts are:

| References | Positive text/template tokens | Negative text/template tokens |
|---|---:|---:|
| 1 | 37 | 15 |
| 4 | 55 | 33 |
| 10 | 92 | 70 |

These counts were derived using the checkpoint's official tokenizer, raw pipeline
templates, image-slot exclusion, and the actual system-only template. Runtime
instrumentation checks them again, including cache-off cases, rather than assuming
the static fixture matches an installed pipeline.

The benchmark generates ten deterministic 1024-square RGB geometric references,
uses the first 1/4/10, and records the encoded PNG hashes. The pipeline resizes
them to the chosen reference resolution. Content is held fixed across the
cache/CFG/routing comparison. A production workload with different text lengths,
reference aspect ratios, or image dimensions is a different benchmark.

## What can be determined exactly without GPU execution

Using the pinned source, let:

```text
D = 4096                 hidden width
M = 12288                SwiGLU intermediate width
L = 32                   DiT layers
Hkv = 32, dhead = 128
b = 2                    BF16 bytes per element
B = 1 sequential, 4 parallel
N = resolution² / 256    target latent tokens
R = input_count * N      square reference latent tokens
P+ = R + positive non-image count
P- = R + negative non-image count, for CFG=2 only
C(P) = B * P * L * 2 * Hkv * dhead * b
```

Persistent prefix K/V is exactly `C(P+)` for CFG=1, or `C(P+) + C(P-)` for CFG=2.
With caching off it is zero. This does not include temporary K/V projection
outputs, concatenated K/V, or attention/MLP workspaces. The pinned cache stores
cloned `[B,P,32,128]` K and V tensors in each layer. [1, 2]

The full checkpoint parameter payload, if all parameters are BF16, is
**32,439,977,672 bytes / 30.212083526 GiB**. The resident DiT/VAE parameter
payload in default encoder-offload mode is
**14,905,730,280 bytes / 13.882043101 GiB**. All components coexist on GPU during
conditioning in this policy. The weight-header inventory is in
[qwen-image-2.1-spec.json](qwen-image-2.1-spec.json).

### A stronger eager tensor lower bound

Cache-off still needs large transient tensors. The pinned eager SwiGLU is:

```text
down( SiLU(gate(x)) * up(x) )
```

The activated gate, up-projection, and out-of-place product each have shape
`[B,S,M]`. They coexist during multiplication, giving a known **three-tensor
payload** of `3*B*S*M*b` bytes. Here `S=P+N` during full/prefill execution and
`S=N` in later cached blocks. This is derived from the unfused eager expression;
it must be reconsidered if compilation/fusion changes the implementation. [1]

At the last prefill layer the prefix cache has already been stored when the MLP
runs. During positive prefill only the positive cache is full. During negative
prefill both positive and negative caches are full. Consequently the matrix
computes these candidate live-payload bounds:

```text
positive prefill:  C(P+)              + 3*B*(P+ + N)*M*b
negative prefill:  C(P+) + C(P-)      + 3*B*(P- + N)*M*b   [CFG=2]
```

With caching off, remove `C` but keep the full-prefix MLP payload. For later
cached attention, Q has `N` positions and concatenated K and V each have `P+N`,
so the three attention inputs have payload `B*(3*N+2*P)*D*b`. The matrix compares
this with the later MLP triplet as another candidate, adding the then-live caches.

Add resident parameter payload to the largest candidate, and compare it with
the full-parameter conditioning floor. **Do not add prefill attention tensors
to MLP tensors just because both exist at different points in the same layer.**
The result is a **lower bound**, not a peak prediction: input/residual states,
normalization and RoPE intermediates, encoder activations, VAE workspaces,
buffers, CUDA libraries, and allocator rounding are still missing.

Examples:

- One 2K reference, CFG=2, parallel four outputs: K/V is **64.1015625 GiB**;
  the stronger known-tensor floor is **86.987725474 GiB**. It cannot fit a GPU
  exposing only 80 GiB under this eager/default-offload policy.
- Ten 2K references, CFG=2, parallel four outputs: K/V alone is
  **640.31640625 GiB**, before parameters and transient work.
- Ten 2K references, cache off, parallel four outputs: persistent K/V is zero,
  but the eager known-tensor floor is **63.407311656 GiB**. Zero K/V is not
  zero memory, and this floor does not prove the job fits on an 80 GiB GPU.

Changing CFG from one to two adds a negative pass, but the pinned pipeline runs
the two transformer calls sequentially. Cache-on retains both branch caches;
cache-off does not require two simultaneous copies of all branch workspaces.
The current worker's activation allowance nevertheless includes both branches;
that is conservative accounting, not an exact concurrency description. [2]

## What requires a real GPU

There is no universal exact total-VRAM function of those five request settings.
The actual peak also depends on GPU architecture, PyTorch/CUDA/cuDNN versions,
attention kernels, allocator policy, fragmentation, initial resident tensors,
offloading, compile/fusion choices, prompt length, VAE tiling, and cold/warm state.
Published measurements from SGLang, GGUF, Vulkan, or another offload policy cannot
be substituted for this worker's 48 rows.

Record these separately:

| Metric | Meaning | Limitation |
|---|---|---|
| Job peak allocated | Maximum PyTorch live allocated bytes across every phase and all four outputs | Does not count CUDA allocations outside the PyTorch allocator |
| Job peak reserved | Maximum bytes held by the PyTorch caching allocator | Includes reusable free blocks; do not add it to allocated bytes |
| Sampled process memory | `nvidia-smi` memory attributed to this process, sampled every 100 ms plus query latency | Includes more than tensors, but sampling can miss a short peak; unsupported attribution stays null |
| Model-load peak | Allocator peak while creating/placing the pipeline | Separate from a warmed inference job |
| Optional diagnostic phase peaks | Conditioning, reference encoding/noise, positive/negative prefill, cached/full denoising, VAE decode | Collected in a separate job with synchronization; never substituted for the primary warmed peaks |

PyTorch distinguishes allocated and reserved memory, and `empty_cache` only
releases unused allocator blocks. CUDA execution is asynchronous, so synchronize
before resetting/reading phase measurements. [3, 4] PyTorch snapshots can explain
allocation lifetimes but cannot see every non-PyTorch CUDA allocation. [5]

## Executable measurement method

Use the worker's actual CUDA container/dependencies on a dedicated NVIDIA GPU
with sufficient host RAM, disk, and checkpoint storage. The current verification
computer has a 6 GiB GTX 1660 Ti, which lacks the required native BF16 support and
capacity; it cannot measure this full pipeline. Earlier research used a Linux
environment without a GPU. The scripts do not fabricate GPU measurements here.
The Dockerfile now packages the required fixture/spec files and checks case
listing during build; see the [fixes and verification](FIXES_IMPLEMENTED.md).

```bash
# Regenerate all 48 analytical rows. This does not need CUDA or model weights.
python scripts/vram_matrix.py

# Inspect the case IDs without importing CUDA/model libraries.
python scripts/benchmark_vram.py --list

# On the CUDA host, first measure one case.
python scripts/benchmark_vram.py --device 0 \
  --case 1024-cfg1-ref1-kvon-sequential \
  --output-dir outputs/vram-check --repeats 3 --profile-phases

# Run the full comparison in a NEW output directory.
python scripts/benchmark_vram.py --device 0 \
  --output-dir outputs/vram-full --repeats 3

# Join completed measurements into the CSV/JSON and comparison document.
python scripts/vram_matrix.py --results-dir outputs/vram-full
```

`MODEL_PATH` may select a predownloaded snapshot; the benchmark checks component
parameter counts and BF16 dtype. It also checks the pinned Diffusers source-file
hashes and runtime versions, and verifies observed prefix shapes and cache bytes
against the fixture. A mismatch is an error, not a comparable measurement.
Joining also requires matching GPU/runtime/fixture/instrumentation fingerprints
across successful rows; it refuses to silently mix different configurations.

For each case the harness:

1. Launches a fresh subprocess with exactly one visible GPU. The same process
   handles that case's warmup and repeats, so cold and warmed jobs remain distinct.
2. Skips a mathematically impossible row if its identified-tensor floor already
   exceeds actual reported GPU capacity. This is marked analytical impossibility,
   not measured OOM. `--attempt-impossible` can explicitly request an actual attempt.
3. Loads the real worker using default encoder offload and VAE tiling/slicing.
4. Disables the memory guard and forces the requested sequential/parallel route
   **inside this benchmark only**. Otherwise the guard might reject or silently
   reroute the case being measured. It also relaxes the inline output-size cap so
   transport policy does not stop an otherwise completed GPU workload.
5. Runs one cold job, then three measured repeats by default. The cold job serves
   as the default single warmup. Each job has four outputs with identical seeds
   across cases; conditioning reuse follows the real job lifecycle.
6. Preserves maxima between pipeline calls, synchronizing only at job/call
   boundaries for primary runs. This matters because the existing worker resets
   CUDA peak statistics separately for each pipeline call. Primary runs do not
   insert per-step/per-phase GPU barriers. An optional `--profile-phases` adds a
   separate diagnostic job with synchronized per-phase counter resets; its values
   are not mixed into the primary repeat statistics.
7. Records actual K/V byte counts, token counts, physical batches, model-load
   peaks, absolute allocator peaks, initial allocated/reserved baselines, external
   samples, and diagnostic phase events when requested. It does not retain tensor
   references in the log, and reads shape metadata without a GPU `.item()` barrier.
8. Records hardware/driver metadata, package versions, source hashes, processor
   classes, allocator backend/configuration, precision flags, and image hashes.
9. Exits after OOM/error; the next case starts in a fresh process. Failed-run
   allocator peaks are marked observed-at-failure, never completed-job peaks.
10. Saves JSON incrementally. Joining requires completed successful repeats;
    interrupted or missing runs remain unmeasured. Successful rows report the
    maximum warmed peak, with minimum and median also preserved in CSV/JSON.

With default settings, a fully runnable 48-case matrix requests **192 jobs / 768
generated outputs**, before skipped impossible cases. It can take substantial
GPU time. Enabling phase profiles for every case adds 48 jobs / 192 outputs.
This script prepares that work; it has not run those jobs in this chat.

### Interpreting the measurements

Compare cache on/off **within the same resolution, reference count, CFG, routing,
hardware, and runtime fingerprint**. Compare sequential/parallel for the same
four output seeds and conditions. Report both cold and warmed peaks, and use
separate diagnostic phase data to help explain the maximum. Do not sum unrelated phase peaks or
independent allocator/process peaks, and do not average successful rows with OOMs.

For deployment capacity, use actual device capacity and measured process/system
headroom on an otherwise isolated GPU. A single sampled `nvidia-smi` value is not
an exact process high-water mark, and PID attribution may be unavailable in some
container/MIG configurations. The script records that absence rather than using
another process's memory or substituting zero. [6]

The current worker's predictions remain separate columns. Measure their error
per phase and case before replacing constants. A theoretical lower bound proves
that some cases cannot fit; it cannot prove that remaining cases will fit.

## Research sources

1. [Pinned Qwen-Image-2.1 DiT and SwiGLU implementation](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/models/transformers/transformer_qwenimage21.py).
2. [Pinned Qwen-Image-2.1 pipeline, CFG, and cache lifecycle](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).
3. [PyTorch 2.8 CUDA semantics and memory management](https://docs.pytorch.org/docs/2.8/notes/cuda.html).
4. [PyTorch 2.8 peak allocated memory](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.max_memory_allocated.html) and [peak-stat reset](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.reset_peak_memory_stats.html).
5. [PyTorch 2.8: understanding CUDA memory and non-PyTorch allocations](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html).
6. [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
7. [Official Diffusers Qwen-Image-2.1 pipeline documentation](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21). Current documentation is supporting context; numerical results here use the immutable source pins above.
