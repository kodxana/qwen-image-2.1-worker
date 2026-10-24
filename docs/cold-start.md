# Cold-start investigation

RunPod `delayTime` includes waiting for a worker and cold startup. It is not the
same measurement as Python initialization. [RunPod's optimization guide](https://docs.runpod.io/serverless/development/optimization)
describes the distinction.

## The reported 50-second interval

The user identified the two system-log lines enclosing the delay:

```text
loading container image from cache
[approximately 50 seconds]
Loaded image: registry.runpod.net/dodobird-debian-qwen-image-2-1-worker-main-dockerfile:99c28f5d8
```

This locates that interval in RunPod's container-image preparation, before the
worker Python process starts. It is not the Hugging Face model cache lookup or
pipeline initialization. The exact cause within image loading remains unproven:
these lines do not separate cache restoration, layer extraction, storage
contention, or other host-side work. Do not label it a model download or a Python
restart without additional evidence.

The deployed tag identifies `99c28f5`; the newer local worker commits are not in
that image. Compared with its parent `54ce8e2`, that revision keeps the same base
image declaration and dependency pins, and adds about 600 KiB of docs and small
diagnostic scripts plus a build-only benchmark listing. No models are baked
into either image. This source comparison does not establish that their actual
base/dependency layer digests or built image sizes are identical: the base tag
is mutable and the registry manifests were not available for comparison.

For this specific delay, prioritize preserving/reusing prepared workers through
FlashBoot, and compare image digests, compressed/unpacked layer sizes, GPU host,
region, and timestamps between a fast and slow start. Repeated slow cache loads
with the same digest are useful evidence for a RunPod support ticket. Do not
bake another 30 GiB of model weights into the image to address image loading.
Shrinking the runtime image may help image preparation, but needs an actual
layer-size comparison and GPU compatibility test before changing its base.

## Other evidence from this checkout and the supplied worker log

- On October 18, the supplied log reported cache resolution **0.00 s**, pipeline
  component loading **1.04 s**, and service initialization **3.41 s**.
- Those worker logs do not contain the image-loading interval identified above.
  Their short initialization duration does not explain the image-cache delay.
- GitHub main remained `99c28f5` when checked, matching the supplied deployed tag.
  The later cache-only/response-fitting worker changes were local at that time.
- The current Dockerfile installs dependencies and performs import checks during
  the image build, not at worker startup. Added documentation is approximately
  600 KiB. There is no startup compilation, warm-up inference, or paid keepalive.
- Snapshot validation reads metadata and checks shards, without scanning weight
  contents. A cache miss now fails rather than downloading weights in serving mode.

## New measurements

The first startup log precedes the SDK import. Logs report SDK import time,
source fingerprint, cache validation, Torch/Diffusers imports, device/diagnostic
setup, model loading, placement, and process-to-service-ready time. Compare these
with the **same job's** RunPod delay and worker logs. A long delay with short
process startup points outside Python initialization; a slow import/load/placement
stage identifies the code/storage stage to investigate. These logs are diagnostics,
not evidence that startup has become faster.

## Changes to consider after identifying the slow stage

1. Verify FlashBoot is enabled and the exact configured model revision is cached.
   Managed cached models schedule workers onto hosts with preloaded files.
2. Check worker availability and allowed GPU/data-center/CUDA combinations.
   A network volume restricts scheduling to its data center and adds network I/O.
   Broadening choices requires checking each GPU's memory and compatibility.
3. For interactive sessions, a modest idle timeout can reuse a running worker
   between requests. Waiting time is billed. Keeping an active worker avoids
   baseline cold starts but is billed continuously. No such settings were changed.
4. If imports/loading dominate, benchmark shard loading and CPU/RAM allocation
   on the actual GPU host. Do not blindly load the entire model onto a 24 GiB GPU:
   this checkpoint has about 30.2 GiB of BF16 parameters. Avoid unmeasured parallel
   loading that could exhaust host RAM. Keep the existing offload strategy.

See [endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
and [worker lifecycle](https://docs.runpod.io/serverless/workers/overview) for
FlashBoot, scheduling constraints, and billing behavior. An earlier management API check in this session returned HTTP 403; current
endpoint settings have not been verified for this investigation. No paid benchmark jobs were submitted.
