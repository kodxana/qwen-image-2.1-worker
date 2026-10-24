# Shared Base / Turbo worker

The `main` branch and one container image support both Qwen-Image-2.1 models.
The former `trubo` branch contains only the earlier Turbo changes and is merged
into main; a separate branch is no longer needed. Both endpoints can keep their
existing keys and cached Model selections. Set both GitHub build sources to main.
No WebUI change is required.

## Production selection

At startup the worker identifies a complete snapshot under either supported
Hugging Face repository directory:

- `models--Qwen--Qwen-Image-2.1/snapshots/<revision>`
- `models--Qwen--Qwen-Image-2.1-Turbo/snapshots/<revision>`

It validates component metadata, nonempty weight files, all indexed shards, and
Turbo's saved sampling configuration before GPU imports or queue registration.
The revision comes from the cached snapshot, allowing the selected Runpod Model
revision without another hardcoded worker revision change. Production ignores
`MODEL_ID` and `MODEL_REVISION`; those variables select explicit test/download
provisioning only. `CPU_OFFLOAD`, tiling, and memory guard variables still work.

No complete supported snapshot is fatal. Multiple complete models or revisions
are fatal rather than selecting an arbitrary one; `MODEL_PATH` can disambiguate
a manually managed cache. An explicit local snapshot identifies its variant from
its Hub repository path or saved sampling metadata. An unversioned local path is
reported with a null revision. Production never downloads missing weights.

`[CACHE LS]`, `[MODEL DETECT]` and `[STARTUP FATAL]` logs distinguish missing files,
broken shards, ambiguous caches, and the selected model/revision. Detecting a
model cannot repair a missing Runpod cache mount.

## Sampling and shared runtime

Both use `QwenImage21Pipeline`, the same architecture-specific memory formulas,
CPU offload, references, RGBA output, four-image request limit, VRAM-only generation
routing, and transport fitting. Diffusers is pinned to
`da1d3829cf08d4f329b526d89e17cc035c049d8d`, which supports the checkpoint's
`sample_sigmas`. The Docker build verifies that constructor and the common API.

The worker accepts the existing 1–100 step range with default 40 and passes
`num_inference_steps` through unchanged. It has no Turbo-specific input cap or
request rewriting. The WebUI owns its Turbo eight-step control.

The official Turbo checkpoint itself saves eight sampling sigmas. Upstream
Diffusers uses that saved schedule regardless of `num_inference_steps` when no
custom `sigmas` are supplied. This is checkpoint behavior. The worker validates
and reports the saved schedule length without imposing a hardcoded length;
progress logs and response parameters report actual steps. Base uses the
requested count. The worker does not invent an alternate Turbo sampling grid.

## Build and test selection

Docker build logs state that model weights are unavailable during build and print
supported models, default test model, and library compatibility results.
`.runpod/tests.json` enables only explicit test provisioning and uses shared
8-step generation, transparency, and edit inputs. Before a test Pod starts:

- Leave `MODEL_ID`/`MODEL_REVISION` unset for the faster Turbo default:
  `Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b`.
- To test Base, set `MODEL_ID=Qwen/Qwen-Image-2.1`; its default test revision is
  `790c92633540aa0cb11d9abf19eb46d861714758`. Override `MODEL_REVISION` if needed.

The variables go in the test configuration's `config.env`, not the production
endpoint name. `[HUB TEST MODEL]` logs the selection source and exact revision.
Missing selectors default to Turbo; invalid selectors fail rather than silently
changing models. A complete selected cache avoids downloads. Leave
`QWEN_HUB_TEST_PROVISION` disabled on production endpoints.

Sources: [Base model](https://huggingface.co/Qwen/Qwen-Image-2.1),
[Turbo model](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
[pinned pipeline](https://github.com/huggingface/diffusers/blob/da1d3829cf08d4f329b526d89e17cc035c049d8d/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py),
[Runpod test environment](https://docs.runpod.io/hub/publishing-guide#test-environment-configuration).
