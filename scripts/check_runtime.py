"""Check real imports and pipeline call compatibility without downloading model weights."""

import inspect
import json
from importlib.metadata import version

import torch
import torchvision
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

required = {
    "prompt",
    "image",
    "negative_prompt",
    "true_cfg_scale",
    "height",
    "width",
    "num_inference_steps",
    "generator",
    "output_resolution",
    "use_kv_cache",
}
missing = required - set(inspect.signature(QwenImage21Pipeline.__call__).parameters)
if missing:
    raise RuntimeError(f"Unsupported pipeline API; missing parameters: {sorted(missing)}")
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "packages": {
                name: version(name)
                for name in [
                    "diffusers",
                    "transformers",
                    "runpod",
                    "accelerate",
                    "huggingface-hub",
                    "safetensors",
                ]
            },
            "pipeline": QwenImage21Pipeline.__name__,
            "encoder": Qwen3VLForConditionalGeneration.__name__,
            "processor": Qwen3VLProcessor.__name__,
        },
        indent=2,
    )
)
