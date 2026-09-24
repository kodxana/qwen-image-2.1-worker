"""Populate the same pinned cache used by the worker, without requiring a GPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_worker.config import Settings
from qwen_worker.runtime import download_model

if __name__ == "__main__":
    print(download_model(Settings.from_env()))
