"""Project package init.

The HF_HUB_DISABLE_SYMLINKS_WARNING env var must be set BEFORE
huggingface_hub is imported (via transformers), because that library
reads the value into a module-level constant at import time. Setting
it here guarantees it's in place when any src.* module pulls
transformers in.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
