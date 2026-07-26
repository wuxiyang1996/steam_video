#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
try:
    from .language_model.oryx_qwen import OryxQwenConfig, OryxQwenForCausalLM
except ModuleNotFoundError as exc:
    if exc.name != "flash_attn":
        raise
    OryxQwenConfig = None
    OryxQwenForCausalLM = None
