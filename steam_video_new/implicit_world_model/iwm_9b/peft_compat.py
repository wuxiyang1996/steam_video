"""Narrow compatibility guards for optional PEFT quantization backends."""

from __future__ import annotations

from typing import Any


def disable_incompatible_unused_torchao_dispatch(model: Any) -> bool:
    """Disable PEFT's optional torchao dispatcher only for plain weights.

    PEFT 0.19 probes torchao before its default ``torch.nn.Linear`` dispatcher.
    A stale optional torchao installation raises instead of returning false.
    It is safe to bypass only when none of the loaded model parameters are
    torchao tensor subclasses. Quantized torchao models still fail closed.
    """

    from peft.tuners.lora import torchao as peft_torchao

    try:
        peft_torchao.is_torchao_available()
        return False
    except ImportError:
        parameter_modules = {
            type(parameter).__module__ for parameter in model.parameters()
        }
        if any(module == "torchao" or module.startswith("torchao.") for module in parameter_modules):
            raise RuntimeError(
                "incompatible torchao is installed and the model uses torchao weights"
            )
        peft_torchao.is_torchao_available = lambda: False
        return True
