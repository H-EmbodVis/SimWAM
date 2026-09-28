"""Fine-tuning mode helpers for GRPO expert selection."""

from __future__ import annotations

import torch.nn as nn


EXPERT_FINETUNE_MODES = {"full", "lora", "ffn"}
ACTION_FINETUNE_MODES = EXPERT_FINETUNE_MODES


def normalize_action_finetune_mode(grpo_cfg: dict) -> str:
    """Resolve action fine-tuning mode while preserving legacy `lora.enabled`.

    `grpo.finetune: null` keeps the previous behavior:
    - `grpo.lora.enabled: false` -> full action-DiT fine-tuning
    - `grpo.lora.enabled: true` -> LoRA-only fine-tuning
    """
    lora = dict(grpo_cfg.get("lora", {}))
    raw = grpo_cfg.get("finetune", None)
    mode = ("lora" if bool(lora.get("enabled", False)) else "full") if raw is None else str(raw).lower()
    if mode not in EXPERT_FINETUNE_MODES:
        raise ValueError(f"grpo.finetune must be one of {sorted(EXPERT_FINETUNE_MODES)}, got {mode!r}")
    if mode != "lora" and bool(lora.get("enabled", False)):
        raise ValueError(
            f"grpo.finetune={mode!r} conflicts with grpo.lora.enabled=true; "
            "set grpo.lora.enabled=false or use grpo.finetune=lora."
        )
    return mode


def is_block_ffn_parameter(name: str) -> bool:
    """Whether an expert parameter belongs to a transformer block FFN."""
    return name.startswith("blocks.") and ".ffn." in name


def is_action_ffn_parameter(name: str) -> bool:
    """Backward-compatible alias for the Action-DiT FFN parameter rule."""
    return is_block_ffn_parameter(name)


def set_expert_finetune_requires_grad(expert: nn.Module, mode: str, expert_name: str = "expert") -> int:
    """Set one expert's trainability for a GRPO fine-tuning mode.

    Returns the number of trainable expert parameters after applying the mode.
    The caller is expected to freeze the full model before invoking this helper.
    """
    mode = str(mode).lower()
    if mode not in EXPERT_FINETUNE_MODES:
        raise ValueError(f"{expert_name} finetune mode must be one of {sorted(EXPERT_FINETUNE_MODES)}, got {mode!r}")

    for name, param in expert.named_parameters():
        if mode == "full":
            trainable = True
        elif mode == "lora":
            trainable = "lora_" in name
        else:
            trainable = is_block_ffn_parameter(name)
        param.requires_grad_(trainable)
    count = sum(p.numel() for p in expert.parameters() if p.requires_grad)
    if count == 0:
        raise ValueError(
            f"{expert_name} finetune mode={mode!r} selected 0 trainable params "
            "(expert naming changed or LoRA was not applied?)"
        )
    return count


def set_action_finetune_requires_grad(model: nn.Module, mode: str) -> int:
    """Set action expert trainability for a GRPO fine-tuning mode."""
    return set_expert_finetune_requires_grad(model.action_expert, mode, "action")


def expert_trainable_param_names(expert: nn.Module) -> list[str]:
    """Return trainable expert parameter names, useful for lightweight tests."""
    return [name for name, param in expert.named_parameters() if param.requires_grad]


def action_trainable_param_names(model: nn.Module) -> list[str]:
    """Return trainable action-expert parameter names, useful for lightweight tests."""
    return expert_trainable_param_names(model.action_expert)
