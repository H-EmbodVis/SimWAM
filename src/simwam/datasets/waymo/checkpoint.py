"""Strict loading of complete XY Waymo IL/merged-RL inference weights."""

from pathlib import Path

import torch


def load_waymo_checkpoint(model, checkpoint_path: str | Path) -> dict:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Waymo checkpoint not found: {path}")
    if model.action_expert.action_dim != 2 or model.proprio_dim != 8:
        raise ValueError("Waymo requires action_dim=2 and proprio_dim=8")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict) or not {"mot", "proprio_encoder"}.issubset(payload):
        raise ValueError("Pass a trained Waymo weights/step_*.pt containing mot and proprio_encoder")
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    return {"path": str(path), "step": payload.get("step"), "size_bytes": path.stat().st_size}
