"""Configure actual DeepSpeed clipping, including optimizer-state restoration."""

from __future__ import annotations

import math


def configure_gradient_clipping(accelerator, max_grad_norm):
    """Run before prepare(): Accelerate's DeepSpeed clip call does not clip."""
    requested = float(max_grad_norm)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("V2 max_grad_norm must be finite and positive")
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is not None:
        # Insert the key explicitly: Accelerate only processes existing keys.
        plugin.deepspeed_config["gradient_clipping"] = requested


def verify_gradient_clipping(accelerator, engine, max_grad_norm, *, resumed=False):
    """Verify the initialized engine and its ZeRO optimizer before training.

    ZeRO restores clip_grad from optimizer checkpoints. On resume the requested
    run configuration takes precedence, including checkpoints saved before the
    clipping fix. Fresh-engine mismatches fail instead of being hidden.
    """
    requested = float(max_grad_norm)
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    report = {"requested_max_grad_norm": requested, "rank": accelerator.process_index}
    if plugin is None:
        return {**report, "backend": "torch", "clipping": "accelerator.clip_grad_norm_"}
    actual = float(engine.gradient_clipping())
    if not math.isfinite(actual) or not math.isclose(actual, requested, rel_tol=1e-12):
        raise RuntimeError(f"DeepSpeed gradient_clipping={actual}, expected {requested}")
    optimizer = engine.optimizer
    stage = int(engine.zero_optimization_stage())
    previous = getattr(optimizer, "clip_grad", None)
    if stage > 0 and previous is None:
        raise RuntimeError("Cannot verify ZeRO optimizer gradient clipping")
    if previous is not None:
        previous = float(previous)
        if resumed:
            optimizer.clip_grad = requested
        effective = float(optimizer.clip_grad)
        if not math.isfinite(effective) or not math.isclose(effective, requested, rel_tol=1e-12):
            raise RuntimeError(f"ZeRO optimizer clip_grad={effective}, expected {requested}")
    else:
        effective = actual
    return {**report, "backend": "deepspeed", "zero_stage": stage,
            "engine_gradient_clipping": actual, "optimizer_clip_grad": effective,
            "optimizer_clip_grad_before_resume_override": previous,
            "resume_configuration_applied": bool(resumed)}
