#!/usr/bin/env python3
"""Evaluate a Waymo-trained SimWAM checkpoint on native 5s/4Hz XY val labels."""

from __future__ import annotations

from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys

# Use this checkout even when a different SimWAM directory is installed editable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, broadcast_object_list, set_seed
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from simwam.datasets.waymo.evaluation import evaluate_waymo_actions
from simwam.datasets.waymo.checkpoint import load_waymo_checkpoint
from simwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_waymo")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt"):
        raise ValueError("Pass ckpt=/path/to/Waymo/checkpoints/weights/step_XXXXXX.pt")
    checkpoint_path = Path(str(cfg.ckpt)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Waymo checkpoint not found: {checkpoint_path}")
    eval_cfg = cfg.EVALUATION
    device_kind = str(eval_cfg.device)
    if device_kind not in {"cuda", "cpu"}:
        raise ValueError("EVALUATION.device must be cuda or cpu; select GPUs with CUDA_VISIBLE_DEVICES")
    if device_kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    precision = str(cfg.mixed_precision).lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    # Each rank owns a complete inference model. No optimizer/ZeRO wrapper is needed.
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    accelerator = Accelerator(
        cpu=device_kind == "cpu", mixed_precision=precision,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    set_seed(int(eval_cfg.seed))
    # Resolve timestamp-based directories once, so every rank writes the same run.
    shared_output = [str(Path(str(eval_cfg.output_dir)).expanduser().resolve()) if accelerator.is_main_process else None]
    broadcast_object_list(shared_output, from_process=0)
    output_dir = Path(shared_output[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.EVALUATION.output_dir = str(output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {accelerator.process_index}] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / f"eval_rank_{accelerator.process_index:03d}.log")],
        force=True,
    )
    if int(cfg.model.action_dit_config.action_dim) != 2 or str(cfg.model.mot_attention_mask_mode) != "isolated":
        raise ValueError("Waymo action-only evaluation requires 2D actions and isolated MoT attention")
    if not bool(cfg.model.skip_dit_load_from_pretrain):
        raise ValueError("Evaluation must use model.skip_dit_load_from_pretrain=true and a full Waymo checkpoint")
    dataset = instantiate(cfg.data.val)
    logger.info("Loaded %d val samples; prediction grid +0.25..+5s, 20 XY points", len(dataset))
    model = instantiate(cfg.model, model_dtype=dtype, device=str(accelerator.device))
    checkpoint = load_waymo_checkpoint(model, checkpoint_path)
    model.eval()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
        (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()
    metrics = evaluate_waymo_actions(
        model=model, dataset=dataset, accelerator=accelerator, output_dir=output_dir,
        num_inference_steps=int(eval_cfg.num_inference_steps), seed=int(eval_cfg.seed),
        max_samples=eval_cfg.max_samples,
    )
    if accelerator.is_main_process:
        logger.info("samples=%d ADE=%.6f m FDE=%.6f m RFS=%.6f", metrics["num_samples"], metrics["ADE"], metrics["FDE"], metrics["RFS"])
        logger.info("Results: %s", output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
