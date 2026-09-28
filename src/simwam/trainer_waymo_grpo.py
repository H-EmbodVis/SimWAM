"""Waymo RFS GRPO adapters: collation, true data epochs, and full-set RFS eval."""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .datasets.waymo.evaluation import evaluate_waymo_actions
from .datasets.waymo.waymo_dataset import WaymoVideoDataset
from .trainer_grpo import SimWAMGRPOTrainer
from .utils.logging_config import get_logger

logger = get_logger(__name__)


def collate_waymo_grpo(samples):
    # RFS looks up references by token. Variable-length references must not enter
    # default_collate; keep them available on dataset[i] for full-set evaluation.
    return default_collate([
        {key: value for key, value in sample.items()
         if key not in {"preference_trajectories", "action_raw"}}
        for sample in samples
    ])


class WaymoGRPOTrainer(SimWAMGRPOTrainer):
    """Reuse the NavSim sampler/PPO/IL-anchor math with native XY RFS rewards."""

    def __init__(self, model, train_dataset, reward, *, cfg):
        if not isinstance(train_dataset, WaymoVideoDataset):
            raise TypeError("Waymo GRPO requires WaymoVideoDataset")
        if train_dataset.evaluation_mode != "action_only" or train_dataset.num_frames != 1:
            raise ValueError("Waymo RFS GRPO uses the current-only val samples")
        if not train_dataset.normalize_action:
            raise ValueError("Reuse the IL checkpoint's full-training XY normalization")
        if getattr(reward, "name", None) != "rfs":
            raise ValueError("Waymo GRPO requires grpo.reward.name=rfs")
        if int(cfg.gradient_accumulation_steps) != 1 or int(cfg.grpo.train.rollout_buffer_batches) != 1:
            raise ValueError("Waymo data-epoch accounting requires gradient_accumulation_steps=1 and rollout_buffer_batches=1")
        if int(cfg.num_epochs) < 1:
            raise ValueError("num_epochs must be positive")
        super().__init__(model, train_dataset, reward, cfg=cfg)
        self.eval_dir = Path(self.output_dir) / "eval"
        logger.info(
            "Waymo RFS GRPO: %d RL training samples, %d data epochs, %d inner updates per rollout, "
            "%d optimizer steps; reward=official RFS. Evaluation uses the same optimization set.",
            len(train_dataset), self.num_epochs, self.num_inner_epochs, self.max_steps,
        )

    def _build_loader(self, dataset, worker_init_fn=None):
        loader = super()._build_loader(dataset, worker_init_fn=worker_init_fn)
        loader.collate_fn = collate_waymo_grpo
        return loader

    def _estimate_total_train_steps(self):
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)
        global_batch = self.batch_size * max(int(self.accelerator.num_processes), 1)
        rollouts_per_epoch = ceil(len(self.train_dataset) / global_batch)
        # The base loop advances global_step on each inner PPO update, but reads
        # a new batch only once per rollout. Count all four updates for 10 passes.
        return max(rollouts_per_epoch * self.num_epochs * self.num_inner_epochs, 1)

    def _log(self, payload):
        payload = dict(payload)
        if "step/reward" in payload:
            payload["step/RFS"] = payload["step/reward"]
        if "step/reward_std" in payload:
            payload["step/RFS_std"] = payload["step/reward_std"]
        if "step/pdm_fail_frac" in payload:
            payload["step/rfs_fail_frac"] = payload.pop("step/pdm_fail_frac")
        super()._log(payload)

    @torch.no_grad()
    def _eval_deterministic(self):
        if not self.eval_enabled:
            return None
        model = self.accelerator.unwrap_model(self.model)
        output = self.eval_dir / f"step_{self.global_step:06d}"
        devices = [self.accelerator.device.index] if self.accelerator.device.type == "cuda" else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                metrics = evaluate_waymo_actions(
                    model=model, dataset=self.train_dataset, accelerator=self.accelerator,
                    output_dir=output, num_inference_steps=int(self.cfg.grpo.sample.num_inference_steps),
                    seed=self.seed,
                )
        finally:
            self._set_train_mode()
        if self.accelerator.is_main_process:
            path = output / "metrics.json"
            report = json.loads(path.read_text())
            report.update(
                evaluation_data_role="rl_training_set",
                il_checkpoint=str(self.cfg.model.checkpoint_path),
            )
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        self.accelerator.wait_for_everyone()
        return {f"eval/{key}": float(value) for key, value in metrics.items()
                if isinstance(value, (int, float))}

    def _periodic(self):
        final_step = self.global_step >= self.max_steps
        due_eval = self.eval_every > 0 and self.global_step % self.eval_every == 0
        if self.eval_enabled and (due_eval or final_step):
            metrics = self._eval_deterministic()
            if metrics is not None and self.accelerator.is_main_process:
                logger.info(
                    "[waymo eval on RL training set] step=%d samples=%d RFS=%.6f ADE=%.6f FDE=%.6f",
                    self.global_step, int(metrics["eval/num_samples"]), metrics["eval/RFS"],
                    metrics["eval/ADE"], metrics["eval/FDE"],
                )
                self._log(metrics)
            self.accelerator.wait_for_everyone()
        # The parent train() always saves final weights and resume state once.
        if not final_step and self.save_every > 0 and self.global_step % self.save_every == 0:
            checkpoint = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info("[ckpt] step=%d weights=%s", self.global_step, checkpoint["weights_path"])
