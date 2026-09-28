"""Action-only FlowGRPO training.

Shared rollout reuse, relative advantages, frozen-IL mean anchor and optional old-policy KL gate.
"""

from __future__ import annotations

import json
import os
import re
import time
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from .datasets.navsim.pdm_reward import NavSimPDMReward
from .models.wan22.finetune import set_action_finetune_requires_grad
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler

logger = get_logger(__name__)


class SimWAMGRPOTrainer:

    target = "action"

    def __init__(
        self, model, train_dataset, reward: NavSimPDMReward, *, cfg: DictConfig
    ):
        self.model = model
        if str(cfg.grpo.get("target", "action")) != self.target:
            raise ValueError(f"{type(self).__name__} requires grpo.target={self.target}")
        self.action_reward = reward
        self.train_dataset = train_dataset
        self.reward = reward
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.lr_warmup_steps = int(cfg.get("lr_warmup_steps", 100))
        self.resume = cfg.resume
        grpo = cfg.grpo
        ka = self._read_action_knobs()
        self.group_size = ka.group_size
        self.adv_q_lo = ka.adv_q_lo
        self.adv_q_hi = ka.adv_q_hi
        self.adv_eps = ka.adv_eps
        self.ppo_clip_range = ka.ppo_clip_range
        self.adv_clip_max = ka.adv_clip_max
        adv_cfg = grpo.get("train", {})
        self.adv_global_std = bool(adv_cfg.get("adv_global_std", False))
        self.drop_zero_adv = bool(adv_cfg.get("drop_zero_adv", False))
        self._kl_gate_cfg = adv_cfg.get("kl_gate", {}) or {}
        self.kl_gate_enabled = bool(self._kl_gate_cfg.get("enabled", False))
        self.kl_gate_delta = float(self._kl_gate_cfg.get("delta", 0.0001))
        self.kl_gate_asymmetric = bool(self._kl_gate_cfg.get("asymmetric", False))
        if self.kl_gate_enabled:
            if self.kl_gate_delta <= 0.0:
                raise ValueError(
                    f"grpo.train.kl_gate.delta must be > 0, got {self.kl_gate_delta}"
                )
            if not ka.ppo_enabled:
                raise ValueError(
                    "grpo.train.kl_gate.enabled=true needs the importance ratio, which requires ppo_clip_range set or num_inner_epochs>1 (that is what makes `old_logp` and the old transition mean available)."
                )
        self.num_inner_epochs = ka.num_inner_epochs
        self.rollout_buffer_batches = ka.rollout_buffer_batches
        self.ppo_enabled = ka.ppo_enabled
        self.eval_enabled = bool(grpo.get("eval", {}).get("enabled", True))
        self.eval_num_batches = int(grpo.get("eval", {}).get("num_batches", 4))
        vis = grpo.get("vis", {})
        self.vis_enabled = bool(vis.get("enabled", False))
        self.vis_every = int(vis.get("every", 100))
        self.vis_num_conditions = int(vis.get("num_conditions", 2))
        self.vis_max_samples = int(vis.get("max_samples", 0))
        self.vis_plot_mode = str(vis.get("plot_mode", "bev_camera"))
        if self.vis_plot_mode not in {"bev_camera", "bev"}:
            raise ValueError(
                f"grpo.vis.plot_mode must be 'bev_camera' or 'bev', got {self.vis_plot_mode!r}"
            )
        self.vis_dpi = int(vis.get("dpi", 150))
        self.vis_dir = os.path.join(self.output_dir, "vis")
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision: {cfg.mixed_precision}.")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            project_dir=self.output_dir,
            step_scheduler_with_optimizer=False,
        )
        if cfg.get("deepspeed_clip_from_max_grad_norm", False):
            from .utils.deepspeed_clipping import configure_gradient_clipping

            configure_gradient_clipping(self.accelerator, self.max_grad_norm)
        logger.info(
            "GRPO Accelerate: distributed_type=%s world_size=%d process_index=%d mp=%s group_size=%d",
            self.accelerator.distributed_type,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.group_size,
        )
        if (
            self.accelerator.num_processes > 1
            and "DEEPSPEED" not in str(self.accelerator.distributed_type).upper()
        ):
            raise RuntimeError(
                "SimWAMGRPOTrainer requires DeepSpeed (ZeRO-1) for multi-GPU training (plain DDP gradient sync is not wired). Launch via scripts/train_navsim_grpo_zero1_torchrun.sh (sets ACCELERATE_USE_DEEPSPEED=true)."
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._apply_train_mode(self.model)
        active_expert = getattr(self.model, f"{self.target}_expert")
        trainable_params = [p for p in active_expert.parameters() if p.requires_grad]
        num_trainable = sum((p.numel() for p in trainable_params))
        logger.info(
            "GRPO trainable (%s expert) params: %.3f M",
            self.target,
            num_trainable / 1000000.0,
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.train_loader = self._build_loader(
            self.train_dataset, worker_init_fn=worker_init_fn
        )
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = min(int(total_train_steps * 0.05), self.lr_warmup_steps)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        for d in (
            self.output_dir,
            self.checkpoint_root,
            self.weights_dir,
            self.state_dir,
        ):
            ensure_dir(d)
        if self.vis_enabled:
            ensure_dir(self.vis_dir)
        self.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.init_trackers("grpo")
        self._resume_or_load_checkpoint()
        if cfg.get("deepspeed_clip_from_max_grad_norm", False):
            from .utils.deepspeed_clipping import verify_gradient_clipping

            clipping_report = verify_gradient_clipping(
                self.accelerator, self.model, self.max_grad_norm, resumed=bool(self.resume)
            )
            path = Path(self.output_dir) / f"gradient_clipping.rank{self.accelerator.process_index:02d}.json"
            path.write_text(json.dumps(clipping_report, indent=2, allow_nan=False) + "\n")
            logger.info("Verified gradient clipping: %s", clipping_report)
        self._eval_batch = None
        logger.info("Train dataset size: %d", len(self.train_dataset))

    def _read_action_knobs(self):
        """Read action rollout and PPO hyperparameters."""
        from types import SimpleNamespace

        grpo = self.cfg.grpo
        group_size = int(grpo.sample.group_size)
        t = grpo.train
        ppo_clip = t.get("ppo_clip_range", None)
        ppo_clip_range = None if ppo_clip is None else float(ppo_clip)
        adv_clip = t.get("adv_clip_max", None)
        adv_clip_max = None if adv_clip is None else float(adv_clip)
        num_inner_epochs = int(t.get("num_inner_epochs", 1))
        rollout_buffer_batches = int(t.get("rollout_buffer_batches", 1))
        if num_inner_epochs < 1:
            raise ValueError(
                f"action: num_inner_epochs must be >= 1, got {num_inner_epochs}"
            )
        if rollout_buffer_batches < 1:
            raise ValueError(
                f"action: rollout_buffer_batches must be >= 1, got {rollout_buffer_batches}"
            )
        if num_inner_epochs > 1 and ppo_clip_range is None:
            raise ValueError(
                f"action: num_inner_epochs>1 reuses each rollout off-policy and is biased without an importance ratio; set ppo_clip_range (PPO clip) or keep num_inner_epochs=1."
            )
        if ppo_clip_range is not None and 0.0 < ppo_clip_range < 0.002:
            logger.warning(
                "%s: ppo_clip_range=%.1e is BELOW the ~1.4e-3 forward-nondeterminism floor of `ratio`; the clip will trigger on numerical noise and silently zero real gradients. Use >= ~5e-3 for the %d-dim action (do NOT copy flow_grpo's 1e-3 / Flow-Factory's 1e-4 -- their log-prob is meaned over ~1e5 dims).",
                "action",
                ppo_clip_range,
                24,
            )
        return SimpleNamespace(
            group_size=group_size,
            adv_q_lo=float(t.get("adv_clip_lower_quantile", 0.0)),
            adv_q_hi=float(t.get("adv_clip_upper_quantile", 1.0)),
            adv_eps=float(t.get("adv_eps", 1e-08)),
            ppo_clip_range=ppo_clip_range,
            adv_clip_max=adv_clip_max,
            num_inner_epochs=num_inner_epochs,
            rollout_buffer_batches=rollout_buffer_batches,
            ppo_enabled=ppo_clip_range is not None or num_inner_epochs > 1,
        )

    @staticmethod
    def _apply_action_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        set_action_finetune_requires_grad(model, "lora")

    def _apply_train_mode(self, model):
        self._apply_action_only_train_mode(model)

    def _set_train_mode(self):
        self._apply_train_mode(self.accelerator.unwrap_model(self.model))

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(
            ceil(len(self.train_dataset) / global_batch_size), 1
        )
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(
        self, scheduler_type, total_train_steps: int, warmup_steps: int = 0
    ):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), max(total_train_steps - 1, 0))
        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=remaining_steps, eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(
                self.optimizer, factor=1.0, total_iters=remaining_steps
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}.")
        if warmup_steps <= 0:
            return main_scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-06)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-09))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return (f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec)

    def _batch_condition_inputs(self, batch):
        """Extract (input_image, action_horizon, context, context_mask, proprio, tokens)."""
        video = batch["video"]
        if video.ndim != 5:
            raise ValueError(
                f"`batch['video']` must be [B,3,T,H,W], got {tuple(video.shape)}"
            )
        input_image = video[:, :, 0].contiguous()
        action = batch["action"]
        action_horizon = int(action.shape[1])
        context = batch["context"]
        context_mask = batch["context_mask"]
        proprio = None
        if self.accelerator.unwrap_model(self.model).proprio_encoder is not None:
            proprio = batch["proprio"][:, 0, :]
        tokens = list(batch["token"])
        return (input_image, action_horizon, context, context_mask, proprio, tokens)

    @torch.no_grad()
    def _pg_group_mask(self, raw_std_r, rewards_mat, advantages):
        """Exclude tied groups from the local policy mean; keep the IL anchor unmasked."""
        if not self.drop_zero_adv:
            return (torch.ones_like(advantages), {})
        tied = raw_std_r.reshape(-1) <= self.adv_eps
        return ((~tied).unsqueeze(1).expand_as(rewards_mat).reshape(-1).float(), {})

    @torch.no_grad()
    def _compute_advantages(self, rewards_mat):
        """Default scalar-reward normalization; constrained training overrides this hook."""
        mean_r = rewards_mat.mean(dim=1, keepdim=True)
        raw_std_r = rewards_mat.std(dim=1, unbiased=False, keepdim=True)
        if self.adv_global_std:
            all_rewards = self.accelerator.gather(rewards_mat.reshape(-1).detach())
            std_r = all_rewards.std(unbiased=False) + self.adv_eps
        else:
            std_r = raw_std_r + self.adv_eps
        advantages = ((rewards_mat - mean_r) / std_r).reshape(-1)
        adv_diag = {}
        zero_std_frac = float(
            (raw_std_r.reshape(-1) <= self.adv_eps).float().mean().item()
        )
        adv_mask, mask_diag = self._pg_group_mask(raw_std_r, rewards_mat, advantages)
        adv_diag.update(mask_diag)
        if self.adv_q_lo > 0.0 or self.adv_q_hi < 1.0:
            lo = torch.quantile(advantages, self.adv_q_lo)
            hi = torch.quantile(advantages, self.adv_q_hi)
            advantages = advantages.clamp(min=lo, max=hi)
        advantages = advantages.detach()
        return {"advantages": advantages, "adv_mask": adv_mask.detach(),
                "adv_diag": adv_diag, "zero_std_frac": zero_std_frac,
                "nonzero_adv_frac": float((advantages.abs() > 0).float().mean().item())}

    @torch.no_grad()
    def _rollout(self, model, batch):
        input_image, action_horizon, context, context_mask, proprio, tokens = (
            self._batch_condition_inputs(batch)
        )
        cond_b = model.build_action_condition(
            input_image, action_horizon, context, context_mask, proprio
        )
        cond_bg = model.expand_condition(cond_b, self.group_size)
        chain, timesteps, deltas, stoch_idx = model.sample_action_chain(
            cond_bg, action_horizon, deterministic=False
        )
        final_norm = chain[:, -1]
        abs_poses = self.train_dataset.denormalize_action(final_norm)
        tokens_bg = [tok for tok in tokens for _ in range(self.group_size)]
        rewards = self.action_reward.score_batch(
            abs_poses, tokens_bg, device=self.accelerator.device
        )
        pdm_fail_frac = float(getattr(self.action_reward, "last_fail_frac", 0.0))
        true_pdms = float(getattr(self.action_reward, "last_true_pdms", float("nan")))
        reward_submetrics = getattr(self.action_reward, "last_submetrics", None)
        bsz = len(tokens)
        rewards_mat = rewards.view(bsz, self.group_size)
        advantage_info = self._compute_advantages(rewards_mat)
        num_unavailable = int(
            sum((0 if self.action_reward.available(t) else 1 for t in tokens))
        )
        il_abs = None
        steps_ahead = max(self.num_inner_epochs * self.rollout_buffer_batches, 1)
        will_visualize = (
            self.vis_enabled
            and self.accelerator.is_main_process
            and (self.vis_every > 0)
            and any(
                (
                    (self.global_step + 1 + j) % self.vis_every == 0
                    for j in range(steps_ahead)
                )
            )
        )
        if will_visualize:
            det_chain, _, _, _ = model.sample_action_chain(
                cond_b, action_horizon, deterministic=True
            )
            il_abs = self.train_dataset.denormalize_action(det_chain[:, -1])
        old_logp, old_mean, old_std = (None, None, None)
        want_old_terms = self.kl_gate_enabled
        if self.ppo_enabled:
            if want_old_terms:
                old_logp, old_terms = model.action_chain_logprobs(
                    cond_bg,
                    chain,
                    timesteps,
                    deltas,
                    stoch_idx=stoch_idx,
                    return_terms=True,
                )
                old_mean, old_std = (
                    old_terms["mean"].detach(),
                    old_terms["std"].detach(),
                )
            else:
                old_logp = model.action_chain_logprobs(
                    cond_bg, chain, timesteps, deltas, stoch_idx=stoch_idx
                )
            old_logp = old_logp.detach()
        return {
            "cond_b": cond_b,
            "cond_bg": cond_bg,
            "chain": chain,
            "timesteps": timesteps,
            "deltas": deltas,
            "stoch_idx": stoch_idx,
            "num_sde_steps": int(stoch_idx.numel()),
            "pdm_fail_frac": pdm_fail_frac,
            "true_pdms": true_pdms,
            "reward_submetrics": reward_submetrics,
            "rewards": rewards.detach(),
            **advantage_info,
            "num_unavailable": num_unavailable,
            "tokens": tokens,
            "abs_group": abs_poses.reshape(
                bsz, self.group_size, abs_poses.shape[1], abs_poses.shape[2]
            ),
            "il_abs": il_abs,
            "old_logp": old_logp,
            "old_mean": old_mean,
            "old_std": old_std,
        }

    def _effective_kl_beta(self, model):
        return float(model.grpo_kl_beta)

    def _policy_loss(self, model, rollout):
        """PPO clipping, or the optional old-policy KL gate, plus the frozen IL mean anchor.

        Reward ties mask only policy gradients. The IL anchor remains active on every group.
        """
        chain = rollout["chain"]
        timesteps = rollout["timesteps"]
        deltas = rollout["deltas"]
        advantages = rollout["advantages"]
        stoch_idx = rollout.get("stoch_idx")
        num_steps = int(timesteps.shape[0])
        kl_beta = self._effective_kl_beta(model)
        need_terms = kl_beta > 0.0 or self.kl_gate_enabled
        if need_terms:
            new_logp, terms = model.action_chain_logprobs(
                rollout["cond_bg"],
                chain,
                timesteps,
                deltas,
                stoch_idx=stoch_idx,
                return_terms=True,
            )
        else:
            terms = None
            new_logp = model.action_chain_logprobs(
                rollout["cond_bg"], chain, timesteps, deltas, stoch_idx=stoch_idx
            )
        kl_gate_div = None
        if self.kl_gate_enabled:
            old_mean, old_std = (rollout.get("old_mean"), rollout.get("old_std"))
            if old_mean is None:
                raise RuntimeError(
                    "kl_gate is on but the rollout carries no `old_mean`."
                )
            with torch.no_grad():
                d2 = (terms["mean"] - old_mean).pow(2).mean(dim=(2, 3))
                kl_gate_div = d2 / (2.0 * old_std.pow(2))[None, :]
            rollout = dict(rollout, kl_gate_div=kl_gate_div)
        if stoch_idx is None:
            denoise_idx = torch.arange(
                num_steps, device=new_logp.device, dtype=torch.float32
            )
        else:
            denoise_idx = stoch_idx.to(device=new_logp.device, dtype=torch.float32)
        discount = model.grpo_denoising_discount ** (num_steps - 1 - denoise_idx)
        adv_weighted = advantages[:, None] * discount[None, :]
        if self.adv_clip_max is not None:
            adv_weighted = adv_weighted.clamp(-self.adv_clip_max, self.adv_clip_max)
        extra = {}
        mask = rollout.get("adv_mask")
        if mask is None:
            mask = torch.ones(
                new_logp.shape[0], device=new_logp.device, dtype=new_logp.dtype
            )
        mask = mask.to(device=new_logp.device, dtype=new_logp.dtype)
        denom = mask.sum().clamp(min=1.0) * float(new_logp.shape[1])

        def _masked_mean(per_elem):
            return (per_elem * mask[:, None]).sum() / denom

        if self.ppo_clip_range is None and (not self.kl_gate_enabled):
            policy_loss = -_masked_mean(new_logp * adv_weighted)
        else:
            old_logp = rollout["old_logp"]
            ratio = torch.exp(new_logp - old_logp)
            unclipped = -adv_weighted * ratio
            if self.kl_gate_enabled:
                d_gate = rollout.get("kl_gate_div")
                if d_gate is None:
                    raise RuntimeError(
                        "grpo.train.kl_gate.enabled=true needs the rollout-time mean; `_rollout` must store `kl_gate_div` (it requires ppo machinery to be active so `old_logp` and the old mean are captured)."
                    )
                keep = d_gate <= self.kl_gate_delta
                if self.kl_gate_asymmetric:
                    moving_away = adv_weighted * (ratio - 1.0) > 0
                    keep = keep | ~moving_away
                policy_loss = _masked_mean(unclipped * keep.to(unclipped.dtype))
                with torch.no_grad():
                    extra["kl_gate_keep_frac"] = keep.float().mean().detach()
                    extra["kl_gate_div_mean"] = d_gate.mean().detach()
                    extra["kl_gate_div_max"] = d_gate.max().detach()
            else:
                clipped = -adv_weighted * ratio.clamp(
                    1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range
                )
                policy_loss = _masked_mean(torch.max(unclipped, clipped))
            with torch.no_grad():
                extra["ratio_mean"] = ratio.mean().detach()
                if self.ppo_clip_range is not None:
                    extra["clipfrac"] = (
                        (ratio.sub(1.0).abs() > self.ppo_clip_range)
                        .float()
                        .mean()
                        .detach()
                    )
                extra["approx_kl"] = (old_logp - new_logp).mean().detach()
        with torch.no_grad():
            extra["adv_kept_frac"] = mask.mean().detach()
        bc_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
        if model.grpo_use_bc_loss:
            with torch.no_grad():
                ref_chain, ref_ts, ref_deltas, ref_stoch = model.sample_action_chain(
                    rollout["cond_b"],
                    rollout["cond_b"]["action_horizon"],
                    deterministic=False,
                    velocity_use_ref=True,
                )
            # Same restriction as the policy term: the BC anchor is a log-density of the reference
            # policy's samples, which only exists on the stochastic steps.
            bc_logp = model.action_chain_logprobs(
                rollout["cond_b"], ref_chain, ref_ts, ref_deltas, stoch_idx=ref_stoch
            )
            bc_loss = -bc_logp.mean()

        kl_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
        if terms is not None:
            with torch.no_grad():
                _, ref_terms = model.action_chain_logprobs(
                    rollout["cond_bg"],
                    chain,
                    timesteps,
                    deltas,
                    stoch_idx=stoch_idx,
                    use_ref=True,
                    return_terms=True,
                )
            d_mean_raw = (terms["mean"] - ref_terms["mean"]).pow(2)
            d_vel_raw = (terms["vel"] - ref_terms["vel"]).pow(2)
            kl_w = model.kl_weight_matrix(
                horizon=d_mean_raw.shape[2],
                action_dim=d_mean_raw.shape[3],
                device=d_mean_raw.device,
                dtype=d_mean_raw.dtype,
            )
            if kl_w is None:
                d_mean_sq = d_mean_raw.mean(dim=(2, 3))
                d_vel_sq = d_vel_raw.mean(dim=(2, 3))
            else:
                denom_w = kl_w.sum().clamp(min=1e-12)
                d_mean_sq = (d_mean_raw * kl_w).sum(dim=(2, 3)) / denom_w
                d_vel_sq = (d_vel_raw * kl_w).sum(dim=(2, 3)) / denom_w
            kl_norm = d_mean_sq / (2.0 * terms["std"].pow(2))[None, :]
            if model.grpo_kl_type == "x_kl":
                kl_loss = kl_norm.mean()
            elif model.grpo_kl_type == "x":
                kl_loss = d_mean_sq.mean()
            else:
                kl_loss = d_vel_sq.mean()
            with torch.no_grad():
                extra["kl_div"] = kl_loss.detach()
                extra["kl_x_norm"] = kl_norm.mean().detach()
                extra["kl_x"] = d_mean_sq.mean().detach()
                extra["kl_v"] = d_vel_sq.mean().detach()
                per_dim = d_mean_raw.mean(dim=(0, 1, 2))
                for i, value in enumerate(per_dim.tolist()):
                    extra[f"kl_x_dim{i}"] = torch.as_tensor(
                        value, device=policy_loss.device
                    )
        total = policy_loss + model.grpo_bc_coeff * bc_loss + kl_beta * kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "kl_loss": (kl_beta * kl_loss).detach(),
        }
        metrics.update(extra)
        return (total, metrics)

    @torch.no_grad()
    def _build_eval_batch(self):
        n = max(self.eval_num_batches * self.batch_size, 1)
        n = min(n, len(self.train_dataset))
        rng = np.random.RandomState(self.seed + self.accelerator.process_index)
        indices = rng.choice(len(self.train_dataset), size=n, replace=False)
        samples = [self.train_dataset[int(i)] for i in indices]
        batch = {
            "video": torch.stack([s["video"] for s in samples], dim=0),
            "action": torch.stack([s["action"] for s in samples], dim=0),
            "proprio": torch.stack([s["proprio"] for s in samples], dim=0),
            "context": torch.stack([s["context"] for s in samples], dim=0),
            "context_mask": torch.stack([s["context_mask"] for s in samples], dim=0),
            "token": [s["token"] for s in samples],
        }
        return batch

    @torch.no_grad()
    def _eval_deterministic(self):
        if not self.eval_enabled:
            return None
        model = self.accelerator.unwrap_model(self.model)
        if self._eval_batch is None:
            self._eval_batch = self._build_eval_batch()
        batch = self._eval_batch
        input_image, action_horizon, context, context_mask, proprio, tokens = (
            self._batch_condition_inputs(batch)
        )
        cond_b = model.build_action_condition(
            input_image, action_horizon, context, context_mask, proprio
        )
        det_chain, _, _, _ = model.sample_action_chain(
            cond_b, action_horizon, deterministic=True
        )
        det_abs = self.train_dataset.denormalize_action(det_chain[:, -1])
        det_reward = self.action_reward.score_batch(
            det_abs, tokens, device=self.accelerator.device
        )
        reward_diagnostics = self._reward_eval_diagnostics("det")
        cond_bg = model.expand_condition(cond_b, self.group_size)
        sto_chain, _, _, _ = model.sample_action_chain(
            cond_bg, action_horizon, deterministic=False
        )
        sto_abs = self.train_dataset.denormalize_action(sto_chain[:, -1])
        tokens_bg = [tok for tok in tokens for _ in range(self.group_size)]
        sto_reward = self.action_reward.score_batch(
            sto_abs, tokens_bg, device=self.accelerator.device
        )
        reward_diagnostics.update(self._reward_eval_diagnostics("stoch"))
        det_xy = det_abs[..., :2].repeat_interleave(self.group_size, dim=0)
        dev_m = (sto_abs[..., :2] - det_xy).norm(dim=-1).mean()
        local = torch.tensor(
            [
                float(det_reward.mean().item()),
                float(sto_reward.mean().item()),
                float(sto_reward.std().item()),
                float(dev_m.item()),
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local).mean(dim=0)
        return {
            "eval/pdm_reward": float(gathered[0].item()),
            "eval/pdm_stoch": float(gathered[1].item()),
            "eval/pdm_stoch_std": float(gathered[2].item()),
            "eval/action_dev_m": float(gathered[3].item()),
            **reward_diagnostics,
        }

    def _reward_eval_diagnostics(self, prefix):
        return {}

    def _resume_or_load_checkpoint(self):
        if not self.resume:
            return
        resume_path = Path(str(self.resume))
        if resume_path.is_dir():
            self.load_training_state(str(resume_path))
        elif resume_path.exists():
            self.accelerator.unwrap_model(self.model).load_checkpoint(
                str(resume_path), optimizer=None
            )
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {self.resume}")

    def save_checkpoint(self):
        """Save merged inference weights and exact Accelerate resume state."""
        step_tag = f"step_{self.global_step:06d}"
        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
            model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        self.accelerator.wait_for_everyone()
        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            with open(os.path.join(state_path, "trainer_state.json"), "w") as f:
                json.dump(
                    {
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "batch_in_epoch": self.batch_in_epoch,
                    },
                    f,
                )
        self.accelerator.wait_for_everyone()
        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            payload = json.loads(state_file.read_text())
            self.global_step = int(payload["global_step"])
            self.epoch = int(payload.get("epoch", 0))
            self.batch_in_epoch = int(payload.get("batch_in_epoch", 0))
            self.train_sampler.set_epoch_offset(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        else:
            match = re.search("step[_-](\\d+)$", str(state_dir).rstrip("/"))
            self.global_step = int(match.group(1)) if match else 0
        self.accelerator.wait_for_everyone()

    def _log(self, payload: dict):
        self.accelerator.log(payload, step=self.global_step)

    def _optimizer_step(self):
        grad_norm = self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _next_batch(self):
        """Yield the next training batch, advancing the epoch (reshuffle) on exhaustion."""
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self.epoch += 1
            self.batch_in_epoch = 0
            self.train_sampler.set_epoch(self.epoch)
            self.train_sampler.clear_resume_batch_offset()
            self._data_iter = iter(self.train_loader)
            batch = next(self._data_iter)
        self.batch_in_epoch += 1
        return batch

    def train(self):
        self._set_train_mode()
        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before training.")
        unwrapped = self.accelerator.unwrap_model(self.model)
        logger.info(
            "Starting GRPO training: max_steps=%d num_inner_epochs=%d rollout_buffer_batches=%d ppo_clip=%s sde=%s sde_steps=%s/%s noise_level=%.3f discount=%.3f",
            self.max_steps,
            self.num_inner_epochs,
            self.rollout_buffer_batches,
            self.ppo_clip_range,
            unwrapped.grpo_sde_mode,
            getattr(unwrapped, "grpo_num_sde_steps", "?"),
            getattr(unwrapped, "grpo_num_inference_steps", "?"),
            getattr(unwrapped, "grpo_noise_level", float("nan")),
            getattr(unwrapped, "grpo_denoising_discount", float("nan")),
        )
        train_model = (
            self.model
            if hasattr(self.model, "build_action_condition")
            else self.accelerator.unwrap_model(self.model)
        )
        self._data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        try:
            self._train_single(train_model)
            ckpt = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] step=%d weights=%s", self.global_step, ckpt["weights_path"]
                )
        finally:
            self.accelerator.end_training()

    def _train_single(self, train_model):
        """Action rollout buffer with four inner updates per collected batch."""
        while self.global_step < self.max_steps:
            buffer = []
            for _ in range(self.rollout_buffer_batches):
                with self.accelerator.autocast():
                    buffer.append(self._rollout(train_model, self._next_batch()))
            stop = False
            for _ in range(self.num_inner_epochs):
                order = torch.randperm(len(buffer)).tolist() if len(buffer) > 1 else [0]
                for idx in order:
                    rollout = buffer[idx]
                    with self.accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            loss, loss_metrics = self._policy_loss(train_model, rollout)
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self._optimizer_step()
                            self.global_step += 1
                            self._log_step(loss, loss_metrics, rollout, grad_norm)
                            self._periodic()
                            self._maybe_visualize(train_model, rollout)
                    if self.global_step >= self.max_steps:
                        stop = True
                        break
                if stop:
                    break
            if self.global_step >= self.max_steps:
                break

    def _log_step(self, loss, loss_metrics, rollout, grad_norm):
        if self.log_every <= 0 or self.global_step % self.log_every != 0:
            return
        reward = rollout["rewards"]
        adv = rollout["advantages"]
        local = torch.tensor(
            [
                float(loss.detach().item()),
                float(reward.mean().item()),
                float(reward.std(unbiased=False).item()),
                float(adv.abs().mean().item()),
                float(loss_metrics["policy_loss"].item()),
                float(loss_metrics["bc_loss"].item()),
                float(grad_norm),
                float(rollout.get("pdm_fail_frac", 0.0)),
                float(rollout.get("zero_std_frac", 0.0)),
                float(rollout.get("nonzero_adv_frac", 1.0)),
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered = self.accelerator.gather(local).mean(dim=0)
        true_pdms = float(
            torch.nanmean(
                self.accelerator.gather(
                    torch.tensor(
                        [[float(rollout.get("true_pdms", float("nan")))]],
                        device=self.accelerator.device,
                        dtype=torch.float32,
                    )
                )
            ).item()
        )
        # EPDMS exposes additional diagnostics; fixed names keep all ranks' collectives aligned.
        reward_submetrics = {}
        names = getattr(self.action_reward, "submetric_names", ())
        if names:
            values = rollout.get("reward_submetrics") or {}
            local_sub = torch.tensor(
                [[float(values.get(name, float("nan"))) for name in names]],
                device=self.accelerator.device,
                dtype=torch.float32,
            )
            means = torch.nanmean(self.accelerator.gather(local_sub), dim=0).tolist()
            reward_submetrics = {
                f"step/epdms_{name}": float(value)
                for name, value in zip(names, means)
                if np.isfinite(value)
            }
        if not self.accelerator.is_main_process:
            return
        vals = [float(x) for x in gathered.tolist()]
        lr = float(self.optimizer.param_groups[0]["lr"])
        eta_str, sps = self._estimate_eta()
        kl_val = (
            float(loss_metrics["kl_div"].item()) if "kl_div" in loss_metrics else 0.0
        )
        logger.info(
            "[grpo] ep=%d step=%d/%d loss=%.4f reward=%.4f(±%.3f) adv=%.3f policy=%.4f bc=%.4f kl=%.3g gnorm=%.4f nz_adv=%.2f zstd=%.2f pdmfail=%.3f lr=%.2e %.2f it/s eta=%s",
            self.epoch,
            self.global_step,
            self.max_steps,
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            vals[4],
            vals[5],
            kl_val,
            vals[6],
            vals[9],
            vals[8],
            vals[7],
            lr,
            sps,
            eta_str,
        )
        self._log(
            {
                "train/loss": vals[0],
                "step/reward": vals[1],
                "step/reward_std": vals[2],
                "step/advantage_abs": vals[3],
                "step/policy_loss": vals[4],
                "step/bc_loss": vals[5],
                "train/grad_norm": vals[6],
                "train/lr": lr,
                "step/num_unavailable_tokens": float(rollout["num_unavailable"]),
                "step/adv_std": float(adv.std(unbiased=False).item()),
                "step/num_sde_steps": float(rollout.get("num_sde_steps", 0)),
                "step/pdm_fail_frac": vals[7],
                "step/zero_std_frac": vals[8],
                "step/nonzero_adv_frac": vals[9],
            }
        )
        if np.isfinite(true_pdms):
            self._log({"step/reward_true_pdms": true_pdms})
        if reward_submetrics:
            self._log(reward_submetrics)
        ppo_payload = {
            f"step/{k}": float(loss_metrics[k].item())
            for k in (
                "ratio_mean",
                "clipfrac",
                "approx_kl",
                "kl_loss",
                "kl_div",
                "kl_x_norm",
                "kl_x",
                "kl_v",
                "adv_kept_frac",
                "kl_gate_keep_frac",
                "kl_gate_div_mean",
                "kl_gate_div_max",
            )
            if k in loss_metrics
        }
        ppo_payload.update(
            {
                f"step/{k}": float(v.item())
                for (k, v) in loss_metrics.items()
                if k.startswith("kl_x_dim")
            }
        )
        if ppo_payload:
            self._log(ppo_payload)
        sub_diag = rollout.get("adv_diag") or {}
        if sub_diag:
            self._log(
                {f"step/{k}": float(v) for (k, v) in sub_diag.items() if np.isfinite(v)}
            )

    def _maybe_visualize(self, model, rollout):
        """Render action group trajectories, reference ODE and GT; visualization errors do not stop training."""
        if not self.vis_enabled or not self.accelerator.is_main_process:
            return
        if self.vis_every <= 0 or self.global_step % self.vis_every != 0:
            return
        try:
            self._visualize_rollout(model, rollout)
        except Exception as exc:
            logger.warning(
                "GRPO rollout visualization failed at step %d: %s",
                self.global_step,
                exc,
            )

    @torch.no_grad()
    def _visualize_rollout(self, model, rollout):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from navsim.common.dataclasses import Trajectory
        from navsim.visualization.bev import (
            add_configured_bev_on_ax,
            add_trajectory_to_bev_ax,
        )
        from navsim.visualization.camera import add_camera_ax
        from navsim.visualization.config import TRAJECTORY_CONFIG
        from navsim.visualization.plots import configure_ax, configure_bev_ax
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )

        ensure_dir(self.vis_dir)
        cond_b = rollout["cond_b"]
        tokens = rollout["tokens"]
        abs_group = rollout["abs_group"]
        action_horizon = int(cond_b["action_horizon"])
        horizon = int(abs_group.shape[2])
        sampling = TrajectorySampling(num_poses=horizon, interval_length=0.5)
        il_abs = rollout.get("il_abs")
        if il_abs is None:
            det_chain, _, _, _ = model.sample_action_chain(
                cond_b, action_horizon, deterministic=True
            )
            il_abs = self.train_dataset.denormalize_action(det_chain[:, -1])
        group_bev = {
            "line_color": "tab:orange",
            "line_color_alpha": 0.35,
            "line_width": 1.0,
            "line_style": "-",
            "marker": None,
            "marker_size": 0,
            "marker_edge_color": "none",
            "zorder": 2,
        }
        il_bev = {
            "line_color": "tab:blue",
            "line_color_alpha": 0.95,
            "line_width": 2.0,
            "line_style": "--",
            "marker": "o",
            "marker_size": 4,
            "marker_edge_color": "black",
            "zorder": 4,
        }
        group_cam = self._front_cam_config(
            "tab:orange", alpha=0.35, width=1.5, zorder=3
        )
        il_cam = self._front_cam_config(
            "tab:blue", alpha=0.95, width=2.5, zorder=4, line_style="--"
        )
        gt_cam = self._front_cam_config("lime", alpha=0.95, width=2.5, zorder=5)
        scene_loader = self.train_dataset.scene_loader
        n_cond = min(self.vis_num_conditions, len(tokens))
        n_samples = (
            self.group_size
            if self.vis_max_samples <= 0
            else min(self.vis_max_samples, self.group_size)
        )
        want_camera = self.vis_plot_mode == "bev_camera"
        for b in range(n_cond):
            token = tokens[b]
            scene = scene_loader.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            frame = scene.frames[frame_idx]
            gt_traj = scene.get_future_trajectory(num_trajectory_frames=horizon)
            group_trajs = [
                Trajectory(abs_group[b, g].numpy().astype(np.float32), sampling)
                for g in range(n_samples)
            ]
            il_traj = Trajectory(il_abs[b].numpy().astype(np.float32), sampling)
            has_camera = (
                want_camera
                and getattr(frame.cameras, "cam_f0", None) is not None
                and (getattr(frame.cameras.cam_f0, "image", None) is not None)
            )
            if has_camera:
                fig = plt.figure(figsize=(18, 9))
                gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
                cam_ax = fig.add_subplot(gs[0])
                bev_ax = fig.add_subplot(gs[1])
                add_camera_ax(cam_ax, frame.cameras.cam_f0)
                for traj in group_trajs:
                    self._add_traj_to_front_cam(
                        cam_ax,
                        frame.cameras.cam_f0,
                        traj.poses,
                        group_cam,
                        add_arrow=False,
                    )
                self._add_traj_to_front_cam(
                    cam_ax, frame.cameras.cam_f0, il_traj.poses, il_cam, add_arrow=True
                )
                self._add_traj_to_front_cam(
                    cam_ax, frame.cameras.cam_f0, gt_traj.poses, gt_cam, add_arrow=True
                )
                cam_ax.axis("off")
                cam_ax.set_xticks([])
                cam_ax.set_yticks([])
                cam_ax.set_aspect("auto")
            else:
                fig, bev_ax = plt.subplots(1, 1, figsize=(7, 7))
            add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
            for traj in group_trajs:
                add_trajectory_to_bev_ax(bev_ax, traj, group_bev)
            add_trajectory_to_bev_ax(bev_ax, il_traj, il_bev)
            add_trajectory_to_bev_ax(bev_ax, gt_traj, TRAJECTORY_CONFIG["human"])
            configure_bev_ax(bev_ax)
            configure_ax(bev_ax)
            handles = [
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    alpha=0.7,
                    lw=1.5,
                    label=f"GRPO samples (G={n_samples})",
                ),
                Line2D(
                    [0],
                    [0],
                    color="tab:blue",
                    lw=2.0,
                    ls="--",
                    marker="o",
                    label="IL ODE (deterministic)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=TRAJECTORY_CONFIG["human"]["line_color"],
                    lw=2.0,
                    marker="o",
                    label="GT",
                ),
            ]
            bev_ax.legend(handles=handles, loc="upper right", fontsize=8)
            reward_row = rollout["rewards"].view(len(tokens), self.group_size)[b]
            fig.suptitle(
                f"step {self.global_step} | {token} | reward mean={reward_row.mean():.3f} max={reward_row.max():.3f}",
                fontsize=10,
            )
            fig.tight_layout()
            out_path = os.path.join(
                self.vis_dir, f"step_{self.global_step:06d}_cond{b}_{token}.png"
            )
            fig.savefig(out_path, bbox_inches="tight", dpi=self.vis_dpi)
            plt.close(fig)
        logger.info(
            "[vis] step=%d wrote %d figure(s) (mode=%s) to %s",
            self.global_step,
            n_cond,
            self.vis_plot_mode,
            self.vis_dir,
        )

    @staticmethod
    def _front_cam_config(color, alpha=0.9, width=2.5, zorder=3, line_style="-"):
        return {
            "line_color": color,
            "line_color_alpha": alpha,
            "line_width": width,
            "line_style": line_style,
            "marker": None,
            "marker_size": 0,
            "marker_edge_color": "none",
            "zorder": zorder,
            "arrow_color": color,
            "arrow_edge_color": color,
            "arrow_alpha": alpha,
            "arrow_line_width": 1.5,
        }

    @staticmethod
    def _front_cam_intersection_bottom(start_pt, end_pt, width, height):
        x1, y1 = (float(start_pt[0]), float(start_pt[1]))
        x2, y2 = (float(end_pt[0]), float(end_pt[1]))
        if abs(y2 - y1) < 1e-06:
            return None
        target_y = float(height - 1)
        t = (target_y - y1) / (y2 - y1)
        if t < 0.0 or t > 1.0:
            return None
        x = x1 + t * (x2 - x1)
        if x < 0 or x > width - 1:
            return None
        return np.array([x, target_y], dtype=np.float32)

    def _add_traj_to_front_cam(self, ax, camera, poses, config, add_arrow=True):
        """Project ego-frame trajectory poses onto the front camera image (ref: plt_all_vis.py)."""
        import matplotlib.patches as patches
        from navsim.visualization.camera import _transform_pcs_to_images

        poses_2d = np.asarray(poses, dtype=np.float32)[:, :2]
        poses_3d = np.concatenate(
            [poses_2d, np.zeros((poses_2d.shape[0], 1), dtype=np.float32)], axis=1
        )
        all_poses = np.concatenate(
            [np.array([[0.0, 0.0, 0.0]], dtype=np.float32), poses_3d], axis=0
        )
        projected, in_fov = _transform_pcs_to_images(
            all_poses.T,
            camera.sensor2lidar_rotation,
            camera.sensor2lidar_translation,
            camera.intrinsics,
            img_shape=camera.image.shape[:2],
        )
        h, w = camera.image.shape[:2]
        pts = []
        first = projected[1] if len(projected) > 1 else None
        second = projected[2] if len(projected) > 2 else None
        if first is not None and in_fov[1]:
            pts.append(first)
        elif first is not None and second is not None:
            inter = self._front_cam_intersection_bottom(first, second, w, h)
            if inter is not None:
                pts.append(inter)
        for idx in range(2, len(projected)):
            if in_fov[idx]:
                pts.append(projected[idx])
        if len(pts) < 2:
            return
        pp = np.asarray(pts, dtype=np.float32)
        ax.plot(
            pp[:, 0],
            pp[:, 1],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config.get("marker"),
            markersize=config.get("marker_size", 0),
            markeredgecolor=config.get("marker_edge_color"),
            zorder=config["zorder"],
        )
        if add_arrow and len(pp) >= 2:
            last, prev = (pp[-1], pp[-2])
            dx, dy = (last[0] - prev[0], last[1] - prev[1])
            ax.add_patch(
                patches.FancyArrowPatch(
                    posA=(last[0], last[1]),
                    posB=(last[0] + dx, last[1] + dy),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    fc=config["arrow_color"],
                    ec=config["arrow_edge_color"],
                    alpha=config["arrow_alpha"],
                    linewidth=config.get("arrow_line_width", config["line_width"]),
                    connectionstyle="arc3,rad=0.0",
                    zorder=config["zorder"] + 1,
                )
            )

    def _periodic(self):
        if self.eval_every > 0 and self.global_step % self.eval_every == 0:
            swap_for_eval = False
            metrics = self._eval_deterministic()
            self.accelerator.wait_for_everyone()
            if metrics is not None and self.accelerator.is_main_process:
                logger.info(
                    "[eval] step=%d pdm=%.4f stoch=%.4f(±%.3f) dev_m=%.3f",
                    self.global_step,
                    metrics["eval/pdm_reward"],
                    metrics["eval/pdm_stoch"],
                    metrics["eval/pdm_stoch_std"],
                    metrics["eval/action_dev_m"],
                )
                self._log(metrics)
            self.accelerator.wait_for_everyone()
        if self.save_every > 0 and self.global_step % self.save_every == 0:
            ckpt = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[ckpt] step=%d weights=%s", self.global_step, ckpt["weights_path"]
                )
