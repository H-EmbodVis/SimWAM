"""GRPO trainer for SimWAM's action expert (recogdrive-faithful, NavSim PDM reward).

Structure mirrors `simwam.trainer_tensorboard.Wan22Trainer` (Accelerate + DeepSpeed
ZeRO-1, ResumableEpochSampler, capped-warmup LR schedule, TensorBoard, accelerate-state
checkpointing) but the optimizer trains **only the action expert** and each step runs a
group rollout -> PDM reward -> clipped group-relative PPO (+ BC anchor) update instead of
the supervised flow-matching loss.
"""

import json
import os
import re
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .datasets.navsim.pdm_reward import NavSimPDMReward
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler

logger = get_logger(__name__)


class SimWAMGRPOTrainer:
    def __init__(self, model, train_dataset, reward: NavSimPDMReward, *, cfg: DictConfig):
        self.model = model
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
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.eval_every = int(cfg.eval_every)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.lr_warmup_steps = int(cfg.get("lr_warmup_steps", 100))
        self.resume = cfg.resume

        grpo = cfg.grpo
        self.group_size = int(grpo.sample.group_size)
        self.adv_q_lo = float(grpo.train.get("adv_clip_lower_quantile", 0.0))
        self.adv_q_hi = float(grpo.train.get("adv_clip_upper_quantile", 1.0))
        self.adv_eps = float(grpo.train.get("adv_eps", 1e-8))
        ppo_clip = grpo.train.get("ppo_clip_range", None)
        if ppo_clip is None:
            raise ValueError("SimWAM FlowGRPO requires grpo.train.ppo_clip_range.")
        self.ppo_clip_range = float(ppo_clip)
        adv_clip = grpo.train.get("adv_clip_max", None)
        if adv_clip is None:
            raise ValueError("SimWAM FlowGRPO requires grpo.train.adv_clip_max.")
        self.adv_clip_max = float(adv_clip)
        self.num_inner_epochs = int(grpo.train.get("num_inner_epochs", 1))
        self.rollout_buffer_batches = int(grpo.train.get("rollout_buffer_batches", 1))
        if self.num_inner_epochs < 1:
            raise ValueError(f"grpo.train.num_inner_epochs must be >= 1, got {self.num_inner_epochs}")
        if self.rollout_buffer_batches < 1:
            raise ValueError(f"grpo.train.rollout_buffer_batches must be >= 1, got {self.rollout_buffer_batches}")
        self.eval_enabled = bool(grpo.get("eval", {}).get("enabled", True))
        self.eval_num_batches = int(grpo.get("eval", {}).get("num_batches", 4))

        # Rollout visualization (BEV: group samples + IL ODE + GT on one figure).
        vis = grpo.get("vis", {})
        self.vis_enabled = bool(vis.get("enabled", False))
        self.vis_every = int(vis.get("every", 100))
        self.vis_num_conditions = int(vis.get("num_conditions", 2))
        self.vis_max_samples = int(vis.get("max_samples", 0))  # <=0 -> all G samples
        self.vis_plot_mode = str(vis.get("plot_mode", "bev_camera"))  # "bev_camera" | "bev"
        if self.vis_plot_mode not in {"bev_camera", "bev"}:
            raise ValueError(f"grpo.vis.plot_mode must be 'bev_camera' or 'bev', got {self.vis_plot_mode!r}")
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
        logger.info(
            "GRPO Accelerate: distributed_type=%s world_size=%d process_index=%d mp=%s group_size=%d",
            self.accelerator.distributed_type,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.group_size,
        )
        # The loss is computed via the unwrapped module and reduced in accelerator.backward,
        # which is correct for DeepSpeed ZeRO-1 but NOT for plain multi-GPU DDP (no allreduce
        # without routing through model.forward). Fail fast if launched without DeepSpeed.
        if self.accelerator.num_processes > 1 and "DEEPSPEED" not in str(self.accelerator.distributed_type).upper():
            raise RuntimeError(
                "SimWAMGRPOTrainer requires DeepSpeed (ZeRO-1) for multi-GPU training "
                "(plain DDP gradient sync is not wired). Launch via "
                "scripts/train_navsim_grpo_zero1_torchrun.sh (sets ACCELERATE_USE_DEEPSPEED=true)."
            )

        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)

        # Train only the action expert; everything else (video/vae/text/proprio/ref) frozen.
        self._apply_action_only_train_mode(self.model)
        trainable_params = [p for p in self.model.action_expert.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info("GRPO trainable (action expert) params: %.3f M", num_trainable / 1e6)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
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
        for d in (self.output_dir, self.checkpoint_root, self.weights_dir, self.state_dir):
            ensure_dir(d)
        if self.vis_enabled:
            ensure_dir(self.vis_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.init_trackers("grpo")
        self._resume_or_load_checkpoint()
        self._eval_batch = None
        logger.info("Train dataset size: %d", len(self.train_dataset))

    # ----- frozen / trainable setup --------------------------------------------------
    @staticmethod
    def _apply_action_only_train_mode(model):
        # Keep the WHOLE model in eval() (incl. the action expert): the policy must be
        # deterministic apart from the injected SDE noise, so old_logp/new_logp match for PPO
        # (LoRA dropout would otherwise desync them). eval() does not affect autograd; only
        # dropout/BN. Trainability is set via requires_grad below. (No-op for dropout=0.)
        model.eval()
        model.requires_grad_(False)
        if not getattr(model, "lora_enabled", False):
            raise ValueError("SimWAM FlowGRPO trains LoRA adapters only.")
        for name, p in model.action_expert.named_parameters():
            p.requires_grad_("lora_" in name)

    def _set_action_only_train_mode(self):
        self._apply_action_only_train_mode(self.accelerator.unwrap_model(self.model))

    # ----- dataloader / scheduler (copied from Wan22Trainer) -------------------------
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
        # STEP-controlled: if `max_steps` is set it is the training length (the loop runs
        # `while global_step < max_steps`), and `num_epochs` is ignored. `num_epochs` is only
        # used to derive total steps as a fallback when `max_steps is None`.
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1)
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), max(total_train_steps - 1, 0))
        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining_steps, eta_min=self.learning_rate * 0.01)
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}.")
        if warmup_steps <= 0:
            return main_scheduler
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps
        )
        return SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    # ----- batch -> condition --------------------------------------------------------
    def _batch_condition_inputs(self, batch):
        """Extract (input_image, action_horizon, context, context_mask, proprio, tokens)."""
        video = batch["video"]
        if video.ndim != 5:
            raise ValueError(f"`batch['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        input_image = video[:, :, 0].contiguous()  # current frame [B,3,H,W]
        action = batch["action"]
        action_horizon = int(action.shape[1])
        context = batch["context"]
        context_mask = batch["context_mask"]
        proprio = None
        if self.accelerator.unwrap_model(self.model).proprio_encoder is not None:
            proprio = batch["proprio"][:, 0, :]  # [B, Dp]
        tokens = list(batch["token"])
        return input_image, action_horizon, context, context_mask, proprio, tokens

    # ----- rollout + reward + advantage (no grad) ------------------------------------
    @torch.no_grad()
    def _rollout(self, model, batch):
        input_image, action_horizon, context, context_mask, proprio, tokens = self._batch_condition_inputs(batch)
        cond_b = model.build_action_condition(input_image, action_horizon, context, context_mask, proprio)
        cond_bg = model.expand_condition(cond_b, self.group_size)

        chain, timesteps, deltas = model.sample_action_chain(cond_bg, action_horizon, deterministic=False)
        final_norm = chain[:, -1]
        abs_poses = self.train_dataset.denormalize_action(final_norm)  # [B*G, H, 3] cpu float
        tokens_bg = [tok for tok in tokens for _ in range(self.group_size)]
        rewards = self.reward.score_batch(abs_poses, tokens_bg, device=self.accelerator.device)  # [B*G]

        bsz = len(tokens)
        rewards_mat = rewards.view(bsz, self.group_size)
        mean_r = rewards_mat.mean(dim=1, keepdim=True)
        # population std (unbiased=False): well-defined for group_size==1 (-> 0, zero advantage)
        # and is the standard group-relative normalization.
        std_r = rewards_mat.std(dim=1, unbiased=False, keepdim=True) + self.adv_eps
        advantages = ((rewards_mat - mean_r) / std_r).reshape(-1)
        if self.adv_q_lo > 0.0 or self.adv_q_hi < 1.0:
            lo = torch.quantile(advantages, self.adv_q_lo)
            hi = torch.quantile(advantages, self.adv_q_hi)
            advantages = advantages.clamp(min=lo, max=hi)
        advantages = advantages.detach()

        num_unavailable = int(sum(0 if self.reward.available(t) else 1 for t in tokens))

        # For consistent visualization: sample the deterministic ODE trajectory under the SAME
        # (pre-optimizer-step) policy snapshot as the group samples. Under rollout reuse (mu>1 /
        # buffer>1) this rollout drives several optimizer steps, so compute the ODE if ANY step in
        # the upcoming buffer window is a viz step (else `_visualize_rollout` falls back to recompute).
        il_abs = None
        steps_ahead = max(self.num_inner_epochs * self.rollout_buffer_batches, 1)
        will_visualize = (
            self.vis_enabled
            and self.accelerator.is_main_process
            and self.vis_every > 0
            and any((self.global_step + 1 + j) % self.vis_every == 0 for j in range(steps_ahead))
        )
        if will_visualize:
            det_chain, _, _ = model.sample_action_chain(cond_b, action_horizon, deterministic=True)
            il_abs = self.train_dataset.denormalize_action(det_chain[:, -1])

        return {
            "cond_b": cond_b,
            "cond_bg": cond_bg,
            "chain": chain,
            "timesteps": timesteps,
            "deltas": deltas,
            "rewards": rewards.detach(),
            "advantages": advantages,
            "num_unavailable": num_unavailable,
            "tokens": tokens,
            # group sampled trajectories in absolute ego frame, [B, G, H, 3] (for visualization)
            "abs_group": abs_poses.reshape(bsz, self.group_size, abs_poses.shape[1], abs_poses.shape[2]),
            # deterministic ODE trajectory [B, H, 3] under the same snapshot (None unless viz this step)
            "il_abs": il_abs,
            "old_logp": model.action_chain_logprobs(cond_bg, chain, timesteps, deltas).detach(),
        }

    # ----- policy (+ BC) loss (grad on action expert) --------------------------------
    def _policy_loss(self, model, rollout):
        chain = rollout["chain"]
        timesteps = rollout["timesteps"]
        deltas = rollout["deltas"]
        advantages = rollout["advantages"]
        num_steps = int(timesteps.shape[0])

        new_logp = model.action_chain_logprobs(rollout["cond_bg"], chain, timesteps, deltas)  # [B*G, N]
        denoise_idx = torch.arange(num_steps, device=new_logp.device)
        discount = model.grpo_denoising_discount ** (num_steps - denoise_idx - 1)  # [N], last step weighted most
        adv_weighted = advantages[:, None] * discount[None, :]
        adv_weighted = adv_weighted.clamp(-self.adv_clip_max, self.adv_clip_max)

        old_logp = rollout["old_logp"]
        ratio = torch.exp(new_logp - old_logp)
        unclipped = -adv_weighted * ratio
        clipped = -adv_weighted * ratio.clamp(1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range)
        policy_loss = torch.max(unclipped, clipped).mean()

        with torch.no_grad():
            ref_chain, ref_ts, ref_deltas = model.sample_action_chain(
                rollout["cond_b"], rollout["cond_b"]["action_horizon"], deterministic=False, velocity_use_ref=True
            )
        bc_logp = model.action_chain_logprobs(rollout["cond_b"], ref_chain, ref_ts, ref_deltas)
        bc_loss = -bc_logp.mean()

        total = policy_loss + model.grpo_bc_coeff * bc_loss
        with torch.no_grad():
            metrics = {
                "policy_loss": policy_loss.detach(),
                "bc_loss": bc_loss.detach(),
                "ratio_mean": ratio.mean().detach(),
                "clipfrac": (ratio.sub(1.0).abs() > self.ppo_clip_range).float().mean().detach(),
                "approx_kl": (old_logp - new_logp).mean().detach(),
            }
        return total, metrics

    # ----- deterministic eval --------------------------------------------------------
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
        input_image, action_horizon, context, context_mask, proprio, tokens = self._batch_condition_inputs(batch)
        cond_b = model.build_action_condition(input_image, action_horizon, context, context_mask, proprio)

        det_chain, _, _ = model.sample_action_chain(cond_b, action_horizon, deterministic=True)
        det_abs = self.train_dataset.denormalize_action(det_chain[:, -1])
        det_reward = self.reward.score_batch(det_abs, tokens, device=self.accelerator.device)

        cond_bg = model.expand_condition(cond_b, self.group_size)
        sto_chain, _, _ = model.sample_action_chain(cond_bg, action_horizon, deterministic=False)
        sto_abs = self.train_dataset.denormalize_action(sto_chain[:, -1])
        tokens_bg = [tok for tok in tokens for _ in range(self.group_size)]
        sto_reward = self.reward.score_batch(sto_abs, tokens_bg, device=self.accelerator.device)

        # sampled-vs-deterministic trajectory deviation in meters (xy of the abs poses).
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
        }

    # ----- checkpoint (copied from Wan22Trainer) -------------------------------------
    def _resume_or_load_checkpoint(self):
        if not self.resume:
            return
        resume_path = Path(str(self.resume))
        if resume_path.is_dir():
            self.load_training_state(str(resume_path))
        elif resume_path.exists():
            self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {self.resume}")

    def save_checkpoint(self):
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
                    {"global_step": self.global_step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch}, f
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
            match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
            self.global_step = int(match.group(1)) if match else 0
        self.accelerator.wait_for_everyone()

    # ----- training loop -------------------------------------------------------------
    def _log(self, payload: dict):
        self.accelerator.log(payload, step=self.global_step)

    def _optimizer_step(self):
        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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
            self.train_sampler.set_epoch(self.epoch)  # reshuffle differently each epoch
            self.train_sampler.clear_resume_batch_offset()
            self._data_iter = iter(self.train_loader)
            batch = next(self._data_iter)
        self.batch_in_epoch += 1
        return batch

    def train(self):
        self._set_action_only_train_mode()
        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before training.")
        logger.info(
            "Starting GRPO training: max_steps=%d num_inner_epochs=%d rollout_buffer_batches=%d ppo_clip=%s sde=%s",
            self.max_steps, self.num_inner_epochs, self.rollout_buffer_batches,
            self.ppo_clip_range, self.accelerator.unwrap_model(self.model).grpo_sde_mode,
        )
        # Run the rollout + loss through the prepared engine (DeepSpeed). For ZeRO-1 the
        # gradient reduction happens in `accelerator.backward`, so this matches the proven
        # SFT trainer path. NOTE: this trainer assumes DeepSpeed (plain DDP unsupported).
        train_model = self.model if hasattr(self.model, "build_action_condition") else self.accelerator.unwrap_model(self.model)
        self._data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        try:
            while self.global_step < self.max_steps:
                # --- sampling phase: collect a buffer of rollouts under the theta_old snapshot ---
                buffer = []
                for _ in range(self.rollout_buffer_batches):
                    with self.accelerator.autocast():
                        buffer.append(self._rollout(train_model, self._next_batch()))

                # --- optimization phase: reuse each rollout over `num_inner_epochs` (ratio/clip) ---
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

            if self.save_final_checkpoint:
                ckpt = self.save_checkpoint()
                if self.accelerator.is_main_process:
                    logger.info("[done] step=%d weights=%s", self.global_step, ckpt["weights_path"])
            elif self.accelerator.is_main_process:
                logger.info("[done] step=%d (final checkpoint disabled)", self.global_step)
        finally:
            self.accelerator.end_training()

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
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered = self.accelerator.gather(local).mean(dim=0)
        if not self.accelerator.is_main_process:
            return
        vals = [float(x) for x in gathered.tolist()]
        lr = float(self.optimizer.param_groups[0]["lr"])
        eta_str, sps = self._estimate_eta()
        logger.info(
            "[grpo] ep=%d step=%d/%d loss=%.4f reward=%.4f(±%.3f) adv=%.3f policy=%.4f bc=%.4f lr=%.2e %.2f it/s eta=%s",
            self.epoch, self.global_step, self.max_steps, vals[0], vals[1], vals[2],
            vals[3], vals[4], vals[5], lr, sps, eta_str,
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
            }
        )
        # PPO-specific diagnostics (only present when ppo_clip_range is set).
        ppo_payload = {
            f"step/{k}": float(loss_metrics[k].item())
            for k in ("ratio_mean", "clipfrac", "approx_kl")
            if k in loss_metrics
        }
        if ppo_payload:
            self._log(ppo_payload)

    # ----- rollout visualization -----------------------------------------------------
    def _maybe_visualize(self, model, rollout):
        """Render a few conditions' rollout (group samples + IL ODE + GT) on BEV figures.

        Main-process only, gated by `vis_every`, wrapped so a viz failure never kills training.
        """
        if not self.vis_enabled or not self.accelerator.is_main_process:
            return
        if self.vis_every <= 0 or self.global_step % self.vis_every != 0:
            return
        try:
            self._visualize_rollout(model, rollout)
        except Exception as exc:  # viz must never break training
            logger.warning("GRPO rollout visualization failed at step %d: %s", self.global_step, exc)

    @torch.no_grad()
    def _visualize_rollout(self, model, rollout):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from navsim.common.dataclasses import Trajectory
        from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
        from navsim.visualization.camera import add_camera_ax
        from navsim.visualization.config import TRAJECTORY_CONFIG
        from navsim.visualization.plots import configure_ax, configure_bev_ax
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        ensure_dir(self.vis_dir)
        cond_b = rollout["cond_b"]
        tokens = rollout["tokens"]
        abs_group = rollout["abs_group"]  # [B, G, H, 3] cpu
        action_horizon = int(cond_b["action_horizon"])
        horizon = int(abs_group.shape[2])
        sampling = TrajectorySampling(num_poses=horizon, interval_length=0.5)

        # IL trajectory = deterministic ODE rollout (no injected noise). Prefer the one captured
        # in `_rollout` under the SAME pre-update snapshot as the group samples; recompute only
        # as a fallback.
        il_abs = rollout.get("il_abs")
        if il_abs is None:
            det_chain, _, _ = model.sample_action_chain(cond_b, action_horizon, deterministic=True)
            il_abs = self.train_dataset.denormalize_action(det_chain[:, -1])  # [B, H, 3] cpu

        # BEV configs (add_trajectory_to_bev_ax keys)
        group_bev = {"line_color": "tab:orange", "line_color_alpha": 0.35, "line_width": 1.0,
                     "line_style": "-", "marker": None, "marker_size": 0, "marker_edge_color": "none", "zorder": 2}
        il_bev = {"line_color": "tab:blue", "line_color_alpha": 0.95, "line_width": 2.0,
                  "line_style": "--", "marker": "o", "marker_size": 4, "marker_edge_color": "black", "zorder": 4}
        # Front-camera configs (projected; group has no arrow to avoid clutter)
        group_cam = self._front_cam_config("tab:orange", alpha=0.35, width=1.5, zorder=3)
        il_cam = self._front_cam_config("tab:blue", alpha=0.95, width=2.5, zorder=4, line_style="--")
        gt_cam = self._front_cam_config("lime", alpha=0.95, width=2.5, zorder=5)

        scene_loader = self.train_dataset.scene_loader
        n_cond = min(self.vis_num_conditions, len(tokens))
        n_samples = self.group_size if self.vis_max_samples <= 0 else min(self.vis_max_samples, self.group_size)
        want_camera = self.vis_plot_mode == "bev_camera"

        for b in range(n_cond):
            token = tokens[b]
            scene = scene_loader.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            frame = scene.frames[frame_idx]
            gt_traj = scene.get_future_trajectory(num_trajectory_frames=horizon)
            group_trajs = [Trajectory(abs_group[b, g].numpy().astype(np.float32), sampling) for g in range(n_samples)]
            il_traj = Trajectory(il_abs[b].numpy().astype(np.float32), sampling)

            has_camera = want_camera and getattr(frame.cameras, "cam_f0", None) is not None and getattr(frame.cameras.cam_f0, "image", None) is not None
            if has_camera:
                fig = plt.figure(figsize=(18, 9))
                gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
                cam_ax = fig.add_subplot(gs[0])
                bev_ax = fig.add_subplot(gs[1])
                # --- front camera: project trajectories onto the image ---
                add_camera_ax(cam_ax, frame.cameras.cam_f0)
                for traj in group_trajs:
                    self._add_traj_to_front_cam(cam_ax, frame.cameras.cam_f0, traj.poses, group_cam, add_arrow=False)
                self._add_traj_to_front_cam(cam_ax, frame.cameras.cam_f0, il_traj.poses, il_cam, add_arrow=True)
                self._add_traj_to_front_cam(cam_ax, frame.cameras.cam_f0, gt_traj.poses, gt_cam, add_arrow=True)
                cam_ax.axis("off")
                cam_ax.set_xticks([])
                cam_ax.set_yticks([])
                cam_ax.set_aspect("auto")
            else:
                fig, bev_ax = plt.subplots(1, 1, figsize=(7, 7))

            # --- BEV ---
            add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
            for traj in group_trajs:
                add_trajectory_to_bev_ax(bev_ax, traj, group_bev)
            add_trajectory_to_bev_ax(bev_ax, il_traj, il_bev)
            add_trajectory_to_bev_ax(bev_ax, gt_traj, TRAJECTORY_CONFIG["human"])
            configure_bev_ax(bev_ax)
            configure_ax(bev_ax)
            handles = [
                Line2D([0], [0], color="tab:orange", alpha=0.7, lw=1.5, label=f"GRPO samples (G={n_samples})"),
                Line2D([0], [0], color="tab:blue", lw=2.0, ls="--", marker="o", label="IL ODE (deterministic)"),
                Line2D([0], [0], color=TRAJECTORY_CONFIG["human"]["line_color"], lw=2.0, marker="o", label="GT"),
            ]
            bev_ax.legend(handles=handles, loc="upper right", fontsize=8)
            reward_row = rollout["rewards"].view(len(tokens), self.group_size)[b]
            fig.suptitle(
                f"step {self.global_step} | {token} | reward mean={reward_row.mean():.3f} max={reward_row.max():.3f}",
                fontsize=10,
            )
            fig.tight_layout()
            out_path = os.path.join(self.vis_dir, f"step_{self.global_step:06d}_cond{b}_{token}.png")
            fig.savefig(out_path, bbox_inches="tight", dpi=self.vis_dpi)
            plt.close(fig)

        logger.info("[vis] step=%d wrote %d figure(s) (mode=%s) to %s", self.global_step, n_cond, self.vis_plot_mode, self.vis_dir)

    @staticmethod
    def _front_cam_config(color, alpha=0.9, width=2.5, zorder=3, line_style="-"):
        return {
            "line_color": color, "line_color_alpha": alpha, "line_width": width, "line_style": line_style,
            "marker": None, "marker_size": 0, "marker_edge_color": "none", "zorder": zorder,
            "arrow_color": color, "arrow_edge_color": color, "arrow_alpha": alpha, "arrow_line_width": 1.5,
        }

    @staticmethod
    def _front_cam_intersection_bottom(start_pt, end_pt, width, height):
        x1, y1 = float(start_pt[0]), float(start_pt[1])
        x2, y2 = float(end_pt[0]), float(end_pt[1])
        if abs(y2 - y1) < 1e-6:
            return None
        target_y = float(height - 1)
        t = (target_y - y1) / (y2 - y1)
        if t < 0.0 or t > 1.0:
            return None
        x = x1 + t * (x2 - x1)
        if x < 0 or x > (width - 1):
            return None
        return np.array([x, target_y], dtype=np.float32)

    def _add_traj_to_front_cam(self, ax, camera, poses, config, add_arrow=True):
        """Project ego-frame trajectory poses onto the front camera image (ref: plt_all_vis.py)."""
        import matplotlib.patches as patches

        from navsim.visualization.camera import _transform_pcs_to_images

        poses_2d = np.asarray(poses, dtype=np.float32)[:, :2]
        poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1), dtype=np.float32)], axis=1)
        all_poses = np.concatenate([np.array([[0.0, 0.0, 0.0]], dtype=np.float32), poses_3d], axis=0)
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
            pp[:, 0], pp[:, 1],
            color=config["line_color"], alpha=config["line_color_alpha"], linewidth=config["line_width"],
            linestyle=config["line_style"], marker=config.get("marker"), markersize=config.get("marker_size", 0),
            markeredgecolor=config.get("marker_edge_color"), zorder=config["zorder"],
        )
        if add_arrow and len(pp) >= 2:
            last, prev = pp[-1], pp[-2]
            dx, dy = last[0] - prev[0], last[1] - prev[1]
            ax.add_patch(patches.FancyArrowPatch(
                posA=(last[0], last[1]), posB=(last[0] + dx, last[1] + dy), arrowstyle="-|>",
                mutation_scale=12, fc=config["arrow_color"], ec=config["arrow_edge_color"],
                alpha=config["arrow_alpha"], linewidth=config.get("arrow_line_width", config["line_width"]),
                connectionstyle="arc3,rad=0.0", zorder=config["zorder"] + 1,
            ))

    def _periodic(self):
        if self.eval_every > 0 and self.global_step % self.eval_every == 0:
            metrics = self._eval_deterministic()
            self.accelerator.wait_for_everyone()
            if metrics is not None and self.accelerator.is_main_process:
                logger.info(
                    "[eval] step=%d pdm=%.4f stoch=%.4f(±%.3f) dev_m=%.3f",
                    self.global_step, metrics["eval/pdm_reward"], metrics["eval/pdm_stoch"],
                    metrics["eval/pdm_stoch_std"], metrics["eval/action_dev_m"],
                )
                self._log(metrics)
            self.accelerator.wait_for_everyone()
        if self.save_every > 0 and self.global_step % self.save_every == 0:
            ckpt = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info("[ckpt] step=%d weights=%s", self.global_step, ckpt["weights_path"])
