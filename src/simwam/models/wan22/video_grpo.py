"""Video-only GRPO sampling, kept separate from the retained action recipes.

The current frame is clamped at every transition. Stochastic transitions are
Gaussian in FP32, with matching sampling/scoring parameters and no density clamp.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from numbers import Integral
import math
import torch
from torch.distributions import Normal
from simwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class VideoGRPOMixin:
    def _configure_video_grpo(self, vcfg: Dict[str, Any]) -> None:
        """Video-DiT GRPO knobs (target=video): SDE sampler params + video-expert fine-tuning."""
        self.grpo_video_coupling = str(vcfg.get("coupling", "video_only"))
        if self.grpo_video_coupling != "video_only":
            raise NotImplementedError(
                f"grpo.video.coupling={self.grpo_video_coupling!r} not implemented; MVP supports 'video_only'."
            )
        # video_only feeds action=None to the video expert; that is only valid when the video expert
        # is NOT action-conditioned (else pre_dit requires an action for multi-frame video).
        if bool(getattr(self.video_expert, "action_conditioned", False)):
            raise ValueError(
                "grpo.video.coupling=video_only requires a non-action-conditioned video expert "
                "(video_dit_config.action_conditioned=false). Use a joint coupling (not yet implemented) "
                "to reinforce an action-conditioned video model."
            )
        if getattr(self, "mot_attention_mask_mode", "isolated") == "bidirectional":
            raise ValueError("video_only GRPO cannot reproduce bidirectional Action→Video conditioning.")
        self.grpo_video_reward = str(vcfg.get("reward", "traj"))
        self.grpo_video_sde_mode = str(vcfg.get("sde_mode", "rigorous"))
        if self.grpo_video_sde_mode not in {"bridge", "rigorous"}:
            raise ValueError("grpo.video.sde_mode must be 'bridge' or 'rigorous'.")
        self.grpo_video_noise_level = float(vcfg.get("noise_level", 0.1))
        self.grpo_video_num_inference_steps = int(vcfg.get("num_inference_steps", 20))
        self.grpo_video_train_sde_steps = int(vcfg.get("train_sde_steps", 1))
        self.grpo_video_sde_transition = str(vcfg.get("sde_transition", "existing"))
        if self.grpo_video_sde_transition not in {"existing", "linear_noise"}:
            raise ValueError("grpo.video.sde_transition must be 'existing' or 'linear_noise'.")
        if self.grpo_video_sde_transition == "linear_noise" and self.grpo_video_sde_mode != "rigorous":
            raise ValueError("linear_noise requires grpo.video.sde_mode=rigorous.")
        indices = vcfg.get("sde_step_indices")
        self.grpo_video_sde_step_indices = None
        if indices is not None:
            try:
                indices = tuple(indices)
            except TypeError as exc:
                raise ValueError("grpo.video.sde_step_indices must be a sequence of integers.") from exc
            if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)
                    or len(set(indices)) != len(indices)
                    or any(i < 0 or i >= self.grpo_video_num_inference_steps for i in indices)):
                raise ValueError("grpo.video.sde_step_indices must contain unique in-range integer indices.")
            if len(indices) != self.grpo_video_train_sde_steps:
                raise ValueError("grpo.video.sde_step_indices count must equal train_sde_steps.")
            self.grpo_video_sde_step_indices = tuple(sorted(indices))
        self.grpo_video_sample_noise_std = float(vcfg.get("sample_noise_std", 0.1))
        lp = vcfg.get("logprob_noise_std", None)
        self.grpo_video_logprob_noise_std = self.grpo_video_sample_noise_std if lp is None else float(lp)
        self.grpo_video_anneal_noise = bool(vcfg.get("anneal_noise", False))
        self.grpo_video_min_std = float(vcfg.get("min_std", 1e-3))
        if vcfg.get("randn_clip") is not None:
            raise ValueError("Video Gaussian sampling requires grpo.video.randn_clip=null.")
        if self.grpo_video_logprob_noise_std != self.grpo_video_sample_noise_std:
            raise ValueError("Video sampling and log-probability must use the same noise std.")
        for name, value in (("sample_noise_std", self.grpo_video_sample_noise_std),
                            ("noise_level", self.grpo_video_noise_level),
                            ("min_std", self.grpo_video_min_std)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"grpo.video.{name} must be finite and positive.")
        self.grpo_video_denoising_discount = float(vcfg.get("denoising_discount", 0.6))
        if not 0.0 <= self.grpo_video_denoising_discount <= 1.0:
            raise ValueError("grpo.video.denoising_discount must be between 0 and 1.")
        self.grpo_video_reward_only_generated = bool(vcfg.get("reward_only_generated_frames", True))
        self.grpo_video_gradient_checkpointing = bool(vcfg.get("gradient_checkpointing", True))
        self.grpo_video_offload_rollout = bool(vcfg.get("offload_rollout_to_cpu", True))
        if not 1 <= self.grpo_video_train_sde_steps <= self.grpo_video_num_inference_steps:
            raise ValueError("Video train_sde_steps must be between 1 and num_inference_steps.")
        sigmas = self.infer_video_scheduler.build_inference_schedule(
            self.grpo_video_num_inference_steps, "cpu", torch.float32,
        )[0] / float(self.infer_video_scheduler.num_train_timesteps)
        self.grpo_video_sigma_max_guard = float(sigmas[1]) if len(sigmas) > 1 else 0.999
        if self.grpo_video_gradient_checkpointing:
            self.video_expert.use_gradient_checkpointing = True  # enable per-block checkpointing
        finetune = str(vcfg.get("finetune", "lora")).lower()
        if finetune not in ("lora", "full", "ffn"):
            raise ValueError(f"grpo.video.finetune must be 'lora'|'full'|'ffn', got {finetune!r}")
        self.video_finetune_mode = finetune
        if self.video_finetune_mode == "lora":
            from .lora import apply_lora_to_module

            lcfg = dict(vcfg.get("lora", {}))
            r = int(lcfg.get("rank", lcfg.get("r", 16)))
            alpha = float(lcfg.get("alpha", 32.0))
            drop = float(lcfg.get("dropout", 0.0))
            if drop > 0.0:
                logger.warning(
                    "grpo.video.lora.dropout=%.3f is ignored during video GRPO: the video expert runs "
                    "in eval() so old/new log-probs stay self-consistent (dropout would desync them).", drop,
                )
            targets = list(lcfg.get("target_modules", ["q", "k", "v", "o"]))
            n = apply_lora_to_module(self.video_expert, target_names=targets, r=r, alpha=alpha, dropout=drop)
            if n == 0:
                raise ValueError(f"video LoRA matched 0 layers for targets={targets}")
            self.video_lora_enabled = True
            logger.info("Video expert fine-tuning = LoRA: wrapped %d layers (r=%d alpha=%.1f).", n, r, alpha)
        elif self.video_finetune_mode == "ffn":
            logger.info("Video expert fine-tuning = FFN (only blocks.*.ffn.* params trainable).")
        else:
            logger.info("Video expert fine-tuning = FULL (all 5B video-expert params trainable).")
        logger.info(
            "Configured VIDEO GRPO: reward=%s N=%d train_sde_steps=%d sample_std=%.3f discount=%.3f gckpt=%s offload=%s indices=%s transition=%s",
            self.grpo_video_reward, self.grpo_video_num_inference_steps, self.grpo_video_train_sde_steps,
            self.grpo_video_sample_noise_std, self.grpo_video_denoising_discount,
            self.grpo_video_gradient_checkpointing, self.grpo_video_offload_rollout,
            self.grpo_video_sde_step_indices, self.grpo_video_sde_transition,
        )

    def _video_step_std(self, sigma_k: float, base_std: float) -> float:
        std = base_std * (float(sigma_k) if self.grpo_video_anneal_noise else 1.0)
        return max(std, self.grpo_video_min_std)

    def _video_transition(self, x, velocity, sigma, delta):
        """One Gaussian transition, shared by rollout and log-prob recomputation."""
        x, velocity = x.float(), velocity.float()
        delta = float(delta)
        if self.grpo_video_sde_transition == "linear_noise":
            # Same marginal-preserving SDE as the early-five-step experiments.
            # g(sigma)=noise_level*sigma stays finite at the first step, sigma=1.
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(delta) or delta >= 0:
                raise ValueError("linear_noise requires finite sigma > 0 and delta < 0.")
            g = self.grpo_video_noise_level * sigma
            g2 = g * g
            mean = (x * (1.0 + g2 / (2.0 * sigma) * delta)
                    + velocity * (1.0 + g2 * (1.0 - sigma) / (2.0 * sigma)) * delta)
            return mean, g * math.sqrt(-delta)
        if self.grpo_video_sde_mode == "bridge":
            return x + velocity * delta, self._video_step_std(sigma, self.grpo_video_sample_noise_std)
        # Same score-corrected formula and sigma=1 finite-step guard as action GRPO.
        sigma = self.grpo_video_sigma_max_guard if sigma >= 1.0 else float(sigma)
        g2 = self.grpo_video_noise_level ** 2 * sigma / (1.0 - sigma)
        mean = (x * (1.0 + g2 / (2.0 * sigma) * delta)
                + velocity * (1.0 + g2 * (1.0 - sigma) / (2.0 * sigma)) * delta)
        return mean, max(math.sqrt(g2 * -delta), self.grpo_video_min_std)

    @torch.no_grad()
    def build_video_condition(
        self,
        input_image: torch.Tensor,
        num_video_frames: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> Dict[str, Any]:
        """Condition for video-DiT GRPO: clean first-frame latent + text(+proprio) + the video
        self-attention mask. Video is the DENOISED variable (not prefilled/cached)."""
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [B, 3, H, W], got {tuple(input_image.shape)}")
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if int(num_video_frames) <= 1 or (int(num_video_frames) - 1) % temporal_factor:
            raise ValueError("Video GRPO needs 1 + k * VAE temporal factor frames, with k >= 1.")
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = input_image.shape[0]
        first_frame_latents = torch.cat(
            [self._encode_input_image_latents_tensor(input_image[i : i + 1], tiled=tiled) for i in range(batch_size)],
            dim=0,
        )  # [B, z, 1, h, w]
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio.to(device=self.device, dtype=self.torch_dtype)
            )
        z_dim = int(first_frame_latents.shape[1])
        latent_h = int(first_frame_latents.shape[-2])
        latent_w = int(first_frame_latents.shape[-1])
        latent_t = (int(num_video_frames) - 1) // int(self.vae.temporal_downsample_factor) + 1
        # Probe pre_dit to get the full-video seq len + tokens/frame for the self-attention mask.
        probe = torch.zeros((batch_size, z_dim, latent_t, latent_h, latent_w), device=self.device, dtype=self.torch_dtype)
        video_pre = self.video_expert.pre_dit(
            x=probe, timestep=torch.zeros((batch_size,), device=self.device, dtype=self.torch_dtype),
            context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        video_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len, video_tokens_per_frame=tokens_per_frame, device=self.device,
        )
        return {
            "first_frame_latents": first_frame_latents,
            "fuse_flag": fuse_flag,
            "context": context,
            "context_mask": context_mask,
            "latent_shape": (z_dim, latent_t, latent_h, latent_w),
            "video_attn_mask": video_attn_mask,
            "num_video_frames": int(num_video_frames),
        }

    def expand_video_condition(self, cond: Dict[str, Any], group_size: int) -> Dict[str, Any]:
        g = int(group_size)
        out = dict(cond)
        out["first_frame_latents"] = cond["first_frame_latents"].repeat_interleave(g, dim=0)
        out["context"] = cond["context"].repeat_interleave(g, dim=0)
        out["context_mask"] = cond["context_mask"].repeat_interleave(g, dim=0)
        return out

    def video_velocity(self, latents_video: torch.Tensor, timestep_video: torch.Tensor, cond: Dict[str, Any]) -> torch.Tensor:
        """Predicted video flow velocity (video-only self-attention; grad-enabled)."""
        video_pre = self.video_expert.pre_dit(
            x=latents_video, timestep=timestep_video, context=cond["context"], context_mask=cond["context_mask"],
            action=None, fuse_vae_embedding_in_latents=cond["fuse_flag"],
        )
        tokens = self.mot.forward_video_only(
            video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"], video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=cond["video_attn_mask"],
        )
        return self.video_expert.post_dit(tokens, video_pre)

    @torch.no_grad()
    def sample_video_chain(self, cond: Dict[str, Any], deterministic: bool = False,
                           init_noise: Optional[torch.Tensor] = None, generator=None):
        """Use explicit SDE indices, or the last `train_sde_steps` by default.

        The clean first frame is re-injected every step (never noised / never in log-prob). Returns
        only the final latent + the stochastic transitions (x_in, x_out) needed for log-prob, to
        keep memory bounded (optionally offloaded to CPU).
        """
        sched = self.infer_video_scheduler
        device, dtype = self.device, self.torch_dtype
        batch_size = int(cond["context"].shape[0])
        z_dim, latent_t, latent_h, latent_w = cond["latent_shape"]
        first_frame = cond["first_frame_latents"]
        timesteps, deltas = sched.build_inference_schedule(
            num_inference_steps=self.grpo_video_num_inference_steps, device=device, dtype=dtype,
        )
        num_steps = int(timesteps.shape[0])
        if num_steps != self.grpo_video_num_inference_steps:
            raise RuntimeError("Video scheduler returned an unexpected number of denoising steps.")
        if deterministic:
            stoch_steps = set()
        elif self.grpo_video_sde_step_indices is not None:
            stoch_steps = set(self.grpo_video_sde_step_indices)
        else:
            stoch_steps = set(range(max(num_steps - self.grpo_video_train_sde_steps, 0), num_steps))

        if init_noise is not None:
            x = init_noise.to(device=device, dtype=dtype).clone()
        else:
            x = torch.randn((batch_size, z_dim, latent_t, latent_h, latent_w),
                            device=device, dtype=dtype, generator=generator)
        if tuple(x.shape) != (batch_size, z_dim, latent_t, latent_h, latent_w):
            raise ValueError("init_noise shape does not match the video condition.")
        x[:, :, 0:1] = first_frame

        x_ins, x_outs, s_ts, s_deltas = [], [], [], []
        for k in range(num_steps):
            t = timesteps[k].to(device=device, dtype=dtype).reshape(1).expand(batch_size)
            v = self.video_velocity(x.to(dtype=dtype), t, cond)
            if k in stoch_steps:
                sigma_k = float(timesteps[k].item()) / float(sched.num_train_timesteps)
                mean, std = self._video_transition(x, v, sigma_k, float(deltas[k]))
                eps = torch.randn(mean.shape, device=device, dtype=torch.float32, generator=generator)
                x_new = mean + std * eps
                x_new[:, :, 0:1] = first_frame
                store = (lambda z: z.detach().float().cpu().clone()) if self.grpo_video_offload_rollout else (lambda z: z.detach().float().clone())
                x_ins.append(store(x))
                x_outs.append(store(x_new))
                s_ts.append(timesteps[k].detach())
                s_deltas.append(deltas[k].detach())
                x = x_new
            else:
                x = sched.step(v.to(dtype=x.dtype), deltas[k], x)
                x[:, :, 0:1] = first_frame

        x_in = torch.stack(x_ins, dim=1) if x_ins else None      # [B, n_stoch, z, lt, h, w]
        x_out = torch.stack(x_outs, dim=1) if x_outs else None
        stoch_timesteps = torch.stack(s_ts) if s_ts else timesteps.new_zeros(0)
        stoch_deltas = torch.stack(s_deltas) if s_deltas else deltas.new_zeros(0)
        return {"final": x.detach(), "x_in": x_in, "x_out": x_out,
                "stoch_timesteps": stoch_timesteps, "stoch_deltas": stoch_deltas}

    def video_chain_logprobs(self, cond: Dict[str, Any], x_in, x_out, stoch_timesteps, stoch_deltas) -> torch.Tensor:
        """Per-(stochastic)-step Gaussian log-prob over GENERATED frames only -> [B, n_stoch].
        Sum coordinate log-probs for the true transition density. Reduction is in
        float64; loss scaling by dimension belongs outside the PPO ratio.
        Grad flows through the video expert velocity (first frame excluded).
        """
        sched = self.infer_video_scheduler
        device, dtype = self.device, self.torch_dtype
        batch_size = int(x_in.shape[0])
        n_stoch = int(x_in.shape[1])
        logps = []
        for j in range(n_stoch):
            x_k = x_in[:, j].to(device=device, dtype=torch.float32)
            x_next = x_out[:, j].to(device=device, dtype=torch.float32)
            t = stoch_timesteps[j].to(device=device, dtype=dtype).reshape(1).expand(batch_size)
            v = self.video_velocity(x_k.to(dtype=dtype), t, cond)
            sigma_k = float(stoch_timesteps[j].item()) / float(sched.num_train_timesteps)
            mean, std = self._video_transition(x_k, v, sigma_k, float(stoch_deltas[j]))
            mean_gen = mean[:, :, 1:].float()          # exclude the clean first frame
            x_next_gen = x_next[:, :, 1:].float()
            dist = Normal(mean_gen, torch.tensor(std, device=device, dtype=torch.float32))
            lp = dist.log_prob(x_next_gen).double().sum(dim=(1, 2, 3, 4))  # [B]
            logps.append(lp)
        return torch.stack(logps, dim=1)  # [B, n_stoch]
