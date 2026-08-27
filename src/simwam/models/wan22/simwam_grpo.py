"""GRPO fine-tuning wrapper for SimWAM's action expert.

A faithful adaptation of recogdrive's flow-matching GRPO
(`recogdrive/navsim/agents/recogdrive/recogdrive_diffusion_planner.py`:
`sample_chain` / `get_logprobs` / `forward_grpo` / `reward_fn`) to SimWAM:

* Only the **action expert** is trained; the video expert / VAE / text encoder stay
  frozen. The video branch is prefilled **once** into a K/V cache and reused, so each
  group rollout and each log-prob recompute only runs the small action DiT.
* The denoising chain uses the score-corrected SDE from FlowGRPO. Only the action
  expert's LoRA adapters are optimized; alternate RL targets and objectives are not
  part of this release.

**Time directionality (the key divergence from recogdrive).** recogdrive parameterizes
``t: 0=noise -> 1=data`` and integrates with +dt. SimWAM's
``WanContinuousFlowMatchScheduler`` uses ``sigma=t/T``: ``sigma=0`` is data, ``sigma=1``
is noise; ``build_inference_schedule`` walks ``sigma: 1 -> 0`` with **negative** deltas;
the predicted velocity is ``noise - data`` and ``step`` does ``x += v * dsigma``. This
module therefore drives sampling/log-prob with SimWAM's own schedule and feeds the model
``timestep = sigma_k * T`` (decreasing), never recogdrive's increasing ``t``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

import torch
from torch.distributions import Normal

from simwam.utils.logging_config import get_logger

from .simwam import SimWAM

logger = get_logger(__name__)


class SimWAMGRPO(SimWAM):
    """SimWAM subclass adding action-expert GRPO sampling / log-prob / reference policy."""

    # ----- GRPO configuration --------------------------------------------------------
    def configure_grpo(self, grpo_cfg: Dict[str, Any]) -> None:
        """Store GRPO knobs and snapshot the (warm-started) action expert as the BC reference.

        Must be called *after* the IL checkpoint is loaded so the reference == IL policy.
        """
        sample = dict(grpo_cfg.get("sample", {}))
        train = dict(grpo_cfg.get("train", {}))

        self.grpo_group_size = int(sample.get("group_size", 8))
        self.grpo_num_inference_steps = int(sample.get("num_inference_steps", 10))
        infer_shift = sample.get("infer_shift", None)
        self.grpo_infer_shift = None if infer_shift is None else float(infer_shift)
        self.grpo_min_std = float(sample.get("min_std", 1e-3))
        self.grpo_randn_clip = float(sample.get("randn_clip", 5.0))
        self.grpo_sde_mode = "rigorous"
        if str(sample.get("sde_mode", "rigorous")) != "rigorous":
            raise ValueError("SimWAM only releases grpo.sample.sde_mode=rigorous.")
        self.grpo_noise_level = float(sample.get("noise_level", 0.1))
        # sigma==1 (first step) divide-by-zero guard for g=noise_level*sqrt(sigma/(1-sigma)):
        # use the second-largest sigma in the inference schedule (flow_grpo's `sigmas[1]`).
        sigma_steps = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.grpo_num_inference_steps, device="cpu", dtype=torch.float32,
            shift_override=self.grpo_infer_shift,
        )[0] / float(self.infer_action_scheduler.num_train_timesteps)
        self.grpo_sigma_max_guard = float(sigma_steps[1].item()) if sigma_steps.numel() > 1 else 0.999
        if self.grpo_noise_level > 0.3:
            logger.warning(
                "rigorous SDE noise_level=%.2f may be too large for the %d-dim action space "
                "(std ~ noise_level*sqrt(sigma/(1-sigma)) explodes at high sigma -> reward may "
                "collapse to ~0); consider ~0.1 (flow_grpo's 0.7 is tuned for image latents).",
                self.grpo_noise_level, int(self.action_expert.action_dim),
            )

        self.grpo_denoising_discount = float(train.get("denoising_discount", 0.6))
        self.grpo_adv_q_lo = float(train.get("adv_clip_lower_quantile", 0.0))
        self.grpo_adv_q_hi = float(train.get("adv_clip_upper_quantile", 1.0))
        self.grpo_adv_eps = float(train.get("adv_eps", 1e-8))
        if not bool(train.get("use_bc_loss", True)):
            raise ValueError("SimWAM FlowGRPO requires grpo.train.use_bc_loss=true.")
        self.grpo_use_bc_loss = True
        self.grpo_bc_coeff = float(train.get("bc_coeff", 0.1))

        self.ref_action_expert = copy.deepcopy(self.action_expert)
        self.ref_action_expert.eval()
        for p in self.ref_action_expert.parameters():
            p.requires_grad_(False)

        # LoRA is applied after the reference snapshot, so the BC reference remains the IL policy.
        lora = dict(grpo_cfg.get("lora", {}))
        if not bool(lora.get("enabled", False)):
            raise ValueError("SimWAM FlowGRPO requires grpo.lora.enabled=true.")
        self.lora_enabled = True
        from .lora import apply_lora_to_module

        self.lora_r = int(lora.get("r", 16))
        self.lora_alpha = float(lora.get("alpha", 32.0))
        self.lora_dropout = float(lora.get("dropout", 0.0))
        self.lora_target_modules = list(lora.get("target_modules", ["q", "k", "v", "o"]))
        if not self.lora_target_modules:
            raise ValueError("grpo.lora.enabled=true but `target_modules` is empty.")
        num_wrapped = apply_lora_to_module(
            self.action_expert,
            target_names=self.lora_target_modules,
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        if num_wrapped == 0:
            raise ValueError(
                f"LoRA enabled but no Linear layers matched target_modules={self.lora_target_modules}; "
                "check the names (expected e.g. ['q','k','v','o'])."
            )
        logger.info(
            "Action expert fine-tuning = LoRA: wrapped %d Linear layers (r=%d alpha=%.1f dropout=%.2f targets=%s)",
            num_wrapped, self.lora_r, self.lora_alpha, self.lora_dropout, self.lora_target_modules,
        )

        logger.info(
            "Configured FlowGRPO: group_size=%d num_inference_steps=%d noise_level=%.4f "
            "discount=%.3f bc_coeff=%.3f",
            self.grpo_group_size,
            self.grpo_num_inference_steps,
            self.grpo_noise_level,
            self.grpo_denoising_discount,
            self.grpo_bc_coeff,
        )

    # ----- condition (frozen video) prefill ------------------------------------------
    @torch.no_grad()
    def build_action_condition(
        self,
        input_image: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> Dict[str, Any]:
        """Prefill the frozen video K/V cache for a batch of conditions (one frame each).

        Args:
            input_image: [B, 3, H, W] current-frame images.
            action_horizon: number of action steps (action token count).
            context / context_mask: cached text embeddings [B, L, D] / [B, L].
            proprio: optional ego state [B, Dp] appended to the text context.

        Returns a condition dict consumed by the action sampler/log-prob.
        """
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("GRPO action rollout requires `video_attention_mask_mode='first_frame_causal'`.")
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [B, 3, H, W], got {tuple(input_image.shape)}")

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = input_image.shape[0]
        first_frame_latents = torch.cat(
            [self._encode_input_image_latents_tensor(input_image[i : i + 1], tiled=tiled) for i in range(batch_size)],
            dim=0,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        timestep_video = torch.zeros((batch_size,), dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(action_horizon),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return {
            "video_kv_cache": video_kv_cache,
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": int(action_horizon),
        }

    def expand_condition(self, cond: Dict[str, Any], group_size: int) -> Dict[str, Any]:
        """Repeat a condition `group_size` times along the batch dim (same-condition group)."""
        g = int(group_size)
        kv = [
            {"k": layer["k"].repeat_interleave(g, dim=0), "v": layer["v"].repeat_interleave(g, dim=0)}
            for layer in cond["video_kv_cache"]
        ]
        return {
            "video_kv_cache": kv,
            "attention_mask": cond["attention_mask"],
            "video_seq_len": cond["video_seq_len"],
            "context": cond["context"].repeat_interleave(g, dim=0),
            "context_mask": cond["context_mask"].repeat_interleave(g, dim=0),
            "action_horizon": cond["action_horizon"],
        }

    # ----- action expert forward with the cached frozen video ------------------------
    def _run_action_expert_with_cache(self, expert, action_pre: Dict[str, Any], cond: Dict[str, Any]):
        """Run an arbitrary action `expert` against the cached video K/V (grad-enabled).

        Mirrors `MoT.forward_action_with_video_cache` but parameterized by `expert`
        so the same path serves both the trained expert and the frozen reference.
        """
        mot = self.mot
        x = action_pre["tokens"]
        action_freqs = action_pre["freqs"]
        action_t_mod = action_pre["t_mod"]
        action_ctx = {"context": action_pre["context"], "mask": action_pre["context_mask"]}
        video_kv_cache = cond["video_kv_cache"]
        video_seq_len = cond["video_seq_len"]
        action_seq_len = int(x.shape[1])
        total_seq_len = video_seq_len + action_seq_len
        action_attention_mask = cond["attention_mask"][video_seq_len:total_seq_len, :total_seq_len]

        for layer_idx in range(mot.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = mot._build_expert_attention_io(
                expert=expert, block=block, x=x, freqs=action_freqs, t_mod=action_t_mod
            )
            layer_cache = video_kv_cache[layer_idx]
            k_cat = torch.cat([layer_cache["k"], k_action], dim=1)
            v_cat = torch.cat([layer_cache["v"], v_action], dim=1)
            mixed = mot._mixed_attention(
                q_cat=q_action, k_cat=k_cat, v_cat=v_cat, attention_mask=action_attention_mask
            )
            x = mot._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_ctx,
            )
        return x

    def action_velocity(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        cond: Dict[str, Any],
        use_ref: bool = False,
    ) -> torch.Tensor:
        """Predicted flow **velocity** for the action latents given the cached condition.

        For x-prediction models the clean-sample output is converted to velocity via
        `_action_x_to_v` (identical to `SimWAM.infer_action`).
        """
        expert = self.ref_action_expert if use_ref else self.action_expert
        action_pre = expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=cond["context"],
            context_mask=cond["context_mask"],
        )
        action_tokens = self._run_action_expert_with_cache(expert, action_pre, cond)
        pred = expert.post_dit(action_tokens, action_pre)
        if self.action_prediction_type == "sample":
            pred = self._action_x_to_v(pred, latents_action, timestep_action)
        return pred

    def _sde_mean_std(self, x: torch.Tensor, v: torch.Tensor, sigma_k: float, delta_k: float):
        """Rigorous score-corrected SDE step (flow_grpo `sd3_sde_with_logprob`), in SimWAM's
        sigma:1->0 / Delta-sigma<0 convention (already sign-aligned with flow_grpo).

        g = noise_level * sqrt(sigma/(1-sigma));  std = g * sqrt(-dsigma)
        mean = x*(1 + g^2/(2 sigma)*dsigma) + v*(1 + g^2 (1-sigma)/(2 sigma))*dsigma
        At g->0 (or deterministic) this reduces to the Euler step x + v*dsigma.
        """
        sig = self.grpo_sigma_max_guard if sigma_k >= 1.0 else sigma_k  # sigma==1 divide-by-zero guard
        g = self.grpo_noise_level * math.sqrt(sig / (1.0 - sig))
        g2 = g * g
        # Compute the drift in fp32 (g^2/(2 sigma) blows up at high sigma; bf16 is unsafe),
        # then cast the mean back to the input dtype for chain/sampler consistency (matches flow_grpo).
        xf, vf = x.float(), v.float()
        mean = xf * (1.0 + g2 / (2.0 * sig) * delta_k) + vf * (1.0 + g2 * (1.0 - sig) / (2.0 * sig)) * delta_k
        std = g * math.sqrt(max(-float(delta_k), 0.0))
        return mean.to(x.dtype), max(std, self.grpo_min_std)

    # ----- stochastic denoising chain ------------------------------------------------
    @torch.no_grad()
    def sample_action_chain(
        self,
        cond: Dict[str, Any],
        action_horizon: int,
        deterministic: bool = False,
        init_actions: Optional[torch.Tensor] = None,
        velocity_use_ref: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """Roll out the stochastic (or deterministic) denoising chain.

        Returns:
            chain: [B, N+1, H, A] latents from x_0 (noise) to x_N (clean), detached.
            timesteps: [N] SimWAM scheduler timesteps (sigma*T, **decreasing**).
            deltas: [N] sigma deltas (negative).
        """
        sched = self.infer_action_scheduler
        device = self.device
        dtype = self.torch_dtype
        batch_size = int(cond["context"].shape[0])
        action_dim = int(self.action_expert.action_dim)

        timesteps, deltas = sched.build_inference_schedule(
            num_inference_steps=self.grpo_num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=self.grpo_infer_shift,
        )
        num_steps = int(timesteps.shape[0])

        if init_actions is not None:
            x = init_actions.to(device=device, dtype=dtype)
        else:
            x = self._randn((batch_size, action_horizon, action_dim), device, dtype, generator)

        chain = [x.clone()]
        for k in range(num_steps):
            timestep_action = timesteps[k].to(device=device, dtype=dtype).reshape(1).expand(batch_size)
            v = self.action_velocity(x, timestep_action, cond, use_ref=velocity_use_ref)
            sigma_k = float(timesteps[k].item()) / float(sched.num_train_timesteps)
            if deterministic:
                x = sched.step(v, deltas[k], x)
            else:
                mean, std = self._sde_mean_std(x, v, sigma_k, float(deltas[k].item()))
                eps = self._randn(mean.shape, device, dtype, generator).clamp(
                    -self.grpo_randn_clip, self.grpo_randn_clip
                )
                x = mean + std * eps
            chain.append(x.clone())

        chain = torch.stack(chain, dim=1)
        return chain.detach(), timesteps.detach(), deltas.detach()

    def action_chain_logprobs(
        self,
        cond: Dict[str, Any],
        chain: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        use_ref: bool = False,
    ) -> torch.Tensor:
        """Per-step Gaussian log-prob of a recorded chain under the (current/ref) policy.

        Recomputes each step's mean with the same SimWAM schedule used for sampling and
        evaluates ``Normal(mean, logprob_std).log_prob(x_{k+1})``, clamped and reduced over
        the (horizon, action_dim) dims -> [B, N]. Mirrors recogdrive `get_logprobs`.
        """
        sched = self.infer_action_scheduler
        device = self.device
        dtype = self.torch_dtype
        batch_size = int(chain.shape[0])
        num_steps = int(timesteps.shape[0])

        logps = []
        for k in range(num_steps):
            x_k = chain[:, k]
            x_next = chain[:, k + 1]
            timestep_action = timesteps[k].to(device=device, dtype=dtype).reshape(1).expand(batch_size)
            v = self.action_velocity(x_k, timestep_action, cond, use_ref=use_ref)
            sigma_k = float(timesteps[k].item()) / float(sched.num_train_timesteps)
            mean, std = self._sde_mean_std(x_k, v, sigma_k, float(deltas[k].item()))
            dist = Normal(mean.float(), torch.tensor(std, device=device, dtype=torch.float32))
            lp = dist.log_prob(x_next.float()).clamp(min=-5.0, max=2.0).mean(dim=(1, 2))
            logps.append(lp)
        return torch.stack(logps, dim=1)  # [B, N]

    # ----- helpers -------------------------------------------------------------------
    def load_checkpoint(self, path, optimizer=None):
        """Load a `{mot, ...}` checkpoint, remapping vanilla keys for a LoRA-wrapped model.

        Warm-start (before `configure_grpo`) hits the vanilla path. After LoRA is applied,
        a merged/vanilla `mot` (keys `<p>.weight`) is remapped to `<p>.base.weight` so the
        **base** weights actually load (adapters keep their init: A=kaiming, B=0). This is a
        valid restart point but NOT bit-exact — for exact mid-run resume use the accelerate
        state dir (`resume=<...>/checkpoints/state/step_*`), which preserves the LoRA tensors.
        """
        if not getattr(self, "lora_enabled", False):
            return super().load_checkpoint(path, optimizer=optimizer)

        import torch as _torch

        from .lora import remap_vanilla_to_lora_state_dict

        payload = _torch.load(path, map_location="cpu")
        if "mot" in payload:
            state = payload["mot"]
            if not any(".lora_" in k for k in state):  # vanilla/merged -> remap to LoRA layout
                state = remap_vanilla_to_lora_state_dict(self.mot, state)
            self.mot.load_state_dict(state, strict=False)
        else:
            raise ValueError(f"LoRA checkpoint missing `mot` key: {path}")
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save a SimWAM-compatible `{mot, ...}` checkpoint.

        With LoRA, the action expert's adapters are merged back into plain Linear weights so
        the saved `mot` loads into a vanilla SimWAM/`ActionDiT` (e.g. NavSim eval) unchanged.
        """
        if not getattr(self, "lora_enabled", False):
            return super().save_checkpoint(path, optimizer=optimizer, step=step)

        import torch as _torch

        from .lora import merged_state_dict

        payload = {
            "mot": merged_state_dict(self.mot),  # LoRA merged -> plain Linear keys
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "lora_merged": True,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        _torch.save(payload, path)

    def _randn(self, shape, device, dtype, generator: Optional[torch.Generator]) -> torch.Tensor:
        if generator is not None:
            return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
                device=device, dtype=dtype
            )
        return torch.randn(shape, device=device, dtype=dtype)
