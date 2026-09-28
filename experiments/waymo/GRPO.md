# Waymo RFS Reinforcement Learning (Action LoRA)

GRPO on the Waymo E2E benchmark: the action expert's LoRA adapters are optimized
against the official **Rater Feedback Score (RFS)** on the 479 validation rows
that carry scored reference trajectories, warm-started from the supervised
checkpoint.

Those 479 rows are the optimization set for this recipe. The RFS/ADE/FDE reported
during training are measured on the same rows, and the evaluation files are tagged
`evaluation_data_role=rl_training_set`.

---

## Launch

```bash
NPROC_PER_NODE=8 \
bash scripts/train_waymo_grpo_zero1_torchrun.sh \
  task=waymo_grpo_rfs \
  num_epochs=10 max_steps=null
```

Add `--dry-run` to print the resolved `torchrun` command without launching.
Outputs go to `./runs_grpo/waymo_grpo_rfs/<run-id>/`.

Configs: [`configs/train_waymo_grpo.yaml`](../../configs/train_waymo_grpo.yaml)
→ task [`waymo_grpo_rfs.yaml`](../../configs/task/waymo_grpo_rfs.yaml)
→ data [`waymo_grpo_val.yaml`](../../configs/data/waymo_grpo_val.yaml).

The supervised warm start defaults to `./weights/SimWAM-Waymo-IL.pt`; override with
`model.checkpoint_path=...` or `WAYMO_IL_CHECKPOINT`. The loader strictly validates
the `mot` / `proprio_encoder` payload and the 2-dimensional action head. The frozen
IL reference policy is copied and the LoRA adapters are added **after** that load,
so the reference is exactly the supplied IL checkpoint.

---

## Recipe

The recipe matches the NAVSIM RL preset
([`configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml`](../../configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml));
only the KL dimension weights change, because Waymo actions are `[x, y]` instead
of `[x, y, heading]`.

| Item | Setting |
| --- | --- |
| Per-GPU condition batch size | 1 |
| Candidates per condition (G) | 8 |
| Denoising steps | 10 |
| Stochastic SDE steps | indices [7, 8, 9] (0-based) |
| SDE | `rigorous`, `noise_level=0.1`, windows not merged |
| Log-prob clamp | `relative`, width 7 |
| PPO clip range | 0.005 |
| Inner updates per rollout batch | 4 |
| Rollout buffer batches | 1 |
| Advantage | group-mean centered, normalized by the cross-rank reward std |
| Zero-variance groups | contribute no policy gradient; the IL anchor still applies |
| Advantage clip | 5 |
| BC loss | off |
| IL reference anchor | `kl_type=x`, `kl_beta=50` |
| KL dimension weights | `[x, y] = [0.2, 1.0]` |
| KL gate | off |
| LoRA | r=16, alpha=32, dropout=0, on the action expert's q/k/v/o |
| Learning rate | 5e-5, constant, warmup up to 100 steps |
| Precision / distributed | BF16, DeepSpeed ZeRO-1 |
| Epochs | 10 full passes over the data |

Only the action LoRA parameters are updated. The video DiT, VAE, text conditioning,
proprio encoder, the action base weights and the copied IL reference expert are all
frozen.

---

## Data and reward flow

1. Load the 479 rows of `data/waymo_val_front_current_traj5s_4hz_xy_pref_only.jsonl`.
2. Read the current FRONT frame from `data/waymo/images/...` and resize to
   480&times;512. Only the current frame is read; no future frames are required.
3. The state is `[vx, vy, ax, ay, command(4)]`; the fixed prompt and its T5 cache
   are the same as during supervised training.
4. Build the current frame's video K/V cache once per condition and reuse it for
   all 8 candidates.
5. Denoise a normalized 5 s / 4 Hz / 20-point XY trajectory per candidate.
6. Denormalize to meters with `data/waymo_dataset_stats.json`. The q1/q99 bounds
   always come from the 333,537 supervised training rows, never re-estimated on
   the 479 validation rows.
7. Look up the current speed and the three scored reference trajectories for that
   token, and score each candidate with the official RFS.
8. Turn those rewards into advantages and run the 4 PPO / IL-anchor updates.

The reward lives in
[`rfs_reward.py`](../../src/simwam/datasets/waymo/rfs_reward.py). `score_batch`
takes `[B*G, 20, 2]` and returns `[B*G]`: every candidate is scored independently,
with no group mean and no division by 10. The vendored official NumPy
implementation supplies the trust region, thresholds and decay; reference
truncation and end-point padding are handled there too. Predicted XY points are
neither interpolated, nor extended with headings, nor clipped.

RFS does not use ground-truth ADE/FDE as a weight — those are evaluation only.
An invalid prediction or a token without reference trajectories raises, so a data
error can never masquerade as a low reward.

---

## How "10 epochs" is counted

[`WaymoGRPOTrainer`](../../src/simwam/trainer_waymo_grpo.py) reuses the shared
rollout and loss, and adds Waymo collation, epoch counting and full-dataset
evaluation.
With the default 8 GPUs at batch size 1:

```text
rollout batches per data epoch = ceil(479 / (8 * 1)) = 60
optimizer steps per data epoch = 60 * 4 inner updates = 240
10 data epochs                 = 240 * 10 = 2400 optimizer steps
```

Each epoch has one duplicated training slot from distributed batch alignment, so
all 479 rows are still covered. Evaluation shards by index and always covers
exactly 479 distinct rows.

The generic GRPO step estimator does not multiply by the inner-update count; the
Waymo subclass corrects this. Passing an explicit `max_steps` overrides the epoch
budget, so the 10-epoch command keeps `max_steps=null`.

---

## Saving, evaluation and resuming

By default a checkpoint is saved and evaluated every 240 optimizer steps — one
data epoch at 8 GPUs / batch 1. The final step always triggers a full evaluation,
even when it does not divide `eval_every`.

TensorBoard and `train_grpo.log` are written to the output directory. Key series:

- `step/RFS`, `step/RFS_std` — candidate scores of the training rollouts.
- `step/nonzero_adv_frac`, `step/zero_std_frac` — whether RFS discriminates between candidates.
- `step/kl_x`, `step/kl_x_dim0`, `step/kl_x_dim1` — the anchor against the frozen IL policy.
- `eval/RFS`, `eval/ADE`, `eval/FDE` — deterministic action evaluation on the same 479 rows.

`checkpoints/weights/step_XXXXXX.pt` holds the LoRA-merged MoT and proprio weights,
so the plain action-only evaluator works on it directly:

```bash
NPROC_PER_NODE=8 \
bash experiments/waymo/run_eval_waymo_action_only.sh \
  ckpt=./runs_grpo/waymo_grpo_rfs/<run-id>/checkpoints/weights/step_002400.pt
```

`checkpoints/state/step_XXXXXX/` keeps the LoRA adapters, the frozen reference, the
optimizer and the progress counters for resuming. Keep the same batch/GPU
configuration:

```bash
NPROC_PER_NODE=8 \
bash scripts/train_waymo_grpo_zero1_torchrun.sh \
  resume=./runs_grpo/waymo_grpo_rfs/<run-id>/checkpoints/state/step_001200
```
