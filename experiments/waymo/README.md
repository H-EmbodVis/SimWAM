# Waymo Open Dataset E2E — Supervised Training and Evaluation

SimWAM on the Waymo Open Dataset End-to-End Driving benchmark: a single FRONT
camera at 480&times;512, native 20&times;2 XY actions (5 s at 4 Hz), and
`[vx, vy, ax, ay, command(4)]` proprioception.

All commands run from the repository root and use relative paths.

---

## Data layout

Manifests live in `data/`; images are resolved relative to `data/waymo/`
(override with `WAYMO_DATA_ROOT`):

```text
data/
  waymo_training_front_video4s_traj5s_4hz_xy.jsonl        # 333,537 IL training rows
  waymo_val_front_current_traj5s_4hz_xy_pref_only.jsonl   #   479 scored-reference rows (val + RL)
  waymo_test_front_current_input.jsonl                    # 1,505 label-free test rows
  waymo_dataset_stats.json                                # q1/q99 action normalization
  waymo/
    images/training/<scene_id>/FRONT/<frame_idx>.jpg
    images/val/<scene_id>/FRONT/<frame_idx>.jpg
    images/test/<scene_id>/FRONT/<frame_idx>.jpg
    metadata/{training,val,test}.jsonl                    # only needed to rebuild manifests
    test_sequence_frames_for_submission.json              # only needed to rebuild the test manifest
  text_embeds_cache/navsim/<sha256(prompt)>.t5_len256.wan22ti2v5b.pt
```

Manifest image paths are **relative to `image_root`** — no absolute or remote
location is ever recorded. For example
`images/training/003b6282.../FRONT/009.jpg` resolves to
`data/waymo/images/training/003b6282.../FRONT/009.jpg`.

Training rows carry the current FRONT frame plus the frames at +1/+2/+3/+4 s;
validation and test rows carry the current frame only. Validation rows also carry
three scored reference trajectories per sample, which drive the RFS reward
(see [GRPO.md](GRPO.md)).

### Rebuilding the manifests

The manifests are generated from the native extracted WOD-E2E metadata. Point
`WAYMO_METADATA_DIR` at the directory holding `training.jsonl` / `val.jsonl` /
`test.jsonl`, place the camera frames under `data/waymo/images/`, then run:

```bash
# IL training rows (current frame + four future frames)
python scripts/preprocess_waymo_train.py --split training

# RL/validation rows (current frame only, scored references required)
python scripts/preprocess_waymo_train.py --split val

# Official test rows (label-free)
python scripts/preprocess_waymo_test.py

# Action normalization statistics, computed over the training rows only
python scripts/compute_waymo_traj_stats.py
```

Useful flags: `--image-root`, `--image-prefix`, `--image-check {list,decode,none}`,
`--trajectory-horizon-s {4,5}`, `--limit`, `--max-scenes`, `--force`.

---

## Supervised training

```bash
NPROC_PER_NODE=8 \
bash scripts/train_waymo_zero1_torchrun.sh \
  task=waymo_uncond_front_512x480_1e-4
```

The launcher verifies the manifests, statistics and prompt cache before starting,
prints the resolved `torchrun` command, and writes to
`./runs/waymo_uncond_front_512x480_1e-4/<run-id>/`. Pass `--dry-run` to print the
command without launching.

Environment overrides: `WAYMO_TRAIN_JSONL`, `WAYMO_VAL_JSONL`, `WAYMO_STATS_PATH`,
`WAYMO_DATA_ROOT`, `WAYMO_TEXT_EMBED_CACHE`. Any extra argument is forwarded to
Hydra, e.g. `learning_rate=5e-5 num_workers=4`.

Validation runs action-only inference over all 479 scored rows and reports
ADE/FDE/RFS; results land in `<output_dir>/eval/step_XXXXXX/`.

---

## Evaluation

```bash
CKPT=./weights/SimWAM-Waymo-IL.pt \
NPROC_PER_NODE=8 \
bash experiments/waymo/run_eval_waymo_action_only.sh
```

Outputs `metrics.json` (ADE/FDE at 1/3/5 s, RFS, ADE against the best reference)
and a per-sample `predictions.jsonl`. Results default to
`./runs/waymo_eval/<checkpoint-tag>/`; a checkpoint under
`.../checkpoints/weights/step_XXXXXX.pt` writes next to its run instead.
Override with `EVALUATION.output_dir=...`.

Evaluation calls `model.infer_action`: it encodes only the current frame, caches
its video K/V, then denoises the action. No video generation and no joint
training loss are involved; ground-truth and reference trajectories are used only
for scoring after inference.

---

## Test-set prediction and official submission

```bash
CKPT=./weights/SimWAM-Waymo-RL.pt \
NPROC_PER_NODE=8 \
bash experiments/waymo/run_predict_waymo_test.sh
```

This writes `predictions.jsonl` and `prediction_summary.json` under
`./runs/waymo_test/<model>_<checkpoint>/`. The test manifest carries no labels, so
no local metric is available.

To package an official submission, point `WAYMO_OPEN_DATASET_SRC` at the `src/`
directory of a [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
checkout, write a challenge metadata JSON (your registered `account_name` email,
`unique_method_name`, `num_model_parameters` such as `1B`,
`uses_public_model_pretraining`, and optionally `authors`, `affiliation`,
`description`, `method_link`, `public_model_names`), then run:

```bash
python experiments/waymo/make_test_submission.py \
  --predictions runs/waymo_test/<model>_<checkpoint>/predictions.jsonl \
  --metadata-json ./my_submission_metadata.json \
  --output-dir runs/waymo_test/<model>_<checkpoint>/submission
```

The official protobuf descriptors are loaded from that checkout rather than from
generated code, so no particular protobuf runtime version is required. See
[TEST_SUBMISSION.md](TEST_SUBMISSION.md).
