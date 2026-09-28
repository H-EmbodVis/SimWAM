# Waymo E2E Test Submission Packaging

Converts SimWAM's native `predictions.jsonl` into the official
`E2EDChallengeSubmission` binproto shards and a flat `tar.gz` archive.

---

## Inputs

| Input | Source |
| --- | --- |
| `--predictions` | `runs/waymo_test/<model>_<checkpoint>/predictions.jsonl`, written by `experiments/waymo/run_predict_waymo_test.sh` |
| `--frames` | `data/waymo/test_sequence_frames_for_submission.json`, the official per-scene frame index |
| `--metadata-json` | challenge metadata (see below) |
| `--waymo-src` | `src/` of a [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) checkout; defaults to `$WAYMO_OPEN_DATASET_SRC` |

`predictions.jsonl` must cover every official frame exactly once, in order, with
20 XY points per frame (5 s at 4 Hz, ego frame, x forward / y left, meters).

### Metadata JSON

`run_predict_waymo_test.sh` already writes most of it: the `checkpoint.json` next
to `predictions.jsonl` carries `num_model_parameters`,
`uses_public_model_pretraining`, `public_model_names` and the checkpoint step.
Copy it and add your own identity fields.

Required: `account_name` (your registered Waymo email), `unique_method_name`,
`num_model_parameters` (an integer with a `K`/`M`/`B`/`T` suffix), and
`uses_public_model_pretraining` (an explicit boolean; when `true`,
`public_model_names` must be a non-empty list of strings).

Optional strings: `affiliation`, `description`, `method_link`. Optional lists:
`authors`, `public_model_names`.

```json
{
  "account_name": "you@example.com",
  "unique_method_name": "SimWAM",
  "affiliation": "Your Institution",
  "authors": ["Author One", "Author Two"],
  "description": "Joint video-action world model with RFS GRPO fine-tuning.",
  "method_link": "https://github.com/H-EmbodVis/SimWAM",
  "num_model_parameters": "6B",
  "uses_public_model_pretraining": true,
  "public_model_names": ["Wan-AI/Wan2.2-TI2V-5B"]
}
```

Pass `--allow-incomplete-metadata` to build a clearly marked draft while identity
fields are still blank; the resulting manifest reports `submission_ready: false`.

---

## Command

```bash
export WAYMO_OPEN_DATASET_SRC=/path/to/waymo-open-dataset/src

python experiments/waymo/make_test_submission.py \
  --predictions runs/waymo_test/<model>_<checkpoint>/predictions.jsonl \
  --frames data/waymo/test_sequence_frames_for_submission.json \
  --metadata-json ./my_submission_metadata.json \
  --output-dir runs/waymo_test/<model>_<checkpoint>/submission \
  --num-shards 1
```

## Outputs

```text
<output-dir>/simwam_waymo.binproto-00000-of-00001   # one shard per --num-shards
<output-dir>/submission_manifest.json               # digests + validation report
<output-dir>.tar.gz                                 # flat GNU tar of the shards only
```

## What is validated

- Every official frame appears exactly once, in the official order.
- Each shard round-trips through the official protobuf: `frame_name` unchanged and
  the XY arrays bit-identical after serialization.
- The archive contains only the flat binproto shards, and each archived blob is
  byte-identical to the validated shard on disk.
- `submission_manifest.json` records the SHA-256 of the predictions, the official
  frame index, the loaded `.proto`, its serialized descriptor, every shard and the
  archive.

The official descriptors are read from the `_pb2.py` files' literal serialized
descriptors rather than from generated code, so the packaging step does not depend
on a particular protobuf runtime version. No TensorFlow or full Waymo SDK install
is required.
