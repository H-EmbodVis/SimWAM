"""Label-free Waymo test inputs, reusing the training image/prompt/normalizer code."""
from __future__ import annotations

import json

import numpy as np
import torch

from .waymo_dataset import FIXED_PROMPT, WaymoVideoDataset, validate_image_path


class WaymoTestDataset(WaymoVideoDataset):
    def __init__(self, **kwargs):
        defaults = dict(num_frames=1, video_frame_mode="current_only", is_training_set=False,
                        require_preference_trajectories=False)
        defaults.update(kwargs)
        if defaults["num_frames"] != 1 or defaults["video_frame_mode"] != "current_only":
            raise ValueError("Waymo test inference requires one current FRONT image")
        if defaults["is_training_set"] or defaults["require_preference_trajectories"]:
            raise ValueError("Waymo test has no future GT or rated references")
        if not defaults.get("normalize_action", True):
            raise ValueError("Use the checkpoint's IL normalization to recover meter-valued predictions")
        super().__init__(**defaults)
        self.evaluation_mode = "prediction_only"

    def _load_index(self):
        tokens, images, states, metadata = [], [], [], []
        seen = set()
        with open(self.dataset_jsonl, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                token = row["token"]
                if not isinstance(token, str) or not token or token in seen:
                    raise ValueError(f"{self.dataset_jsonl}:{line_number}: invalid/duplicate token")
                if row.get("split") != "test" or row.get("frame_name") != token:
                    raise ValueError("Test split and exact official frame_name are required")
                if any(key in row for key in ("future_traj_4hz", "future_traj_10hz", "preference_trajectories")):
                    raise ValueError("Test input must not contain ground-truth or preference labels")
                if row.get("prediction_times_4hz") != [i/4 for i in range(1, 21)]:
                    raise ValueError("Expected a 5s/4Hz native prediction grid")
                if row.get("action_dim") != 2 or row.get("has_heading") is not False:
                    raise ValueError("Waymo test predicts XY only")
                validate_image_path(row["front_image"])
                velocity = np.asarray(row["ego_velocity"], dtype=np.float32)
                acceleration = np.asarray(row["ego_acceleration"], dtype=np.float32)
                command = np.asarray(row["driving_command"], dtype=np.float32)
                if velocity.shape != (2,) or acceleration.shape != (2,) or command.shape != (4,):
                    raise ValueError("Expected vx/vy/ax/ay + command(4)")
                state = np.concatenate([velocity, acceleration, command])
                if not np.isfinite(state).all() or not np.isin(command, [0, 1]).all() or command.sum() != 1:
                    raise ValueError("Invalid test ego state or command")
                tokens.append(token)
                images.append(row["front_image"])
                states.append(state)
                metadata.append({key: row[key] for key in ("clip_id", "current_frame_id", "anchor_timestamp_us")})
                seen.add(token)
        if not tokens:
            raise ValueError("Waymo test dataset is empty")
        self._tokens = np.asarray(tokens, dtype=object)
        self._test_images = np.asarray(images, dtype=object)
        self._states = np.stack(states)
        self._test_metadata = metadata
        self._references = [None] * len(tokens)

    def __len__(self):
        return len(self._tokens)

    def _image_paths(self, index):
        return [str(self._test_images[index])]

    def __getitem__(self, index):
        video = self._build_video_tensor(self._image_paths(index))
        state = torch.from_numpy(self._states[index].copy()).unsqueeze(0).repeat(20, 1)
        context, mask = self._get_cached_text_context()
        token = str(self._tokens[index])
        return {
            "video": video, "state": state, "proprio": state,
            "prompt": FIXED_PROMPT, "context": context, "context_mask": mask,
            "image_is_pad": self._image_is_pad.clone(),
            "token": token, "frame_name": token,
            "front_image": str(self._test_images[index]),
            **self._test_metadata[index],
        }
