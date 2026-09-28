"""Native Waymo XY samples for SimWAM video/action training and action validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as transforms_F

from simwam.utils.logging_config import get_logger

logger = get_logger(__name__)

# Byte-identical to PhysicalAI's NavSimVideoDataset.build_prompt_fixed(...,
# use_dynamic_prompt=False). Keep its trailing space: the cache is content-addressed.
FIXED_PROMPT = (
    "A high-quality, photorealistic dashboard camera view of autonomous driving. "
    "Based on the past 2 seconds videos, "
    "predict and generate the next 4 seconds of realistic driving continuation, "
    "Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
    "clear details, and realistic physics. "
)


def fixed_prompt_cache_path(directory: str | Path, context_len: int = 256, encoder_id: str = "wan22ti2v5b") -> Path:
    digest = hashlib.sha256(FIXED_PROMPT.encode("utf-8")).hexdigest()
    return Path(directory) / f"{digest}.t5_len{context_len}.{encoder_id}.pt"


def validate_image_path(path: str) -> str:
    """Check a manifest image reference is a plain relative POSIX path.

    Manifests ship repo-relative paths (e.g. ``images/training/<scene>/FRONT/000.jpg``)
    that are resolved against ``image_root``. Absolute paths, URL schemes and parent
    directory escapes are rejected so a manifest cannot read outside the data root.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("Image paths must be non-empty strings")
    if "://" in path or path.startswith("//"):
        raise ValueError(f"Image paths must be local relative paths, got a URL: {path!r}")
    if path.startswith("/") or path.startswith("~"):
        raise ValueError(f"Image paths must be relative, got: {path!r}")
    if Path(path).anchor or ".." in Path(path).parts:
        raise ValueError(f"Image paths must stay inside the data root, got: {path!r}")
    return path


class WaymoVideoDataset(torch.utils.data.Dataset):
    """20x2 native actions; five training images or one validation image.

    Kinematics in these JSONLs are already corrected. Do not rescale acceleration
    again. Reference trajectories remain unnormalized, variable-length XY arrays.
    """

    def __init__(
        self,
        dataset_jsonl: str,
        image_root: str | None = None,
        video_size=(512, 480),
        num_frames: int = 5,
        future_action_horizon: int = 20,
        video_frame_mode: str = "current_plus_future",
        camera_layout: str = "front",
        is_training_set: bool = True,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 256,
        text_encoder_id: str = "wan22ti2v5b",
        action_dim: int = 2,
        state_dim: int = 8,
        proprio_dim: int = 8,
        use_dynamic_prompt: bool = False,
        trajectory_mode: str = "absolute",
        normalize_action: bool = True,
        norm_stats_path: str | None = None,
        require_preference_trajectories: bool = False,
    ):
        super().__init__()
        self.dataset_jsonl = str(dataset_jsonl)
        # Manifest image paths are relative and resolved against this root.
        # None keeps them relative to the process working directory.
        self.image_root = None if image_root is None else Path(str(image_root)).expanduser()
        self.video_size = tuple(int(x) for x in video_size)
        if len(self.video_size) != 2 or any(x <= 0 or x % 32 for x in self.video_size):
            raise ValueError("video_size must be [height,width], with both positive multiples of 32")
        self.num_frames = int(num_frames)
        self.video_frame_mode = str(video_frame_mode)
        expected_frames = {"current_plus_future": 5, "current_only": 1}
        if self.video_frame_mode not in expected_frames or self.num_frames != expected_frames[self.video_frame_mode]:
            raise ValueError("Use current_plus_future/5 frames for training or current_only/1 frame for validation")
        self.is_training_set = bool(is_training_set)
        if self.is_training_set and self.video_frame_mode != "current_plus_future":
            raise ValueError("Waymo video/action training requires all five images")
        if camera_layout != "front" or trajectory_mode != "absolute":
            raise ValueError("Waymo supports front images and absolute current-ego XY trajectories")
        if use_dynamic_prompt:
            raise ValueError("Waymo training uses the shared PhysicalAI fixed prompt")
        if int(future_action_horizon) != 20 or int(action_dim) != 2:
            raise ValueError("Native Waymo actions must be 20 XY points: 5 seconds at 4 Hz")
        if int(state_dim) != 8 or int(proprio_dim) != 8:
            raise ValueError("Waymo proprio must be [vx,vy,ax,ay,command(4)]")
        if not text_embedding_cache_dir:
            raise ValueError("text_embedding_cache_dir is required for precomputed context")
        self.future_action_horizon = 20
        self.action_dim, self.state_dim, self.proprio_dim = 2, 8, 8
        self.camera_layout = "front"
        self.trajectory_mode = "absolute"
        self.use_dynamic_prompt = False
        self.evaluation_mode = "action_only" if self.video_frame_mode == "current_only" else "joint"
        self.context_len = int(context_len)
        if self.context_len <= 0:
            raise ValueError("context_len must be positive")
        self.text_encoder_id = str(text_encoder_id)
        self.text_embedding_cache_dir = str(text_embedding_cache_dir)
        self.cache_path = fixed_prompt_cache_path(self.text_embedding_cache_dir, self.context_len, self.text_encoder_id)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Missing shared PhysicalAI fixed-prompt T5 cache: {self.cache_path}")
        self._context = None
        self.normalize_action = bool(normalize_action)
        self.norm_stats_path = norm_stats_path
        self._norm_low = self._norm_range = None
        if self.normalize_action:
            if not norm_stats_path:
                raise ValueError("norm_stats_path is required for action normalization")
            with open(norm_stats_path, encoding="utf-8") as handle:
                stats = json.load(handle)["future_traj_4hz"]
            low = [float(stats[name]["q1"]) for name in ("x", "y")]
            high = [float(stats[name]["q99"]) for name in ("x", "y")]
            if any(not math.isfinite(x) for x in low + high) or any(b <= a for a, b in zip(low, high)):
                raise ValueError("Invalid XY q1/q99 normalization bounds")
            self._norm_low = torch.tensor(low, dtype=torch.float32)
            self._norm_range = torch.tensor(high, dtype=torch.float32) - self._norm_low
        self.require_preference_trajectories = bool(require_preference_trajectories)
        self._load_index()
        self._image_is_pad = torch.zeros(self.num_frames, dtype=torch.bool)
        self._action_is_pad = torch.zeros(self.future_action_horizon, dtype=torch.bool)
        self._proprio_is_pad = torch.zeros(self.future_action_horizon, dtype=torch.bool)
        logger.info(
            "Initialized WaymoVideoDataset samples=%d frames=%d video_size=%s actions=20x2 "
            "mode=%s normalization=%s jsonl=%s",
            len(self), self.num_frames, self.video_size, self.evaluation_mode, norm_stats_path, self.dataset_jsonl,
        )

    @staticmethod
    def build_prompt_fixed() -> str:
        return FIXED_PROMPT

    def _load_index(self):
        try:
            import msgspec
        except ImportError:
            decode = json.loads
        else:
            decode = msgspec.json.decode
        prefixes, prefix_ids, image_names = [], [], []
        prefix_lookup = {}
        trajectories, states, tokens, references = [], [], [], []
        seen = set()
        with open(self.dataset_jsonl, "rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = decode(line)
                    token = str(row["token"])
                    if token in seen:
                        raise ValueError("duplicate token")
                    seen.add(token)
                    action = np.asarray(row["future_traj_4hz"], dtype=np.float32)
                    if action.shape != (20, 2) or not np.isfinite(action).all():
                        raise ValueError("future_traj_4hz must contain 20 finite XY points")
                    if row.get("trajectory_hz", 4) != 4 or row.get("trajectory_horizon_s", 5) != 5:
                        raise ValueError("expected native 5-second, 4-Hz trajectories")
                    if "future_times_4hz" in row and row["future_times_4hz"] != [i / 4 for i in range(1, 21)]:
                        raise ValueError("future_times_4hz must be +0.25,...,+5 seconds without resampling")
                    if row.get("action_dim", 2) != 2 or row.get("has_heading", False):
                        raise ValueError("expected XY-only actions")
                    if "future_valid_mask_4hz" in row and row["future_valid_mask_4hz"] != [1] * 20:
                        raise ValueError("padded or invalid GT points are unsupported")
                    vel = np.asarray(row["ego_velocity"], dtype=np.float32)
                    acc = np.asarray(row["ego_acceleration"], dtype=np.float32)
                    cmd = np.asarray(row["driving_command"], dtype=np.float32)
                    if vel.shape != (2,) or acc.shape != (2,) or cmd.shape != (4,):
                        raise ValueError("invalid ego-state dimensions")
                    state = np.concatenate([vel, acc, cmd])
                    if not np.isfinite(state).all() or not np.isin(cmd, [0, 1]).all() or cmd.sum() != 1:
                        raise ValueError("invalid ego state or command one-hot")
                    paths = [row["front_image"]]
                    if self.video_frame_mode == "current_plus_future":
                        future_paths = row["future_front_images_1s_to_4s"]
                        if not isinstance(future_paths, list) or len(future_paths) != 4:
                            raise ValueError("training requires four future image paths")
                        paths += future_paths
                    row_prefix_ids, row_names = [], []
                    for uri in paths:
                        validate_image_path(uri)
                        base, name = uri.rsplit("/", 1)
                        if not name:
                            raise ValueError("image filename is empty")
                        base += "/"
                        if base not in prefix_lookup:
                            prefix_lookup[base] = len(prefixes)
                            prefixes.append(base)
                        row_prefix_ids.append(prefix_lookup[base])
                        row_names.append(sys.intern(name))
                    refs = row.get("preference_trajectories") or []
                    if self.require_preference_trajectories and not refs:
                        raise ValueError("validation sample lacks scored reference trajectories")
                    if not isinstance(refs, list):
                        raise ValueError("invalid reference trajectories")
                    for ref in refs:
                        x, y = ref["pos_x"], ref["pos_y"]
                        score = ref["preference_score"]
                        if not x or len(x) != len(y) or not all(math.isfinite(v) for v in x + y):
                            raise ValueError("invalid reference XY")
                        if not math.isfinite(score) or not 0 <= score <= 10:
                            raise ValueError("invalid reference score")
                    prefix_ids.append(row_prefix_ids)
                    image_names.append(row_names)
                    trajectories.append(action)
                    states.append(state)
                    tokens.append(token)
                    references.append(refs if not self.is_training_set else None)
                except Exception as exc:
                    raise ValueError(f"{self.dataset_jsonl}:{line_number}: {exc}") from exc
        if not tokens:
            raise ValueError(f"Empty Waymo dataset: {self.dataset_jsonl}")
        # Store image directory IDs + interned filenames instead of N*5 full path strings.
        self._prefixes = tuple(prefixes)
        self._prefix_ids = np.asarray(prefix_ids, dtype=np.int32)
        self._image_names = np.asarray(image_names, dtype=object)
        self._traj = np.stack(trajectories)
        self._states = np.stack(states)
        self._tokens = np.asarray(tokens, dtype=object)
        self._references = references

    def __len__(self) -> int:
        return int(self._traj.shape[0])

    def _image_paths(self, idx: int) -> list[str]:
        return [
            self._prefixes[int(prefix_id)] + name
            for prefix_id, name in zip(self._prefix_ids[idx], self._image_names[idx])
        ]

    def resolve_image_path(self, relative_path: str) -> Path:
        """Resolve a manifest-relative image path against ``image_root``."""
        return Path(relative_path) if self.image_root is None else self.image_root / relative_path

    def _fetch_image(self, relative_path: str) -> Image.Image:
        with Image.open(self.resolve_image_path(relative_path)) as image:
            image.load()
            return image.convert("RGB")

    def _build_video_tensor(self, paths: list[str]) -> torch.Tensor:
        # Match PhysicalAIVideoDataset exactly: PIL resize before float tensor
        # conversion, then RGB mean/std 0.5 and [C,T,H,W].
        height, width = self.video_size
        frames = []
        for path in paths:
            image = self._fetch_image(path).resize((width, height), Image.Resampling.BILINEAR)
            frames.append(transforms_F.to_tensor(image))
        video = torch.stack(frames, dim=0)
        video = transforms_F.normalize(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return video.permute(1, 0, 2, 3).contiguous()

    def norm_traj(self, trajectory: torch.Tensor) -> torch.Tensor:
        if not self.normalize_action:
            return trajectory
        low = self._norm_low.to(device=trajectory.device, dtype=trajectory.dtype)
        span = self._norm_range.to(device=trajectory.device, dtype=trajectory.dtype)
        return 2 * (trajectory - low) / span - 1

    def denorm_traj(self, trajectory: torch.Tensor) -> torch.Tensor:
        if not self.normalize_action:
            return trajectory
        low = self._norm_low.to(device=trajectory.device, dtype=trajectory.dtype)
        span = self._norm_range.to(device=trajectory.device, dtype=trajectory.dtype)
        return (trajectory + 1) * span / 2 + low

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.denorm_traj(action.detach().to(device="cpu", dtype=torch.float32))

    def _get_cached_text_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._context is None:
            payload = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            context, mask = payload["context"], payload["mask"].bool()
            if context.shape != (self.context_len, 4096) or mask.shape != (self.context_len,):
                raise ValueError("Fixed-prompt cache must be [context_len,4096] with a matching mask")
            if not torch.isfinite(context).all():
                raise ValueError("Fixed-prompt cache contains nonfinite values")
            context = context.to(torch.bfloat16).clone()
            context[~mask] = 0
            self._context = (context.contiguous(), torch.ones_like(mask).contiguous())
        return self._context

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video = self._build_video_tensor(self._image_paths(idx))
        action = self.norm_traj(torch.from_numpy(self._traj[idx].copy()))
        state = torch.from_numpy(self._states[idx].copy()).unsqueeze(0).repeat(self.future_action_horizon, 1)
        context, context_mask = self._get_cached_text_context()
        sample = {
            "video": video, "action": action, "state": state, "proprio": state,
            "prompt": FIXED_PROMPT, "context": context, "context_mask": context_mask,
            "image_is_pad": self._image_is_pad.clone(), "action_is_pad": self._action_is_pad.clone(),
            "proprio_is_pad": self._proprio_is_pad.clone(), "token": str(self._tokens[idx]),
        }
        if not self.is_training_set:
            sample["action_raw"] = torch.from_numpy(self._traj[idx].copy())
            sample["preference_trajectories"] = copy.deepcopy(self._references[idx])
        return sample
