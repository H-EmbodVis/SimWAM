"""Local PhysicalAI training dataset for SimWAM."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset
from simwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_FRAME_ID_RE = re.compile(r"/(\d{8})\.jpg$")
_NUM_FUTURE_FRAMES = 4
_ACTION_HORIZON_10HZ = 40


class PhysicalAIVideoDataset(torch.utils.data.Dataset):
    """SimWAM training dataset over local PhysicalAI image files and metadata."""

    def __init__(
        self,
        dataset_jsonl: str,
        num_frames: int = 5,
        future_action_horizon: int = _ACTION_HORIZON_10HZ,
        video_frame_mode: str = "current_plus_future",
        video_size: list[int] | tuple[int, int] = (384, 672),
        camera_layout: str = "front",
        is_training_set: bool = True,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 256,
        text_encoder_id: str = "wan22ti2v5b",
        action_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        normalize_action: bool = True,
        norm_stats_path: Optional[str] = None,
        use_dynamic_prompt: bool = False,
        trajectory_mode: str = "absolute",
    ):
        super().__init__()
        if str(video_frame_mode).strip().lower() != "current_plus_future":
            raise ValueError(
                "PhysicalAIVideoDataset only supports video_frame_mode='current_plus_future' "
                f"(current front frame + 1Hz future frames), got {video_frame_mode!r}."
            )
        if str(camera_layout).strip().lower() != "front":
            raise ValueError(
                f"PhysicalAIVideoDataset only supports camera_layout='front', got {camera_layout!r}."
            )
        if str(trajectory_mode).strip().lower() != "absolute":
            raise ValueError(
                f"PhysicalAIVideoDataset only supports trajectory_mode='absolute', got {trajectory_mode!r}."
            )
        if bool(use_dynamic_prompt):
            raise ValueError(
                "PhysicalAIVideoDataset only supports the fixed prompt (use_dynamic_prompt=false); "
                "the preprocessed jsonl does not carry the history trajectory for dynamic prompts."
            )
        if int(num_frames) != 1 + _NUM_FUTURE_FRAMES:
            raise ValueError(
                f"`num_frames` must be {1 + _NUM_FUTURE_FRAMES} (current + 4 future @ 1Hz), "
                f"got {num_frames}."
            )
        if int(future_action_horizon) != _ACTION_HORIZON_10HZ:
            raise ValueError(
                f"`future_action_horizon` must be {_ACTION_HORIZON_10HZ} (4s @ 10Hz), "
                f"got {future_action_horizon}."
            )

        self.dataset_jsonl = str(dataset_jsonl)
        self.num_frames = int(num_frames)
        self.future_action_horizon = int(future_action_horizon)
        self.video_frame_mode = "current_plus_future"
        self.video_size = tuple(int(v) for v in video_size)
        self._video_size_hw = [int(v) for v in self.video_size]
        self.camera_layout = "front"
        self.is_training_set = bool(is_training_set)
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else str(text_embedding_cache_dir)
        )
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.use_dynamic_prompt = False
        self.trajectory_mode = "absolute"
        self.normalize_action = bool(normalize_action)
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # ---- normalization stats (q1/q99 per channel, same min-max form as NavSim norm_odo) --
        self._norm_bias = None
        self._norm_range = None
        if self.normalize_action:
            if norm_stats_path is None:
                raise ValueError(
                    "`norm_stats_path` is required when normalize_action=true. Generate it with "
                    "`python scripts/compute_physicalai_traj_stats.py`."
                )
            with open(norm_stats_path) as f:
                stats = json.load(f)
            traj_stats = stats.get("future_traj_10hz", stats)
            bias, rng = [], []
            for channel in ("x", "y", "heading"):
                if channel not in traj_stats:
                    raise ValueError(f"`{channel}` missing in norm stats {norm_stats_path}")
                q1 = float(traj_stats[channel]["q1"])
                q99 = float(traj_stats[channel]["q99"])
                if not (q99 > q1):
                    raise ValueError(f"Invalid q1/q99 for `{channel}`: {q1} / {q99}")
                bias.append(-q1)
                rng.append(q99 - q1)
            self._norm_bias = torch.tensor(bias, dtype=torch.float32)
            self._norm_range = torch.tensor(rng, dtype=torch.float32)

        # ---- load the index (compact numpy + path-template reconstruction) ------------------
        self._load_index()

        inferred_action_dim, inferred_state_dim = 3, 8
        if action_dim is not None and int(action_dim) != inferred_action_dim:
            raise ValueError(
                f"`action_dim` config mismatch: expected {inferred_action_dim}, got {action_dim}."
            )
        if state_dim is not None and int(state_dim) != inferred_state_dim:
            raise ValueError(
                f"`state_dim` config mismatch: expected {inferred_state_dim}, got {state_dim}."
            )
        if proprio_dim is not None and int(proprio_dim) != inferred_state_dim:
            raise ValueError(
                f"`proprio_dim` config mismatch: expected {inferred_state_dim}, got {proprio_dim}."
            )
        self.action_dim = inferred_action_dim
        self.state_dim = inferred_state_dim
        self.proprio_dim = inferred_state_dim
        self._image_is_pad = torch.zeros(self.num_frames, dtype=torch.bool)
        self._action_is_pad = torch.zeros(self.future_action_horizon, dtype=torch.bool)
        self._proprio_is_pad = torch.zeros(self.future_action_horizon, dtype=torch.bool)

        logger.info(
            "Initialized PhysicalAIVideoDataset samples=%d num_frames=%d future_action_horizon=%d "
            "video_size=%s normalize_action=%s norm_stats=%s jsonl=%s",
            len(self),
            self.num_frames,
            self.future_action_horizon,
            self.video_size,
            self.normalize_action,
            norm_stats_path,
            self.dataset_jsonl,
        )

    # -------------------------------------------------------------------------------------------
    # Index loading
    # -------------------------------------------------------------------------------------------
    def _load_index(self) -> None:
        if not os.path.isfile(self.dataset_jsonl):
            raise FileNotFoundError(
                f"PhysicalAI dataset jsonl not found: {self.dataset_jsonl}. "
                "Generate it with `python scripts/preprocess_physicalai_train.py`."
            )
        clip_ids: list[str] = []
        frame_ids: list[list[int]] = []
        trajs: list[np.ndarray] = []
        vels: list[np.ndarray] = []
        accs: list[np.ndarray] = []
        cmds: list[np.ndarray] = []
        tokens: list[str] = []
        prefix: Optional[str] = None

        with open(self.dataset_jsonl, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {self.dataset_jsonl}:{lineno + 1}: {e}") from e

                paths = [rec["front_image"]] + list(rec["future_front_images_1s_to_4s"])
                if len(paths) != self.num_frames:
                    raise ValueError(
                        f"{self.dataset_jsonl}:{lineno + 1}: expected {self.num_frames} image "
                        f"paths, got {len(paths)}"
                    )
                fids = []
                for p in paths:
                    m = _FRAME_ID_RE.search(p)
                    if m is None:
                        raise ValueError(
                            f"{self.dataset_jsonl}:{lineno + 1}: cannot parse frame id from {p}"
                        )
                    fids.append(int(m.group(1)))
                    if prefix is None:
                        prefix = p[: m.start()].rsplit("/", 2)[0] + "/"
                    expected = (
                        f"{prefix}{rec['clip_id']}.camera_front_wide_120fov.mp4.frames/all_frames/"
                    )
                    if p[: m.start() + 1] != expected:
                        raise ValueError(
                            f"{self.dataset_jsonl}:{lineno + 1}: image path does not follow the "
                            f"expected template ({expected}...): {p}"
                        )

                traj = np.asarray(rec["future_traj_10hz"], dtype=np.float32)
                if traj.shape != (_ACTION_HORIZON_10HZ, 3):
                    raise ValueError(
                        f"{self.dataset_jsonl}:{lineno + 1}: future_traj_10hz must be "
                        f"[{_ACTION_HORIZON_10HZ}, 3], got {traj.shape}"
                    )
                vel = np.asarray(rec["ego_velocity"], dtype=np.float32)
                acc = np.asarray(rec["ego_acceleration"], dtype=np.float32)
                cmd = np.asarray(rec["driving_command"], dtype=np.float32)
                if vel.shape != (2,) or acc.shape != (2,) or cmd.shape != (4,):
                    raise ValueError(
                        f"{self.dataset_jsonl}:{lineno + 1}: bad ego fields "
                        f"vel={vel.shape} acc={acc.shape} cmd={cmd.shape}"
                    )

                clip_ids.append(rec["clip_id"])
                frame_ids.append(fids)
                trajs.append(traj)
                vels.append(vel)
                accs.append(acc)
                cmds.append(cmd)
                tokens.append(str(rec.get("token") or f"{rec['clip_id']}__{fids[0]:08d}"))

        if not clip_ids:
            raise ValueError(f"PhysicalAI dataset jsonl is empty: {self.dataset_jsonl}")
        assert prefix is not None

        self._path_prefix = prefix
        self._clip_ids = np.array(clip_ids, dtype=object)
        self._frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self._traj = np.stack(trajs, axis=0)
        self._vel = np.stack(vels, axis=0)
        self._acc = np.stack(accs, axis=0)
        self._cmd = np.stack(cmds, axis=0)
        self._tokens = np.array(tokens, dtype=object)

    def _image_paths(self, idx: int) -> list[str]:
        cid = str(self._clip_ids[idx])
        base = f"{self._path_prefix}{cid}.camera_front_wide_120fov.mp4.frames/all_frames/"
        return [f"{base}{fid:08d}.jpg" for fid in self._frame_ids[idx]]

    # -------------------------------------------------------------------------------------------
    # Normalization (same min-max form as NavSimVideoDataset.norm_odo, bounds = q1/q99)
    # -------------------------------------------------------------------------------------------
    def norm_traj(self, trajectory: torch.Tensor) -> torch.Tensor:
        return 2 * (trajectory + self._norm_bias) / self._norm_range - 1

    def denorm_traj(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        return (normalized_trajectory + 1) / 2 * self._norm_range - self._norm_bias

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Inverse of the training-time action normalization (used by the trainer's eval)."""
        action = action.detach().to(device="cpu", dtype=torch.float32)
        if not self.normalize_action:
            return action
        return self.denorm_traj(action)

    # -------------------------------------------------------------------------------------------
    # Sample building
    # -------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._frame_ids.shape[0])

    def _fetch_image(self, path: str) -> Image.Image:
        image_path = Path(path).expanduser()
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _build_video_tensor(self, paths: list[str]) -> torch.Tensor:
        frames = []
        for path in paths:
            image = self._fetch_image(path)
            # Resize on the PIL side first: the raw jpgs are 1920x1080 and resizing the float
            # tensor at full resolution would dominate per-sample CPU time.
            image = image.resize((self._video_size_hw[1], self._video_size_hw[0]), Image.BILINEAR)
            frames.append(transforms_F.to_tensor(image))
        video = torch.stack(frames, dim=0)
        video = transforms_F.normalize(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return video.permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video = self._build_video_tensor(self._image_paths(idx))

        action = torch.from_numpy(self._traj[idx].copy())
        if self.normalize_action:
            action = self.norm_traj(action)

        state_vec = torch.from_numpy(
            np.concatenate([self._vel[idx], self._acc[idx], self._cmd[idx]]).astype(np.float32)
        )
        state = state_vec.unsqueeze(0).repeat(self.future_action_horizon, 1)

        prompt = NavSimVideoDataset.build_prompt_fixed(
            hist_xyh=torch.zeros(1, 3),
            high_cmd_one_hot=torch.from_numpy(self._cmd[idx]),
            speed_mps=float(np.linalg.norm(self._vel[idx])),
            acc_mps2=float(np.linalg.norm(self._acc[idx])),
            use_dynamic_prompt=False,
        )
        context, context_mask = self._get_cached_text_context(prompt)

        return {
            "video": video,
            "action": action,
            "state": state,
            "proprio": state,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": self._image_is_pad.clone(),
            "action_is_pad": self._action_is_pad.clone(),
            "proprio_is_pad": self._proprio_is_pad.clone(),
            "token": str(self._tokens[idx]),
        }

    # -------------------------------------------------------------------------------------------
    # Fixed-prompt T5 context (same cache layout/content as NavSim's fixed prompt)
    # -------------------------------------------------------------------------------------------
    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            return cached
        if self.text_embedding_cache_dir is None:
            raise ValueError(
                "`text_embedding_cache_dir` is required because SimWAM training expects "
                "precomputed `context/context_mask`."
            )
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = (
            Path(self.text_embedding_cache_dir)
            / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing fixed-prompt T5 cache: {cache_path}. "
                "It is shared with NavSim's fixed prompt -- run "
                "the precomputed NavSim fixed-prompt embedding before training."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2 or context_mask.ndim != 1:
            raise ValueError(f"Cached context/mask must be [L, D]/[L], got {context.shape}/{context_mask.shape}")
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached prompt length mismatch: expected {self.context_len}, "
                f"got context={context.shape[0]} mask={context_mask.shape[0]}"
            )
        context = context.to(dtype=torch.bfloat16)
        context[~context_mask] = 0
        context_mask = torch.ones_like(context_mask)
        result = (context.contiguous(), context_mask.contiguous())
        self._text_context_cache[prompt] = result
        return result
