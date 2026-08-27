import hashlib
import inspect
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from simwam.utils.logging_config import get_logger
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

TRAJECTORY_MODES = ("absolute", "relative")

logger = get_logger(__name__)


class NavSimVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        navsim_log_path: str,
        sensor_blobs_path: str,
        scene_filter,
        split_config_path: str,
        split_logs_key: str,
        num_frames: int = 9,
        future_action_horizon: Optional[int] = None,
        frame_stride: int = 1,
        video_frame_mode: str = "history_plus_future",
        video_size: list[int] | tuple[int, int] = (384, 640),
        camera_layout: str = "stitched_front",
        is_training_set: bool = True,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        action_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        normalize_action: bool = True,
        pretrained_norm_stats: Optional[str] = None,
        stats_cache_path: Optional[str] = None,
        use_dynamic_prompt: bool = True,
        trajectory_mode: str = "absolute",
    ):
        super().__init__()
        scene_filter = self._resolve_scene_filter(scene_filter)

        self.trajectory_mode = str(trajectory_mode).strip().lower()
        if self.trajectory_mode not in TRAJECTORY_MODES:
            raise ValueError(
                f"Unsupported `trajectory_mode`: {trajectory_mode}. "
                f"Expected one of {list(TRAJECTORY_MODES)}."
            )

        self.navsim_log_path = str(navsim_log_path)
        self.sensor_blobs_path = str(sensor_blobs_path)
        self.split_config_path = str(split_config_path)
        self.split_logs_key = str(split_logs_key)
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.video_frame_mode = str(video_frame_mode).strip().lower()
        self.future_action_horizon = (
            int(future_action_horizon)
            if future_action_horizon is not None
            else (self.num_frames - 1) * self.frame_stride
        )
        self.video_size = tuple(int(v) for v in video_size)
        self._video_size_hw = [int(v) for v in self.video_size]
        self.camera_layout = str(camera_layout).strip().lower()
        self.is_training_set = bool(is_training_set)
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else str(text_embedding_cache_dir)
        )
        self.context_len = int(context_len)
        # Which text encoder produced the cached embeddings; part of the cache
        # filename so different encoders (Wan umT5 vs LTX T5) never collide.
        # Default preserves the original Wan behavior byte-for-byte.
        self.text_encoder_id = str(text_encoder_id)
        # In-memory cache of loaded T5 contexts keyed by prompt, so __getitem__ does not
        # torch.load the cache file on every sample (avoids per-sample fd churn + disk I/O;
        # with use_dynamic_prompt=False there is a single fixed prompt -> one load total).
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.normalize_action = bool(normalize_action)
        self.pretrained_norm_stats = pretrained_norm_stats
        self.stats_cache_path = None if stats_cache_path is None else str(stats_cache_path)
        self.use_dynamic_prompt = bool(use_dynamic_prompt)

        if self.num_frames < 2:
            raise ValueError(f"`num_frames` must be >= 2, got {self.num_frames}")
        if self.frame_stride < 1:
            raise ValueError(f"`frame_stride` must be >= 1, got {self.frame_stride}")
        if (self.num_frames - 1) <= 0:
            raise ValueError("`num_frames` must define at least one transition.")
        
        #TODO and NOTE: since the VAE encoder use the first frame as reference, and the future frames must be 4 times;
        # so current we dont support "history_plus_future" mode which use the history 4 frames as reference
        # To support "history_plus_future" mode, we need to modify the VAE encoder to use the first frame of the video as reference, and the future frames must be 4 times of the history frames.
        # need change the @src/simwam/models/wan22/wan_video_vae.py 1299-1324: WanVideoVAE.encode_video() to support variable number of history frames and future frames, and use the first frame as reference.
        # @zhaozc: 2026-5-20
        
        if self.video_frame_mode not in {"history_plus_future", "current_plus_future"}:
            raise ValueError(
                f"Unsupported `video_frame_mode`: {self.video_frame_mode}. "
                "Expected one of ['history_plus_future', 'current_plus_future']."
            )
        if self.future_action_horizon <= 0:
            raise ValueError(
                f"`future_action_horizon` must be positive, got {self.future_action_horizon}"
            )
        if self.future_action_horizon % (self.num_frames - 1) != 0:
            raise ValueError(
                "`future_action_horizon` must be divisible by video transitions "
                f"({self.num_frames - 1}), got {self.future_action_horizon}"
            )
        if scene_filter.num_future_frames < self.future_action_horizon:
            raise ValueError(
                f"`scene_filter.num_future_frames` ({scene_filter.num_future_frames}) must be >= "
                f"`future_action_horizon` ({self.future_action_horizon})."
            )
        if self.video_frame_mode == "history_plus_future":
            required_future_for_video = self.num_frames * self.frame_stride
        else:
            required_future_for_video = (self.num_frames - 1) * self.frame_stride
        if scene_filter.num_future_frames < required_future_for_video:
            raise ValueError(
                f"`scene_filter.num_future_frames` ({scene_filter.num_future_frames}) must be >= "
                f"video horizon ({required_future_for_video}) for num_frames={self.num_frames}, "
                f"frame_stride={self.frame_stride}, video_frame_mode={self.video_frame_mode}."
            )

        self.scene_filter = self._apply_split_log_filter(
            scene_filter=scene_filter,
            split_config_path=self.split_config_path,
            split_logs_key=self.split_logs_key,
        )
        self.sensor_config = self._build_sensor_config(self.camera_layout)
        self.scene_loader = self._build_scene_loader(
            data_path=Path(self.navsim_log_path),
            sensor_path=Path(self.sensor_blobs_path),
            scene_filter=self.scene_filter,
            sensor_config=self.sensor_config,
        )
        self.tokens = list(self.scene_loader.tokens)
        if not self.tokens:
            raise ValueError("NavSim dataset resolved to zero samples after applying scene filter and log split.")

        sample_dims = self._infer_sample_dims()
        inferred_action_dim = int(sample_dims["action_dim"])
        inferred_state_dim = int(sample_dims["state_dim"])
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
        self._frame_indices = self._video_frame_indices()
        self._image_is_pad = torch.zeros(len(self._frame_indices), dtype=torch.bool)
        self._action_is_pad = torch.zeros(self.future_action_horizon, dtype=torch.bool)
        self._proprio_is_pad = torch.zeros(self.future_action_horizon, dtype=torch.bool)
        if pretrained_norm_stats or stats_cache_path:
            logger.info(
                "NavSimVideoDataset ignores `pretrained_norm_stats`/`stats_cache_path`; "
                "action normalization uses fixed odo bounds."
            )

        logger.info(
            "Initialized NavSimVideoDataset samples=%d num_frames=%d future_action_horizon=%d "
            "frame_stride=%d video_frame_mode=%s camera_layout=%s video_size=%s trajectory_mode=%s",
            len(self.tokens),
            self.num_frames,
            self.future_action_horizon,
            self.frame_stride,
            self.video_frame_mode,
            self.camera_layout,
            self.video_size,
            self.trajectory_mode,
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scene = self.scene_loader.get_scene_from_token(self.tokens[idx])
        video = self._build_video_tensor(scene, self._frame_indices)

        action = self._extract_future_trajectory(scene)
        state_vec, high_cmd_one_hot, speed_mps, acc_mps2 = self._extract_ego_features(scene)
        state = state_vec.unsqueeze(0).repeat(action.shape[0], 1)

        if self.normalize_action:
            if self.trajectory_mode == "relative":
                action = self.norm_odo_relative(action)
            else:
                action = self.norm_odo(action)

        hist_xyh = torch.tensor(
            scene.get_history_trajectory(
                num_trajectory_frames=self.scene_filter.num_history_frames
            ).poses,
            dtype=torch.float32,
        )
        prompt = self.build_prompt_fixed(
            hist_xyh=hist_xyh,
            high_cmd_one_hot=high_cmd_one_hot,
            speed_mps=speed_mps,
            acc_mps2=acc_mps2,
            use_dynamic_prompt=self.use_dynamic_prompt,
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
            "token": self.tokens[idx],
        }

    @staticmethod
    def _resolve_scene_filter(scene_filter: Any) -> SceneFilter:
        if isinstance(scene_filter, (str, Path)):
            scene_filter = OmegaConf.load(str(scene_filter))
        if isinstance(scene_filter, DictConfig):
            if scene_filter.get("_target_") is not None:
                scene_filter = instantiate(scene_filter)
            else:
                scene_filter = SceneFilter(**OmegaConf.to_container(scene_filter, resolve=True))
        elif isinstance(scene_filter, dict):
            scene_filter = SceneFilter(**scene_filter)
        if not isinstance(scene_filter, SceneFilter):
            raise TypeError(f"`scene_filter` must resolve to SceneFilter, got {type(scene_filter)}")
        return scene_filter

    @staticmethod
    def _build_scene_loader(data_path: Path, sensor_path: Path, scene_filter: SceneFilter, sensor_config: SensorConfig):
        """Build SceneLoader across NAVSIM API variants.

        NAVSIM v1 names the sensor root `sensor_blobs_path`; NAVSIM v2 renamed it to
        `original_sensor_path`. DLC images may expose either version through `navsim`.
        """
        params = inspect.signature(SceneLoader.__init__).parameters
        kwargs = {
            "data_path": data_path,
            "scene_filter": scene_filter,
            "sensor_config": sensor_config,
        }
        if "sensor_blobs_path" in params:
            kwargs["sensor_blobs_path"] = sensor_path
        elif "original_sensor_path" in params:
            kwargs["original_sensor_path"] = sensor_path
        else:
            raise TypeError(
                "Unsupported navsim SceneLoader signature: expected `sensor_blobs_path` "
                f"or `original_sensor_path`, got {list(params)}"
            )
        return SceneLoader(**kwargs)

    @staticmethod
    def _load_split_logs(split_config_path: str, split_logs_key: str) -> list[str]:
        split_cfg = OmegaConf.load(split_config_path)
        split_logs = split_cfg.get(split_logs_key)
        if split_logs is None:
            raise KeyError(f"Missing `{split_logs_key}` in {split_config_path}")
        logs = [str(log_name) for log_name in split_logs]
        if not logs:
            raise ValueError(f"`{split_logs_key}` in {split_config_path} is empty.")
        return logs

    @classmethod
    def _apply_split_log_filter(
        cls,
        scene_filter: SceneFilter,
        split_config_path: str,
        split_logs_key: str,
    ) -> SceneFilter:
        split_logs = cls._load_split_logs(split_config_path, split_logs_key)
        if scene_filter.log_names is not None:
            scene_filter.log_names = [
                log_name for log_name in scene_filter.log_names if log_name in split_logs
            ]
        else:
            scene_filter.log_names = split_logs
        return scene_filter

    @staticmethod
    def _build_sensor_config(camera_layout: str) -> SensorConfig:
        if camera_layout == "front":
            return SensorConfig(
                cam_f0=True,
                cam_l0=False,
                cam_l1=False,
                cam_l2=False,
                cam_r0=False,
                cam_r1=False,
                cam_r2=False,
                cam_b0=False,
                lidar_pc=False,
            )
        if camera_layout == "stitched_front":
            return SensorConfig(
                cam_f0=True,
                cam_l0=True,
                cam_l1=False,
                cam_l2=False,
                cam_r0=True,
                cam_r1=False,
                cam_r2=False,
                cam_b0=False,
                lidar_pc=False,
            )
        raise ValueError(
            f"Unsupported `camera_layout`: {camera_layout}. Expected one of ['front', 'stitched_front']."
        )

    def _infer_sample_dims(self) -> dict[str, int]:
        sample_scene = self.scene_loader.get_scene_from_token(self.tokens[0])
        action = self._extract_future_trajectory(sample_scene)
        state, _, _, _ = self._extract_ego_features(sample_scene)
        return {"action_dim": int(action.shape[-1]), "state_dim": int(state.shape[-1])}

    @staticmethod
    def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

    @staticmethod
    def denorm_odo(normalized_trajectory: torch.Tensor) -> torch.Tensor:
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    # Relative (adjacent-frame delta) normalization constants.
    # bias = -min, range = (max - min) per channel, matching the min-max form of `norm_odo`.
    # Fitted from `compute_relative_action_stats` over the full navtrain split (280k+ deltas;
    # distribution converged). Per-channel p99.9 / max:
    #   X: p99.9=7.0  (max 11.3, highway tail), forward-only
    #   Y: p99.9=0.34 (max 0.67), ~symmetric about 0
    #   H: p99.9=0.24 rad (max 0.38), ~symmetric about 0
    # Ranges set to ~p99.9 (+small margin). min-max is linear (no clipping), so the rare >p99.9
    # tail (e.g. highway X) simply maps slightly beyond [-1, 1], which is fine for the flow target.
    _REL_X_BIAS, _REL_X_RANGE = 0.0, 8.0
    _REL_Y_BIAS, _REL_Y_RANGE = 0.4, 0.8
    _REL_H_BIAS, _REL_H_RANGE = 0.3, 0.6

    @classmethod
    def norm_odo_relative(cls, trajectory: torch.Tensor) -> torch.Tensor:
        x = 2 * (trajectory[..., 0:1] + cls._REL_X_BIAS) / cls._REL_X_RANGE - 1
        y = 2 * (trajectory[..., 1:2] + cls._REL_Y_BIAS) / cls._REL_Y_RANGE - 1
        heading = 2 * (trajectory[..., 2:3] + cls._REL_H_BIAS) / cls._REL_H_RANGE - 1
        return torch.cat([x, y, heading], dim=-1)

    @classmethod
    def denorm_odo_relative(cls, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * cls._REL_X_RANGE - cls._REL_X_BIAS
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * cls._REL_Y_RANGE - cls._REL_Y_BIAS
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * cls._REL_H_RANGE - cls._REL_H_BIAS
        return torch.cat([x, y, heading], dim=-1)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        """Map angle(s) to [-pi, pi] (torch-native)."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _to_relative_deltas(abs_poses: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute future poses (all in the current ego frame, shape [N, 3] = x,y,heading)
        into adjacent-frame SE(2) deltas. Each delta is expressed in the *previous* frame's
        heading-aligned frame; the predecessor of step 0 is the current-frame origin (0,0,0),
        so delta[0] == abs_poses[0]. Heading deltas are wrapped to [-pi, pi].
        """
        if abs_poses.ndim != 2 or abs_poses.shape[-1] != 3:
            raise ValueError(f"`abs_poses` must be [N, 3], got shape {tuple(abs_poses.shape)}")
        origin = torch.zeros((1, 3), dtype=abs_poses.dtype, device=abs_poses.device)
        prev = torch.cat([origin, abs_poses[:-1]], dim=0)  # [N, 3]
        cur = abs_poses
        theta_p = prev[..., 2]
        cos_p, sin_p = torch.cos(theta_p), torch.sin(theta_p)
        dx = cur[..., 0] - prev[..., 0]
        dy = cur[..., 1] - prev[..., 1]
        local_x = cos_p * dx + sin_p * dy
        local_y = -sin_p * dx + cos_p * dy
        dh = NavSimVideoDataset._wrap_angle(cur[..., 2] - theta_p)
        return torch.stack([local_x, local_y, dh], dim=-1)

    @staticmethod
    def _relative_deltas_to_absolute(deltas: torch.Tensor) -> torch.Tensor:
        """
        Inverse of `_to_relative_deltas`. Integrate adjacent-frame SE(2) deltas (shape [..., N, 3])
        back into absolute poses in the current ego frame via chained SE(2) composition.
        Vectorized over leading dims; sequential over the N steps.
        """
        if deltas.shape[-1] != 3:
            raise ValueError(f"`deltas` last dim must be 3, got shape {tuple(deltas.shape)}")
        num_steps = deltas.shape[-2]
        batch_shape = deltas.shape[:-2]
        x = torch.zeros(batch_shape, dtype=deltas.dtype, device=deltas.device)
        y = torch.zeros(batch_shape, dtype=deltas.dtype, device=deltas.device)
        h = torch.zeros(batch_shape, dtype=deltas.dtype, device=deltas.device)
        out = []
        for i in range(num_steps):
            dxi, dyi, dhi = deltas[..., i, 0], deltas[..., i, 1], deltas[..., i, 2]
            cos_h, sin_h = torch.cos(h), torch.sin(h)
            x = x + cos_h * dxi - sin_h * dyi
            y = y + sin_h * dxi + cos_h * dyi
            h = NavSimVideoDataset._wrap_angle(h + dhi)
            out.append(torch.stack([x, y, h], dim=-1))
        return torch.stack(out, dim=-2)

    # NOTE: the fuction must have, because in the evaluation stage, we need to denormalize the predicted action to compute the metrics in the original scale.
    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.detach().to(device="cpu", dtype=torch.float32)
        if self.trajectory_mode == "relative":
            if self.normalize_action:
                action = self.denorm_odo_relative(action)
            # integrate deltas back to absolute poses (current-frame convention)
            return self._relative_deltas_to_absolute(action)
        if not self.normalize_action:
            return action
        return self.denorm_odo(action)

    def _video_frame_indices(self) -> list[int]:
        current_idx = self.scene_filter.num_history_frames - 1
        if self.video_frame_mode == "history_plus_future":
            history_indices = list(range(self.scene_filter.num_history_frames))
            future_indices = [
                current_idx + step * self.frame_stride
                for step in range(1, self.num_frames + 1)
            ]
            return history_indices + future_indices

        return [
            current_idx + step * self.frame_stride
            for step in range(self.num_frames)
        ]

    def _build_video_tensor(self, scene, frame_indices: list[int]) -> torch.Tensor:
        frames = []
        for frame_idx in frame_indices:
            image = self._extract_frame_image(scene.frames[frame_idx].cameras)
            tensor = transforms_F.to_tensor(image)
            tensor = transforms_F.resize(
                tensor,
                size=self._video_size_hw,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            frames.append(tensor)
        video = torch.stack(frames, dim=0)
        video = transforms_F.normalize(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return video.permute(1, 0, 2, 3).contiguous()

    def _extract_frame_image(self, cameras) -> Image.Image:
        if self.camera_layout == "front":
            return Image.fromarray(cameras.cam_f0.image.astype(np.uint8))

        left = cameras.cam_l0.image[28:-28, 416:-416]
        front = cameras.cam_f0.image[28:-28]
        right = cameras.cam_r0.image[28:-28, 416:-416]
        stitched = np.concatenate([left, front, right], axis=1).astype(np.uint8)
        return Image.fromarray(stitched)

    def _extract_future_trajectory(self, scene) -> torch.Tensor:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self.future_action_horizon
        )
        abs_poses = torch.tensor(future_trajectory.poses, dtype=torch.float32)
        if self.trajectory_mode == "relative":
            return self._to_relative_deltas(abs_poses)
        return abs_poses

    @staticmethod
    def _one_hot_to_cmd(high_cmd_one_hot: torch.Tensor) -> str:
        labels = [
            'turn left', 'go straight', 'turn right', "unknown"
        ]
        if high_cmd_one_hot.numel() == 0:
            return "unknown"
        cmd_idx = int(high_cmd_one_hot.argmax().item())
        if 0 <= cmd_idx < len(labels):
            return labels[cmd_idx]
        return "unknown"

    def _extract_ego_features(self, scene) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        ego_status = scene.get_agent_input().ego_statuses[-1]
        velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
        acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
        driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)
        state = torch.cat([velocity, acceleration, driving_command], dim=0)
        speed_mps = float(torch.linalg.norm(velocity).item())
        acc_mps2 = float(torch.linalg.norm(acceleration).item())
        return state, driving_command, speed_mps, acc_mps2

    @classmethod
    def build_prompt_fixed(
        cls,
        hist_xyh: torch.Tensor,
        high_cmd_one_hot: torch.Tensor,
        speed_mps: float,
        acc_mps2: float,
        use_dynamic_prompt: bool = True,
    ) -> str:
        # past_seconds = max((hist_xyh.shape[0] - 1) * 0.5, 0.0)
        past_seconds = 2.0
        future_seconds = 4.0
        if not use_dynamic_prompt:
            return (
                "A high-quality, photorealistic dashboard camera view of autonomous driving. "
                f"Based on the past {past_seconds:.0f} seconds videos, "
                f"predict and generate the next {future_seconds:.0f} seconds of realistic driving continuation, "
                "Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
                "clear details, and realistic physics. "
            )

        if speed_mps < 5.0:
            speed_desc = "at low speed"
        elif speed_mps < 15.0:
            speed_desc = "at moderate speed"
        else:
            speed_desc = "at highway speed"

        stability_desc = "steady motion" if acc_mps2 < 0.5 else "gradually changing speed"

        cmd = cls._one_hot_to_cmd(high_cmd_one_hot).lower()
        if "left" in cmd:
            motion_trend, turning_desc = "turning left", "with controlled steering"
        elif "right" in cmd:
            motion_trend, turning_desc = "turning right", "with controlled steering"
        elif "straight" in cmd:
            motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"
        else:
            motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"

        hist_xyh_str = str([[round(float(value), 2) for value in row] for row in hist_xyh.detach().cpu().tolist()])
        prompt = (
            "A high-quality, photorealistic dashboard camera view of autonomous driving. "
            f"Over the past {past_seconds:.0f} seconds, the ego vehicle followed this trajectory: {hist_xyh_str}. "
            f"It is currently moving {speed_desc} with {stability_desc}, and is expected to continue {motion_trend} {turning_desc}. "
            f"Based on the past {past_seconds:.0f} seconds of video, predict and generate the next {future_seconds:.0f} seconds "
            "of realistic driving continuation. Maintain temporal consistency, stable camera perspective, "
            "natural motion flow without jitter or artifacts, clear details, and realistic physics."
        )
        return prompt

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            return cached
        if self.text_embedding_cache_dir is None:
            raise ValueError(
                "`text_embedding_cache_dir` is required for NavSim dataset because SimWAM training "
                "expects precomputed `context/context_mask`."
            )
        cache_dir = Path(self.text_embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing prompt cache: {cache_path}. "
                "Run `python -m simwam.datasets.navsim.cache_prompt_embeddings` first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be [L, D], got {tuple(context.shape)}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be [L], got {tuple(context_mask.shape)}")
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

