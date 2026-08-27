import os
from pathlib import Path

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class NpyTrajectoryAgent(AbstractAgent):
    """Agent that loads precomputed trajectories from token-named .npy files."""

    requires_scene = True

    def __init__(
        self,
        pred_actions_path: str,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        self._pred_actions_path = Path(os.path.expandvars(pred_actions_path)).expanduser()

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if not self._pred_actions_path.exists():
            raise FileNotFoundError(f"Prediction directory does not exist: {self._pred_actions_path}")
        if not self._pred_actions_path.is_dir():
            raise NotADirectoryError(f"Prediction path is not a directory: {self._pred_actions_path}")

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        token = scene.scene_metadata.initial_token
        trajectory_path = self._pred_actions_path / f"{token}.npy"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Missing trajectory file for token {token}: {trajectory_path}")

        poses = np.load(trajectory_path)
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim == 3 and poses.shape[0] == 1:
            poses = poses[0]
        if poses.ndim != 2:
            raise ValueError(
                f"Expected trajectory array with 2 dims [(num_poses), 3], got shape {tuple(poses.shape)} "
                f"for token {token}"
            )
        if poses.shape[1] != 3:
            raise ValueError(f"Expected last dim == 3 for token {token}, got shape {tuple(poses.shape)}")
        if poses.shape[0] != self._trajectory_sampling.num_poses:
            raise ValueError(
                f"Expected {self._trajectory_sampling.num_poses} poses for token {token}, "
                f"got {poses.shape[0]} in {trajectory_path}"
            )

        return Trajectory(poses=poses, trajectory_sampling=self._trajectory_sampling)
