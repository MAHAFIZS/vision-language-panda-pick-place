# -*- coding: utf-8 -*-

"""
panda_rl_env_targeted.py

Failure-aware Panda reaching environment.

This environment is based on the original PandaReachEnv, but it adds
targeted curriculum sampling based on failure analysis.

Training target distribution:
    70% hard targets
    30% uniform targets

Hard-target clusters:
    1. Low-X region
    2. Negative-Y region
    3. High-Z region

The low-level control still uses the existing Cartesian impedance
controller from pickandplace.py.

No background MuJoCo thread is used.
Physics is stepped synchronously.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np


# ================================================================
# Project import
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pickandplace import Demo


# ================================================================
# Targeted Panda Reach Environment
# ================================================================

class PandaReachTargetedEnv:

    def __init__(
        self,
        max_steps: int = 120,
        action_scale: float = 0.025,
        physics_steps_per_action: int = 40,
        success_threshold: float = 0.035,
        hard_target_probability: float = 0.70,
        seed: int = 42,
    ) -> None:

        self.rng = np.random.default_rng(
            seed
        )

        self.max_steps = int(
            max_steps
        )

        self.action_scale = float(
            action_scale
        )

        self.physics_steps_per_action = int(
            physics_steps_per_action
        )

        self.success_threshold = float(
            success_threshold
        )

        self.hard_target_probability = float(
            hard_target_probability
        )

        # ========================================================
        # Robot
        # ========================================================

        self.robot = Demo()

        # No background control thread.
        self.robot._hold_running = False

        self.step_count = 0

        # ========================================================
        # Dimensions
        # ========================================================

        self.action_dim = 3

        # Observation:
        #
        # EE xyz        3
        # target xyz    3
        # relative xyz  3
        # joints q      7
        # joints dq     7
        #
        # Total = 23

        self.observation_dim = 23

        # ========================================================
        # Robot workspace safety bounds
        # ========================================================

        self.workspace_low = np.array(
            [
                0.30,
                -0.35,
                0.18,
            ],
            dtype=np.float32,
        )

        self.workspace_high = np.array(
            [
                0.70,
                0.35,
                0.60,
            ],
            dtype=np.float32,
        )

        # ========================================================
        # Full target distribution
        # ========================================================

        self.target_low = np.array(
            [
                0.35,
                -0.25,
                0.25,
            ],
            dtype=np.float32,
        )

        self.target_high = np.array(
            [
                0.65,
                0.25,
                0.50,
            ],
            dtype=np.float32,
        )

        # ========================================================
        # Failure-aware target regions
        # ========================================================

        # Cluster 1:
        # Low-X region.
        #
        # Failure analysis:
        # x = 0.35 → 0.45 had roughly 73.5% success.

        self.low_x_min = 0.35
        self.low_x_max = 0.45

        # Cluster 2:
        # Negative-Y region.
        #
        # Failure analysis:
        # y = -0.25 → -0.083 had roughly 72% success.

        self.negative_y_min = -0.25
        self.negative_y_max = -0.08

        # Cluster 3:
        # High-Z region.
        #
        # Z was the dominant residual error axis.

        self.high_z_min = 0.38
        self.high_z_max = 0.50

        # ========================================================
        # Current target
        # ========================================================

        self.target = np.array(
            [
                0.50,
                0.00,
                0.35,
            ],
            dtype=np.float32,
        )

        self.previous_distance = 0.0

        # For analysis/debugging.
        self.last_target_mode = "uniform"

    # ============================================================
    # Robot state
    # ============================================================

    def _ee_position(
        self,
    ) -> np.ndarray:

        return (
            self.robot.data
            .body("panda_hand")
            .xpos
            .copy()
            .astype(np.float32)
        )

    # ============================================================

    def _joint_positions(
        self,
    ) -> np.ndarray:

        q = []

        for i in range(
            1,
            8,
        ):

            joint = (
                self.robot.data.joint(
                    f"panda_joint{i}"
                )
            )

            q.append(
                float(
                    joint.qpos[0]
                )
            )

        return np.asarray(
            q,
            dtype=np.float32,
        )

    # ============================================================

    def _joint_velocities(
        self,
    ) -> np.ndarray:

        dq = []

        for i in range(
            1,
            8,
        ):

            joint_name = (
                f"panda_joint{i}"
            )

            joint_id = mujoco.mj_name2id(
                self.robot.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:

                raise RuntimeError(
                    f"Joint not found: "
                    f"{joint_name}"
                )

            dof_address = int(
                self.robot.model
                .jnt_dofadr[
                    joint_id
                ]
            )

            velocity = float(
                self.robot.data
                .qvel[
                    dof_address
                ]
            )

            dq.append(
                velocity
            )

        return np.asarray(
            dq,
            dtype=np.float32,
        )

    # ============================================================

    def _observation(
        self,
    ) -> np.ndarray:

        ee = self._ee_position()

        relative = (
            self.target
            - ee
        )

        q = (
            self._joint_positions()
        )

        dq = (
            self._joint_velocities()
        )

        obs = np.concatenate(
            [
                ee,
                self.target,
                relative,
                q,
                dq,
            ]
        ).astype(
            np.float32
        )

        if obs.shape != (
            self.observation_dim,
        ):

            raise RuntimeError(
                "Observation shape error. "
                f"Expected "
                f"{(self.observation_dim,)}, "
                f"got {obs.shape}"
            )

        return obs

    # ============================================================
    # Physics
    # ============================================================

    def _physics_step(
        self,
        num_steps: int,
    ) -> None:

        for _ in range(
            num_steps
        ):

            self.robot.control(
                self.robot.target_pos,
                self.robot.target_quat,
            )

            mujoco.mj_step(
                self.robot.model,
                self.robot.data,
            )

    # ============================================================
    # Uniform target
    # ============================================================

    def _sample_uniform_target(
        self,
    ) -> np.ndarray:

        self.last_target_mode = (
            "uniform"
        )

        return self.rng.uniform(
            low=self.target_low,
            high=self.target_high,
        ).astype(
            np.float32
        )

    # ============================================================
    # Low-X target
    # ============================================================

    def _sample_low_x_target(
        self,
    ) -> np.ndarray:

        self.last_target_mode = (
            "hard_low_x"
        )

        x = self.rng.uniform(
            self.low_x_min,
            self.low_x_max,
        )

        y = self.rng.uniform(
            self.target_low[1],
            self.target_high[1],
        )

        z = self.rng.uniform(
            self.target_low[2],
            self.target_high[2],
        )

        return np.asarray(
            [
                x,
                y,
                z,
            ],
            dtype=np.float32,
        )

    # ============================================================
    # Negative-Y target
    # ============================================================

    def _sample_negative_y_target(
        self,
    ) -> np.ndarray:

        self.last_target_mode = (
            "hard_negative_y"
        )

        x = self.rng.uniform(
            self.target_low[0],
            self.target_high[0],
        )

        y = self.rng.uniform(
            self.negative_y_min,
            self.negative_y_max,
        )

        z = self.rng.uniform(
            self.target_low[2],
            self.target_high[2],
        )

        return np.asarray(
            [
                x,
                y,
                z,
            ],
            dtype=np.float32,
        )

    # ============================================================
    # High-Z target
    # ============================================================

    def _sample_high_z_target(
        self,
    ) -> np.ndarray:

        self.last_target_mode = (
            "hard_high_z"
        )

        x = self.rng.uniform(
            self.target_low[0],
            self.target_high[0],
        )

        y = self.rng.uniform(
            self.target_low[1],
            self.target_high[1],
        )

        z = self.rng.uniform(
            self.high_z_min,
            self.high_z_max,
        )

        return np.asarray(
            [
                x,
                y,
                z,
            ],
            dtype=np.float32,
        )

    # ============================================================
    # Failure-aware target sampler
    # ============================================================

    def _sample_target(
        self,
    ) -> np.ndarray:

        """
        70% hard target by default.
        30% normal uniform target.

        Hard examples are equally divided between:

            low X
            negative Y
            high Z
        """

        use_hard_target = (
            self.rng.random()
            < self.hard_target_probability
        )

        # --------------------------------------------------------
        # Normal target
        # --------------------------------------------------------

        if not use_hard_target:

            return (
                self._sample_uniform_target()
            )

        # --------------------------------------------------------
        # Hard-target cluster
        # --------------------------------------------------------

        cluster = int(
            self.rng.integers(
                0,
                3,
            )
        )

        if cluster == 0:

            return (
                self._sample_low_x_target()
            )

        elif cluster == 1:

            return (
                self._sample_negative_y_target()
            )

        else:

            return (
                self._sample_high_z_target()
            )

    # ============================================================
    # Reset
    # ============================================================

    def reset(
        self,
    ) -> Tuple[
        np.ndarray,
        Dict,
    ]:

        self.step_count = 0

        self.robot.stop_flag.clear()

        self.robot.held_obj = None

        # --------------------------------------------------------
        # Reset complete MuJoCo state
        # --------------------------------------------------------

        mujoco.mj_resetData(
            self.robot.model,
            self.robot.data,
        )

        # --------------------------------------------------------
        # Restore Panda home joints
        # --------------------------------------------------------

        for i in range(
            1,
            8,
        ):

            self.robot.data.joint(
                f"panda_joint{i}"
            ).qpos[0] = (
                self.robot.qpos0[
                    i - 1
                ]
            )

        # --------------------------------------------------------
        # Remove velocity
        # --------------------------------------------------------

        self.robot.data.qvel[:] = (
            0.0
        )

        # --------------------------------------------------------
        # Open gripper
        # --------------------------------------------------------

        self.robot.gripper(
            True
        )

        mujoco.mj_forward(
            self.robot.model,
            self.robot.data,
        )

        # --------------------------------------------------------
        # Controller target = current hand pose
        # --------------------------------------------------------

        hand = (
            self.robot.data.body(
                "panda_hand"
            )
        )

        self.robot.target_pos = (
            hand.xpos.copy()
        )

        self.robot.target_quat = (
            hand.xquat.copy()
        )

        self.robot.home_pos = (
            hand.xpos.copy()
        )

        self.robot.home_quat = (
            hand.xquat.copy()
        )

        # --------------------------------------------------------
        # Stabilize
        # --------------------------------------------------------

        self._physics_step(
            50
        )

        # --------------------------------------------------------
        # Failure-aware target
        # --------------------------------------------------------

        self.target = (
            self._sample_target()
        )

        ee = (
            self._ee_position()
        )

        self.previous_distance = float(
            np.linalg.norm(
                self.target
                - ee
            )
        )

        obs = (
            self._observation()
        )

        info = {
            "target":
                self.target.copy(),

            "ee_position":
                ee.copy(),

            "distance":
                self.previous_distance,

            "target_mode":
                self.last_target_mode,
        }

        return (
            obs,
            info,
        )

    # ============================================================
    # Reward
    # ============================================================

    def _compute_reward(
        self,
        action: np.ndarray,
    ):

        ee = (
            self._ee_position()
        )

        distance = float(
            np.linalg.norm(
                self.target
                - ee
            )
        )

        # --------------------------------------------------------
        # Distance reward
        # --------------------------------------------------------

        distance_reward = (
            -distance
        )

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        progress = (
            self.previous_distance
            - distance
        )

        progress_reward = (
            5.0
            * progress
        )

        # --------------------------------------------------------
        # Action regularization
        # --------------------------------------------------------

        action_penalty = (
            -0.01
            * float(
                np.sum(
                    np.square(
                        action
                    )
                )
            )
        )

        # --------------------------------------------------------
        # Success
        # --------------------------------------------------------

        success = bool(
            distance
            < self.success_threshold
        )

        success_bonus = (
            5.0
            if success
            else 0.0
        )

        reward = (
            distance_reward
            + progress_reward
            + action_penalty
            + success_bonus
        )

        self.previous_distance = (
            distance
        )

        reward_info = {
            "distance":
                distance,

            "distance_reward":
                distance_reward,

            "progress":
                progress,

            "progress_reward":
                progress_reward,

            "action_penalty":
                action_penalty,

            "success_bonus":
                success_bonus,

            "success":
                success,
        }

        return (
            float(reward),
            reward_info,
        )

    # ============================================================
    # Step
    # ============================================================

    def step(
        self,
        action: np.ndarray,
    ):

        self.step_count += 1

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (
            self.action_dim,
        ):

            raise ValueError(
                f"Expected action shape "
                f"{(self.action_dim,)}, "
                f"got {action.shape}"
            )

        # --------------------------------------------------------
        # SAC outputs [-1, 1]
        # --------------------------------------------------------

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        current_ee = (
            self._ee_position()
        )

        # --------------------------------------------------------
        # Cartesian delta
        # --------------------------------------------------------

        delta = (
            action
            * self.action_scale
        )

        desired = (
            current_ee
            + delta
        )

        # --------------------------------------------------------
        # Safety clipping
        # --------------------------------------------------------

        desired = np.clip(
            desired,
            self.workspace_low,
            self.workspace_high,
        )

        # --------------------------------------------------------
        # Existing impedance controller
        # --------------------------------------------------------

        self.robot.target_pos = (
            desired.astype(
                float
            )
        )

        self.robot.target_quat = (
            self.robot.home_quat.copy()
        )

        # --------------------------------------------------------
        # Synchronous simulation
        # --------------------------------------------------------

        self._physics_step(
            self.physics_steps_per_action
        )

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------

        reward, reward_info = (
            self._compute_reward(
                action
            )
        )

        terminated = bool(
            reward_info[
                "success"
            ]
        )

        truncated = bool(
            self.step_count
            >= self.max_steps
        )

        obs = (
            self._observation()
        )

        info = {
            **reward_info,

            "target":
                self.target.copy(),

            "ee_position":
                self._ee_position(),

            "step":
                self.step_count,

            "target_mode":
                self.last_target_mode,
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    # ============================================================
    # Random action
    # ============================================================

    def sample_random_action(
        self,
    ) -> np.ndarray:

        return self.rng.uniform(
            -1.0,
            1.0,
            size=self.action_dim,
        ).astype(
            np.float32
        )

    # ============================================================
    # Close
    # ============================================================

    def close(
        self,
    ) -> None:

        self.robot.run = False

        self.robot._hold_running = (
            False
        )


# ================================================================
# Test target distribution
# ================================================================

def main() -> None:

    print(
        "\n"
        "=======================================\n"
        "Targeted Panda RL Environment Test\n"
        "=======================================\n"
    )

    env = PandaReachTargetedEnv(
        hard_target_probability=0.70,
        seed=123,
    )

    target_counts = {
        "uniform": 0,
        "hard_low_x": 0,
        "hard_negative_y": 0,
        "hard_high_z": 0,
    }

    try:

        # --------------------------------------------------------
        # Test several resets
        # --------------------------------------------------------

        for episode in range(
            1,
            21,
        ):

            obs, info = (
                env.reset()
            )

            mode = (
                info[
                    "target_mode"
                ]
            )

            target_counts[
                mode
            ] += 1

            print(
                f"Episode {episode:02d} | "
                f"mode={mode:16s} | "
                f"target="
                f"{np.round(info['target'], 3)} | "
                f"distance="
                f"{info['distance']:.4f}"
            )

        print(
            "\n"
            "Target sampling counts:"
        )

        for key, value in (
            target_counts.items()
        ):

            print(
                f"{key:18s}: "
                f"{value}"
            )

        print(
            "\nObservation shape:",
            obs.shape,
        )

        print(
            "Action dimension:",
            env.action_dim,
        )

        print(
            "\nEnvironment test completed."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()
