# -*- coding: utf-8 -*-

"""
panda_rl_env.py

Stage 1 deep-RL environment for the
Vision-Language Panda Pick-and-Place project.

Task:
    Move the Franka Panda end-effector to a random Cartesian target.

Observation:
    3 EE position
    3 target position
    3 target-relative position
    7 joint positions
    7 joint velocities

    Total: 23

Action:
    [dx, dy, dz] in [-1, 1]

Important:
    MuJoCo is stepped synchronously in this environment.
    No background physics thread is used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np


# ------------------------------------------------------------
# Project import path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pickandplace import Demo


class PandaReachEnv:

    def __init__(
        self,
        max_steps: int = 120,
        action_scale: float = 0.025,
        physics_steps_per_action: int = 40,
        success_threshold: float = 0.035,
        seed: int = 42,
    ):

        self.rng = np.random.default_rng(seed)

        self.max_steps = max_steps
        self.action_scale = action_scale
        self.physics_steps_per_action = physics_steps_per_action
        self.success_threshold = success_threshold

        # Existing Panda model/controller
        self.robot = Demo()

        # IMPORTANT:
        # Do NOT start robot._hold_loop().
        # RL environment owns mj_step().
        self.robot._hold_running = False

        self.step_count = 0

        self.action_dim = 3
        self.observation_dim = 23

        # Conservative Panda workspace
        self.workspace_low = np.array(
            [0.30, -0.35, 0.18],
            dtype=np.float32,
        )

        self.workspace_high = np.array(
            [0.70, 0.35, 0.60],
            dtype=np.float32,
        )

        # Reach targets
        self.target_low = np.array(
            [0.35, -0.25, 0.25],
            dtype=np.float32,
        )

        self.target_high = np.array(
            [0.65, 0.25, 0.50],
            dtype=np.float32,
        )

        self.target = np.zeros(
            3,
            dtype=np.float32,
        )

        self.previous_distance = 0.0

    # ============================================================
    # State
    # ============================================================

    def _ee_position(self):

        return (
            self.robot.data
            .body("panda_hand")
            .xpos.copy()
            .astype(np.float32)
        )

    def _joint_positions(self):

        q = []

        for i in range(1, 8):

            joint = self.robot.data.joint(
                f"panda_joint{i}"
            )

            q.append(
                float(joint.qpos[0])
            )

        return np.asarray(
            q,
            dtype=np.float32,
        )

    def _joint_velocities(self):

        dq = []

        for i in range(1, 8):

            joint_name = f"panda_joint{i}"

            jid = mujoco.mj_name2id(
                self.robot.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if jid < 0:
                raise RuntimeError(
                    f"Joint not found: {joint_name}"
                )

            dofadr = int(
                self.robot.model.jnt_dofadr[jid]
            )

            dq.append(
                float(
                    self.robot.data.qvel[dofadr]
                )
            )

        return np.asarray(
            dq,
            dtype=np.float32,
        )

    def _observation(self):

        ee = self._ee_position()

        relative = (
            self.target - ee
        )

        q = self._joint_positions()
        dq = self._joint_velocities()

        obs = np.concatenate(
            [
                ee,
                self.target,
                relative,
                q,
                dq,
            ]
        ).astype(np.float32)

        if obs.shape != (23,):

            raise RuntimeError(
                f"Expected observation (23,), "
                f"got {obs.shape}"
            )

        return obs

    # ============================================================
    # Physics
    # ============================================================

    def _physics_step(
        self,
        num_steps: int,
    ):
        """
        Run the existing Cartesian impedance controller and
        MuJoCo synchronously.

        This replaces the background _hold_loop().
        """

        for _ in range(num_steps):

            self.robot.control(
                self.robot.target_pos,
                self.robot.target_quat,
            )

            mujoco.mj_step(
                self.robot.model,
                self.robot.data,
            )

    # ============================================================
    # Reset
    # ============================================================

    def reset(
        self,
    ) -> Tuple[np.ndarray, Dict]:

        self.step_count = 0

        self.robot.stop_flag.clear()
        self.robot.held_obj = None

        # Reset the ENTIRE MuJoCo simulation state first.
        mujoco.mj_resetData(
            self.robot.model,
            self.robot.data,
        )

        # Set Panda arm home pose.
        for i in range(1, 8):

            self.robot.data.joint(
                f"panda_joint{i}"
            ).qpos[0] = (
                self.robot.qpos0[i - 1]
            )

        # Start with zero velocity.
        self.robot.data.qvel[:] = 0.0

        # Open gripper.
        self.robot.gripper(True)

        mujoco.mj_forward(
            self.robot.model,
            self.robot.data,
        )

        hand = self.robot.data.body(
            "panda_hand"
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

        # Allow controller to stabilize.
        self._physics_step(50)

        # Random target
        self.target = self.rng.uniform(
            self.target_low,
            self.target_high,
        ).astype(np.float32)

        ee = self._ee_position()

        self.previous_distance = float(
            np.linalg.norm(
                self.target - ee
            )
        )

        obs = self._observation()

        info = {
            "target": self.target.copy(),
            "ee_position": ee.copy(),
            "distance": self.previous_distance,
        }

        return obs, info

    # ============================================================
    # Reward
    # ============================================================

    def _compute_reward(
        self,
        action,
    ):

        ee = self._ee_position()

        distance = float(
            np.linalg.norm(
                self.target - ee
            )
        )

        # Distance objective
        distance_reward = -distance

        # Improvement since last RL step
        progress = (
            self.previous_distance
            - distance
        )

        progress_reward = (
            5.0 * progress
        )

        # Discourage unnecessarily large Cartesian commands
        action_penalty = (
            -0.01
            * float(
                np.sum(
                    np.square(action)
                )
            )
        )

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

        self.previous_distance = distance

        info = {
            "distance": distance,
            "distance_reward": distance_reward,
            "progress": progress,
            "progress_reward": progress_reward,
            "action_penalty": action_penalty,
            "success_bonus": success_bonus,
            "success": success,
        }

        return float(reward), info

    # ============================================================
    # Step
    # ============================================================

    def step(
        self,
        action,
    ):

        self.step_count += 1

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (3,):

            raise ValueError(
                f"Expected action shape (3,), "
                f"got {action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        current_ee = self._ee_position()

        delta = (
            action
            * self.action_scale
        )

        desired = (
            current_ee
            + delta
        )

        desired = np.clip(
            desired,
            self.workspace_low,
            self.workspace_high,
        )

        self.robot.target_pos = (
            desired.astype(float)
        )

        self.robot.target_quat = (
            self.robot.home_quat.copy()
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # The environment itself now steps MuJoCo.
        # --------------------------------------------------------

        self._physics_step(
            self.physics_steps_per_action
        )

        reward, reward_info = (
            self._compute_reward(action)
        )

        terminated = bool(
            reward_info["success"]
        )

        truncated = bool(
            self.step_count
            >= self.max_steps
        )

        obs = self._observation()

        info = {
            **reward_info,
            "target": self.target.copy(),
            "ee_position": self._ee_position(),
            "step": self.step_count,
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    # ============================================================
    # Random policy
    # ============================================================

    def sample_random_action(self):

        return self.rng.uniform(
            -1.0,
            1.0,
            size=3,
        ).astype(np.float32)

    def close(self):

        # Nothing threaded to terminate now.
        self.robot.run = False
        self.robot._hold_running = False


# ================================================================
# Test
# ================================================================

def main():

    print(
        "\n--- Panda RL Environment Test ---\n"
    )

    env = PandaReachEnv()

    try:

        print("Creating/resetting environment...")

        obs, info = env.reset()

        print(
            "Observation dimension:",
            obs.shape,
        )

        print(
            "Action dimension:",
            env.action_dim,
        )

        print(
            "Target:",
            np.round(
                info["target"],
                3,
            ),
        )

        print(
            "Initial EE:",
            np.round(
                info["ee_position"],
                3,
            ),
        )

        print(
            "Initial distance:",
            round(
                info["distance"],
                4,
            ),
        )

        print(
            "\nRunning random actions...\n"
        )

        total_reward = 0.0

        for step_index in range(20):

            action = (
                env.sample_random_action()
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            print(
                f"step={step_index + 1:03d} "
                f"action={np.round(action, 2)} "
                f"reward={reward:+.4f} "
                f"distance={info['distance']:.4f} "
                f"progress={info['progress']:+.4f} "
                f"success={info['success']}"
            )

            if terminated:

                print(
                    "\nSUCCESS: target reached."
                )

                break

            if truncated:

                print(
                    "\nEpisode reached maximum steps."
                )

                break

        print(
            "\nTotal reward:",
            round(total_reward, 4),
        )

        print(
            "Final EE:",
            np.round(
                info["ee_position"],
                3,
            ),
        )

        print(
            "Target:",
            np.round(
                info["target"],
                3,
            ),
        )

        print(
            "Final distance:",
            round(
                info["distance"],
                4,
            ),
        )

        print(
            "\nEnvironment test completed successfully."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()