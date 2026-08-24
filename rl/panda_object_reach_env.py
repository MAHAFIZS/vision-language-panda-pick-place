# -*- coding: utf-8 -*-

"""
panda_object_reach_env.py

Stage 2 RL environment:
    Reach a safe pre-grasp point above a real MuJoCo object.

Why pre-grasp instead of the object's center?
    The object center is close to the table surface, while the Panda
    end-effector has a safe lower workspace limit.

Therefore the RL goal is:

    pregrasp_goal = object_position + [0, 0, pregrasp_height]

The SAC action is:

    [dx, dy, dz] in [-1, 1]

Observation:
    EE xyz                  3
    object xyz              3
    pregrasp goal xyz       3
    goal - EE relative xyz  3
    joint positions         7
    joint velocities        7

Total:
    26 dimensions

Pipeline:
    SAC
      ↓
    Cartesian delta
      ↓
    Panda impedance controller
      ↓
    MuJoCo physics
      ↓
    reward from distance to real-object pre-grasp point
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
# Environment
# ================================================================

class PandaObjectReachEnv:

    def __init__(
        self,
        object_name: str = "red_box",
        max_steps: int = 120,
        action_scale: float = 0.025,
        physics_steps_per_action: int = 40,
        success_threshold: float = 0.040,
        pregrasp_height: float = 0.15,
        seed: int = 42,
    ) -> None:

        self.rng = np.random.default_rng(seed)

        self.object_name = object_name

        self.max_steps = int(max_steps)

        self.action_scale = float(
            action_scale
        )

        self.physics_steps_per_action = int(
            physics_steps_per_action
        )

        self.success_threshold = float(
            success_threshold
        )

        self.pregrasp_height = float(
            pregrasp_height
        )

        # ========================================================
        # Robot
        # ========================================================

        self.robot = Demo()

        # No background MuJoCo thread.
        self.robot._hold_running = False

        self.step_count = 0

        # ========================================================
        # Dimensions
        # ========================================================

        self.action_dim = 3

        # EE xyz                  3
        # object xyz              3
        # goal xyz                3
        # goal-relative xyz       3
        # joint positions         7
        # joint velocities        7
        #
        # Total = 26

        self.observation_dim = 26

        # ========================================================
        # Workspace safety bounds
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
        # Object randomization region
        # ========================================================

        self.object_xy_low = np.array(
            [
                0.40,
                -0.20,
            ],
            dtype=np.float32,
        )

        self.object_xy_high = np.array(
            [
                0.60,
                0.20,
            ],
            dtype=np.float32,
        )

        self.previous_distance = 0.0

        # ========================================================
        # Verify object exists
        # ========================================================

        self.object_body_id = mujoco.mj_name2id(
            self.robot.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.object_name,
        )

        if self.object_body_id < 0:

            raise RuntimeError(
                f"Object body not found: "
                f"{self.object_name}"
            )

        # ========================================================
        # Find free joint once
        # ========================================================

        self.object_free_joint_id = (
            self._find_object_free_joint()
        )

    # ============================================================
    # State helpers
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

    def _object_position(
        self,
    ) -> np.ndarray:

        return (
            self.robot.data
            .body(self.object_name)
            .xpos
            .copy()
            .astype(np.float32)
        )

    # ============================================================

    def _goal_position(
        self,
    ) -> np.ndarray:

        """
        Safe pre-grasp target above the actual object.
        """

        obj = self._object_position()

        goal = obj + np.array(
            [
                0.0,
                0.0,
                self.pregrasp_height,
            ],
            dtype=np.float32,
        )

        # Safety clip goal into allowed robot workspace.
        goal = np.clip(
            goal,
            self.workspace_low,
            self.workspace_high,
        )

        return goal.astype(
            np.float32
        )

    # ============================================================

    def _joint_positions(
        self,
    ) -> np.ndarray:

        values = []

        for i in range(
            1,
            8,
        ):

            joint = self.robot.data.joint(
                f"panda_joint{i}"
            )

            values.append(
                float(
                    joint.qpos[0]
                )
            )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    # ============================================================

    def _joint_velocities(
        self,
    ) -> np.ndarray:

        values = []

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

            values.append(
                float(
                    self.robot.data
                    .qvel[
                        dof_address
                    ]
                )
            )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    # ============================================================

    def _observation(
        self,
    ) -> np.ndarray:

        ee = self._ee_position()

        obj = self._object_position()

        goal = self._goal_position()

        relative = (
            goal - ee
        )

        q = self._joint_positions()

        dq = self._joint_velocities()

        observation = np.concatenate(
            [
                ee,
                obj,
                goal,
                relative,
                q,
                dq,
            ]
        ).astype(
            np.float32
        )

        if observation.shape != (
            self.observation_dim,
        ):

            raise RuntimeError(
                f"Expected observation shape "
                f"{(self.observation_dim,)}, "
                f"got {observation.shape}"
            )

        return observation

    # ============================================================
    # Physics
    # ============================================================

    def _physics_step(
        self,
        steps: int,
    ) -> None:

        for _ in range(
            steps
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
    # Object free joint
    # ============================================================

    def _find_object_free_joint(
        self,
    ):

        body_id = self.object_body_id

        joint_count = int(
            self.robot.model.body_jntnum[
                body_id
            ]
        )

        joint_start = int(
            self.robot.model.body_jntadr[
                body_id
            ]
        )

        for offset in range(
            joint_count
        ):

            joint_id = (
                joint_start
                + offset
            )

            joint_type = (
                self.robot.model.jnt_type[
                    joint_id
                ]
            )

            if (
                joint_type
                == mujoco.mjtJoint.mjJNT_FREE
            ):

                return joint_id

        return None

    # ============================================================
    # Randomize real object position
    # ============================================================

    def _randomize_object_position(
        self,
    ) -> None:

        if self.object_free_joint_id is None:

            print(
                f"[warning] "
                f"{self.object_name} has no free joint. "
                "Object will stay at XML position."
            )

            return

        joint_id = (
            self.object_free_joint_id
        )

        qpos_address = int(
            self.robot.model
            .jnt_qposadr[
                joint_id
            ]
        )

        # Preserve object's original/reset Z.
        current_z = float(
            self.robot.data
            .qpos[
                qpos_address + 2
            ]
        )

        xy = self.rng.uniform(
            low=self.object_xy_low,
            high=self.object_xy_high,
        )

        # --------------------------------------------------------
        # Set object translation
        # --------------------------------------------------------

        self.robot.data.qpos[
            qpos_address
        ] = float(
            xy[0]
        )

        self.robot.data.qpos[
            qpos_address + 1
        ] = float(
            xy[1]
        )

        self.robot.data.qpos[
            qpos_address + 2
        ] = current_z

        # --------------------------------------------------------
        # Reset object orientation
        #
        # Free joint:
        # x y z qw qx qy qz
        # --------------------------------------------------------

        self.robot.data.qpos[
            qpos_address + 3
        ] = 1.0

        self.robot.data.qpos[
            qpos_address + 4
        ] = 0.0

        self.robot.data.qpos[
            qpos_address + 5
        ] = 0.0

        self.robot.data.qpos[
            qpos_address + 6
        ] = 0.0

        # --------------------------------------------------------
        # Zero free-joint velocity
        # --------------------------------------------------------

        dof_address = int(
            self.robot.model
            .jnt_dofadr[
                joint_id
            ]
        )

        self.robot.data.qvel[
            dof_address:
            dof_address + 6
        ] = 0.0

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
        # Reset full MuJoCo state
        # --------------------------------------------------------

        mujoco.mj_resetData(
            self.robot.model,
            self.robot.data,
        )

        # --------------------------------------------------------
        # Restore Panda home pose
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
        # Randomize object
        # --------------------------------------------------------

        self._randomize_object_position()

        mujoco.mj_forward(
            self.robot.model,
            self.robot.data,
        )

        # --------------------------------------------------------
        # Controller target = current EE pose
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Stabilize
        # --------------------------------------------------------

        self._physics_step(
            50
        )

        # --------------------------------------------------------
        # Compute initial state
        # --------------------------------------------------------

        ee = self._ee_position()

        obj = self._object_position()

        goal = self._goal_position()

        self.previous_distance = float(
            np.linalg.norm(
                goal - ee
            )
        )

        observation = (
            self._observation()
        )

        info = {
            "ee_position":
                ee.copy(),

            "object_position":
                obj.copy(),

            "goal_position":
                goal.copy(),

            "distance":
                self.previous_distance,

            "object_name":
                self.object_name,
        }

        return (
            observation,
            info,
        )

    # ============================================================
    # Reward
    # ============================================================

    def _compute_reward(
        self,
        action: np.ndarray,
    ):

        ee = self._ee_position()

        goal = self._goal_position()

        distance = float(
            np.linalg.norm(
                goal - ee
            )
        )

        # --------------------------------------------------------
        # Dense distance reward
        # --------------------------------------------------------

        distance_reward = (
            -distance
        )

        # --------------------------------------------------------
        # Progress reward
        # --------------------------------------------------------

        progress = (
            self.previous_distance
            - distance
        )

        progress_reward = (
            6.0
            * progress
        )

        # --------------------------------------------------------
        # Action penalty
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

        # SAC action output.
        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        current_ee = (
            self._ee_position()
        )

        # --------------------------------------------------------
        # Normalized action → Cartesian delta
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
        # Workspace clipping
        # --------------------------------------------------------

        desired = np.clip(
            desired,
            self.workspace_low,
            self.workspace_high,
        )

        # --------------------------------------------------------
        # Existing Cartesian impedance controller
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
        # Synchronous MuJoCo physics
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

        observation = (
            self._observation()
        )

        info = {
            **reward_info,

            "ee_position":
                self._ee_position(),

            "object_position":
                self._object_position(),

            "goal_position":
                self._goal_position(),

            "object_name":
                self.object_name,

            "step":
                self.step_count,
        }

        return (
            observation,
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

        self.robot._hold_running = False


# ================================================================
# Standalone test
# ================================================================

def main() -> None:

    print(
        "\n"
        "=======================================\n"
        "Panda Object Pre-Grasp Environment Test\n"
        "=======================================\n"
    )

    env = PandaObjectReachEnv(
        object_name="red_box",
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=42,
    )

    try:

        obs, info = env.reset()

        print(
            "Observation shape:",
            obs.shape,
        )

        print(
            "Expected observation:",
            env.observation_dim,
        )

        print(
            "Action dimension:",
            env.action_dim,
        )

        print(
            "\nObject:",
            info["object_name"],
        )

        print(
            "Object position:",
            np.round(
                info[
                    "object_position"
                ],
                4,
            ),
        )

        print(
            "Pre-grasp goal:",
            np.round(
                info[
                    "goal_position"
                ],
                4,
            ),
        )

        print(
            "Initial EE:",
            np.round(
                info[
                    "ee_position"
                ],
                4,
            ),
        )

        print(
            "Initial goal distance:",
            round(
                info[
                    "distance"
                ],
                4,
            ),
            "m",
        )

        print(
            "\nRunning 20 random actions...\n"
        )

        total_reward = 0.0

        final_info = info

        for step_index in range(
            1,
            21,
        ):

            action = (
                env.sample_random_action()
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                final_info,
            ) = env.step(
                action
            )

            total_reward += (
                reward
            )

            print(
                f"step="
                f"{step_index:03d} | "
                f"action="
                f"{np.round(action, 2)} | "
                f"distance="
                f"{final_info['distance']:.4f} | "
                f"progress="
                f"{final_info['progress']:+.4f} | "
                f"reward="
                f"{reward:+.4f} | "
                f"success="
                f"{final_info['success']}"
            )

            if terminated:

                print(
                    "\nSUCCESS: "
                    "pre-grasp goal reached."
                )

                break

            if truncated:

                print(
                    "\nEpisode truncated."
                )

                break

        print(
            "\nFinal EE:",
            np.round(
                final_info[
                    "ee_position"
                ],
                4,
            ),
        )

        print(
            "Object position:",
            np.round(
                final_info[
                    "object_position"
                ],
                4,
            ),
        )

        print(
            "Goal position:",
            np.round(
                final_info[
                    "goal_position"
                ],
                4,
            ),
        )

        print(
            "Final goal distance:",
            round(
                final_info[
                    "distance"
                ],
                4,
            ),
            "m",
        )

        print(
            "Total reward:",
            round(
                total_reward,
                4,
            ),
        )

        print(
            "\nEnvironment test completed."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()