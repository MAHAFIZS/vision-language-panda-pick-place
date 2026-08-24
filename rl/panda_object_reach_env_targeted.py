# -*- coding: utf-8 -*-

"""
panda_object_reach_env_targeted.py

Failure-aware object pre-grasp environment.

Based on PandaObjectReachEnv.

Sampling:
    70% hard object placements
    30% uniform object placements

Failure analysis showed the difficult region is approximately:

    X = 0.448 -> 0.578 m
    Y = -0.191 -> -0.149 m

We slightly expand this region during retraining:

    X = 0.44 -> 0.59 m
    Y = -0.20 -> -0.14 m

This avoids simply memorizing the exact failed samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv


class PandaObjectReachTargetedEnv(
    PandaObjectReachEnv
):

    def __init__(
        self,
        object_name: str = "red_box",
        max_steps: int = 120,
        action_scale: float = 0.025,
        physics_steps_per_action: int = 40,
        success_threshold: float = 0.040,
        pregrasp_height: float = 0.15,
        hard_target_probability: float = 0.70,
        seed: int = 42,
    ) -> None:

        super().__init__(
            object_name=object_name,
            max_steps=max_steps,
            action_scale=action_scale,
            physics_steps_per_action=physics_steps_per_action,
            success_threshold=success_threshold,
            pregrasp_height=pregrasp_height,
            seed=seed,
        )

        self.hard_target_probability = float(
            hard_target_probability
        )

        # --------------------------------------------------------
        # Failure-aware object region
        # --------------------------------------------------------

        self.hard_x_low = 0.44
        self.hard_x_high = 0.59

        self.hard_y_low = -0.20
        self.hard_y_high = -0.14

        self.last_object_mode = "uniform"

    # ============================================================
    # Override object randomization
    # ============================================================

    def _randomize_object_position(
        self,
    ) -> None:

        if self.object_free_joint_id is None:

            print(
                f"[warning] {self.object_name} "
                "has no free joint."
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

        current_z = float(
            self.robot.data
            .qpos[
                qpos_address + 2
            ]
        )

        # --------------------------------------------------------
        # Decide uniform vs failure-focused sample
        # --------------------------------------------------------

        use_hard = bool(
            self.rng.random()
            < self.hard_target_probability
        )

        if use_hard:

            x = self.rng.uniform(
                self.hard_x_low,
                self.hard_x_high,
            )

            y = self.rng.uniform(
                self.hard_y_low,
                self.hard_y_high,
            )

            self.last_object_mode = (
                "hard_negative_y"
            )

        else:

            x = self.rng.uniform(
                self.object_xy_low[0],
                self.object_xy_high[0],
            )

            y = self.rng.uniform(
                self.object_xy_low[1],
                self.object_xy_high[1],
            )

            self.last_object_mode = (
                "uniform"
            )

        # --------------------------------------------------------
        # Set free-joint pose
        # --------------------------------------------------------

        self.robot.data.qpos[
            qpos_address
        ] = float(x)

        self.robot.data.qpos[
            qpos_address + 1
        ] = float(y)

        self.robot.data.qpos[
            qpos_address + 2
        ] = current_z

        # Identity quaternion.
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
        # Zero object velocity
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
    # Add sampling mode to reset info
    # ============================================================

    def reset(
        self,
    ):

        observation, info = (
            super().reset()
        )

        info[
            "object_mode"
        ] = self.last_object_mode

        return (
            observation,
            info,
        )

    # ============================================================
    # Add mode to step info
    # ============================================================

    def step(
        self,
        action,
    ):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(
            action
        )

        info[
            "object_mode"
        ] = self.last_object_mode

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


# ================================================================
# Test
# ================================================================

def main():

    print(
        "\n"
        "=============================================\n"
        "Targeted Object-Reach Environment Test\n"
        "=============================================\n"
    )

    env = PandaObjectReachTargetedEnv(
        object_name="red_box",
        hard_target_probability=0.70,
        seed=123,
    )

    counts = {
        "uniform": 0,
        "hard_negative_y": 0,
    }

    try:

        for episode in range(
            1,
            21,
        ):

            obs, info = env.reset()

            mode = info[
                "object_mode"
            ]

            counts[
                mode
            ] += 1

            obj = info[
                "object_position"
            ]

            goal = info[
                "goal_position"
            ]

            print(
                f"Episode {episode:02d} | "
                f"mode={mode:16s} | "
                f"object={np.round(obj, 3)} | "
                f"goal={np.round(goal, 3)} | "
                f"distance="
                f"{info['distance']:.4f}"
            )

        print(
            "\nSampling counts:"
        )

        for key, value in counts.items():

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
            "\nTargeted environment test completed."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()
