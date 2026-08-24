# -*- coding: utf-8 -*-

"""
sweep_staged_grasp_approach.py

Diagnose low-X grasp failures using a staged Cartesian approach.

Instead of descending vertically directly above the low-X object:

    object XY -> vertical descent

we test:

    1. SAC pre-grasp
    2. move to safe approach X
    3. descend at safe X
    4. move horizontally to object at grasp height
    5. close
    6. lift

This tests whether the previous failures are caused by poor Cartesian
tracking / kinematic conditioning during low-X vertical descent.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Paths
# ================================================================

FAILURE_CSV = (
    PROJECT_ROOT
    / "models"
    / "hybrid_sac_grasp_100"
    / "failure_analysis"
    / "grasp_failures.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid_sac_grasp_100"
    / "staged_approach_sweep"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "staged_trials.csv"
)


# ================================================================
# Models
# ================================================================

ORIGINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "sac_object_reach_best.pt"
)

TARGETED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
    / "sac_object_targeted_best.pt"
)


# ================================================================
# Configuration
# ================================================================

ROUTER_Y_THRESHOLD = -0.10

MAX_RL_STEPS = 120

GRASP_HAND_OFFSET_Z = 0.115

LIFT_DISTANCE = 0.18

MIN_OBJECT_LIFT = 0.05

PHYSICS_STEPS = 20

PREALIGN_SEGMENTS = 30
DESCEND_SEGMENTS = 60
INWARD_SEGMENTS = 50
LIFT_SEGMENTS = 45

GRIPPER_SETTLE_STEPS = 250
FINAL_SETTLE_STEPS = 150


# ================================================================
# Safe approach X candidates
# ================================================================

SAFE_X_VALUES = [
    0.44,
    0.45,
    0.46,
    0.47,
    0.48,
]


# ================================================================
# SAC
# ================================================================

def load_agent(
    model_path,
    state_dim,
    action_dim,
    device,
):

    config = SACConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        initial_alpha=0.2,
    )

    agent = SACAgent(
        config=config,
        device=device,
    )

    agent.load(
        model_path
    )

    agent.actor.eval()
    agent.critics.eval()
    agent.target_critics.eval()

    return agent


# ================================================================
# Physics
# ================================================================

def physics_step(
    env,
    steps,
):

    for _ in range(steps):

        env.robot.control(
            env.robot.target_pos,
            env.robot.target_quat,
        )

        mujoco.mj_step(
            env.robot.model,
            env.robot.data,
        )


# ================================================================
# Exact object placement
# ================================================================

def set_object_xy(
    env,
    x,
    y,
):

    joint_id = env.object_free_joint_id

    if joint_id is None:
        raise RuntimeError(
            "Object has no free joint."
        )

    qadr = int(
        env.robot.model.jnt_qposadr[
            joint_id
        ]
    )

    z = float(
        env.robot.data.qpos[
            qadr + 2
        ]
    )

    env.robot.data.qpos[
        qadr
    ] = float(x)

    env.robot.data.qpos[
        qadr + 1
    ] = float(y)

    env.robot.data.qpos[
        qadr + 2
    ] = z

    env.robot.data.qpos[
        qadr + 3
    ] = 1.0

    env.robot.data.qpos[
        qadr + 4
    ] = 0.0

    env.robot.data.qpos[
        qadr + 5
    ] = 0.0

    env.robot.data.qpos[
        qadr + 6
    ] = 0.0

    dadr = int(
        env.robot.model.jnt_dofadr[
            joint_id
        ]
    )

    env.robot.data.qvel[
        dadr:
        dadr + 6
    ] = 0.0

    mujoco.mj_forward(
        env.robot.model,
        env.robot.data,
    )


# ================================================================
# Smooth Cartesian move
# ================================================================

def move_smooth(
    env,
    target,
    segments,
):

    start = (
        env._ee_position()
        .astype(float)
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    for i in range(
        1,
        segments + 1,
    ):

        alpha = (
            i / segments
        )

        desired = (
            start
            + alpha
            * (
                target - start
            )
        )

        env.robot.target_pos = (
            desired.copy()
        )

        env.robot.target_quat = (
            env.robot.home_quat.copy()
        )

        physics_step(
            env,
            PHYSICS_STEPS,
        )

    final = (
        env._ee_position()
        .astype(float)
    )

    error = float(
        np.linalg.norm(
            target - final
        )
    )

    residual = (
        target - final
    )

    return (
        final,
        error,
        residual,
    )


# ================================================================
# Hybrid SAC pre-grasp
# ================================================================

def run_pregrasp(
    env,
    original_agent,
    targeted_agent,
    x,
    y,
):

    state, info = env.reset()

    set_object_xy(
        env,
        x,
        y,
    )

    ee = env._ee_position()

    env.robot.target_pos = (
        ee.astype(float)
    )

    env.robot.target_quat = (
        env.robot.home_quat.copy()
    )

    goal = env._goal_position()

    env.previous_distance = float(
        np.linalg.norm(
            goal - ee
        )
    )

    state = env._observation()

    if y < ROUTER_Y_THRESHOLD:

        agent = targeted_agent
        policy = "targeted"

    else:

        agent = original_agent
        policy = "original"

    success = False
    final_info = None

    for step in range(
        1,
        MAX_RL_STEPS + 1,
    ):

        action = agent.select_action(
            state,
            deterministic=True,
        )

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        final_info = info

        if terminated:

            success = True
            break

        if truncated:
            break

    return (
        success,
        policy,
        final_info,
    )


# ================================================================
# Staged grasp
# ================================================================

def staged_grasp(
    env,
    safe_x,
):

    object_initial = (
        env._object_position()
        .astype(float)
    )

    hand_initial = (
        env._ee_position()
        .astype(float)
    )

    object_x = float(
        object_initial[0]
    )

    object_y = float(
        object_initial[1]
    )

    grasp_z = float(
        object_initial[2]
        + GRASP_HAND_OFFSET_Z
    )

    # ============================================================
    # 1. Move to safe approach X while remaining high
    # ============================================================

    high_target = np.array(
        [
            safe_x,
            object_y,
            hand_initial[2],
        ],
        dtype=float,
    )

    (
        hand_high,
        high_error,
        high_residual,
    ) = move_smooth(
        env,
        high_target,
        PREALIGN_SEGMENTS,
    )

    # ============================================================
    # 2. Vertical descent at safe X
    # ============================================================

    descend_target = np.array(
        [
            safe_x,
            object_y,
            grasp_z,
        ],
        dtype=float,
    )

    (
        hand_after_descent,
        descent_error,
        descent_residual,
    ) = move_smooth(
        env,
        descend_target,
        DESCEND_SEGMENTS,
    )

    # ============================================================
    # 3. Horizontal inward movement at grasp height
    # ============================================================

    inward_target = np.array(
        [
            object_x,
            object_y,
            grasp_z,
        ],
        dtype=float,
    )

    (
        hand_after_inward,
        inward_error,
        inward_residual,
    ) = move_smooth(
        env,
        inward_target,
        INWARD_SEGMENTS,
    )

    # ============================================================
    # 4. Close
    # ============================================================

    env.robot.gripper(
        False
    )

    physics_step(
        env,
        GRIPPER_SETTLE_STEPS,
    )

    object_after_close = (
        env._object_position()
        .astype(float)
    )

    # ============================================================
    # 5. Lift
    # ============================================================

    hand_now = (
        env._ee_position()
        .astype(float)
    )

    lift_target = (
        hand_now.copy()
    )

    lift_target[2] += (
        LIFT_DISTANCE
    )

    (
        hand_after_lift,
        lift_error,
        _,
    ) = move_smooth(
        env,
        lift_target,
        LIFT_SEGMENTS,
    )

    physics_step(
        env,
        FINAL_SETTLE_STEPS,
    )

    object_final = (
        env._object_position()
        .astype(float)
    )

    object_lift = float(
        object_final[2]
        - object_initial[2]
    )

    success = bool(
        object_lift
        >= MIN_OBJECT_LIFT
    )

    return {
        "success":
            success,

        "safe_x":
            safe_x,

        "high_error":
            high_error,

        "descent_error":
            descent_error,

        "descent_residual_x":
            float(
                descent_residual[0]
            ),

        "descent_residual_y":
            float(
                descent_residual[1]
            ),

        "descent_residual_z":
            float(
                descent_residual[2]
            ),

        "inward_error":
            inward_error,

        "inward_residual_x":
            float(
                inward_residual[0]
            ),

        "inward_residual_y":
            float(
                inward_residual[1]
            ),

        "inward_residual_z":
            float(
                inward_residual[2]
            ),

        "lift_error":
            lift_error,

        "object_lift":
            object_lift,

        "object_after_close_z":
            float(
                object_after_close[2]
            ),
    }


# ================================================================
# Main
# ================================================================

def main():

    print(
        "\n"
        "============================================================"
    )

    print(
        "Low-X Staged Grasp Approach Sweep"
    )

    print(
        "============================================================"
    )

    failures = pd.read_csv(
        FAILURE_CSV
    )

    print(
        "\nFailure positions:"
    )

    print(
        failures[
            [
                "episode",
                "object_x",
                "object_y",
            ]
        ].to_string(
            index=False
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_RL_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=5050,
    )

    original_agent = load_agent(
        ORIGINAL_MODEL,
        env.observation_dim,
        env.action_dim,
        device,
    )

    targeted_agent = load_agent(
        TARGETED_MODEL,
        env.observation_dim,
        env.action_dim,
        device,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
    )

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "source_episode",
            "object_x",
            "object_y",
            "policy",
            "safe_x",
            "success",
            "pregrasp_distance",
            "high_error",
            "descent_error",
            "descent_residual_x",
            "descent_residual_y",
            "descent_residual_z",
            "inward_error",
            "inward_residual_x",
            "inward_residual_y",
            "inward_residual_z",
            "lift_error",
            "object_lift",
        ]
    )

    try:

        results = []

        for safe_x in SAFE_X_VALUES:

            successes = []

            descent_errors = []

            inward_errors = []

            lift_values = []

            for _, row in failures.iterrows():

                source_episode = int(
                    row["episode"]
                )

                x = float(
                    row["object_x"]
                )

                y = float(
                    row["object_y"]
                )

                (
                    pregrasp_success,
                    policy,
                    pregrasp_info,
                ) = run_pregrasp(
                    env,
                    original_agent,
                    targeted_agent,
                    x,
                    y,
                )

                if not pregrasp_success:

                    print(
                        f"WARNING: pregrasp failed "
                        f"episode={source_episode}"
                    )

                    continue

                grasp = staged_grasp(
                    env,
                    safe_x,
                )

                successes.append(
                    int(
                        grasp["success"]
                    )
                )

                descent_errors.append(
                    grasp[
                        "descent_error"
                    ]
                )

                inward_errors.append(
                    grasp[
                        "inward_error"
                    ]
                )

                lift_values.append(
                    grasp[
                        "object_lift"
                    ]
                )

                writer.writerow(
                    [
                        source_episode,
                        x,
                        y,
                        policy,
                        safe_x,
                        int(
                            grasp["success"]
                        ),
                        float(
                            pregrasp_info[
                                "distance"
                            ]
                        ),
                        grasp[
                            "high_error"
                        ],
                        grasp[
                            "descent_error"
                        ],
                        grasp[
                            "descent_residual_x"
                        ],
                        grasp[
                            "descent_residual_y"
                        ],
                        grasp[
                            "descent_residual_z"
                        ],
                        grasp[
                            "inward_error"
                        ],
                        grasp[
                            "inward_residual_x"
                        ],
                        grasp[
                            "inward_residual_y"
                        ],
                        grasp[
                            "inward_residual_z"
                        ],
                        grasp[
                            "lift_error"
                        ],
                        grasp[
                            "object_lift"
                        ],
                    ]
                )

                csv_file.flush()

            rate = (
                100.0
                * np.mean(
                    successes
                )
            )

            result = {
                "safe_x":
                    safe_x,

                "successes":
                    int(
                        np.sum(
                            successes
                        )
                    ),

                "trials":
                    len(
                        successes
                    ),

                "success_rate":
                    rate,

                "mean_descent_error":
                    float(
                        np.mean(
                            descent_errors
                        )
                    ),

                "mean_inward_error":
                    float(
                        np.mean(
                            inward_errors
                        )
                    ),

                "mean_object_lift":
                    float(
                        np.mean(
                            lift_values
                        )
                    ),
            }

            results.append(
                result
            )

            print(
                f"\nsafe_x="
                f"{safe_x:.3f} | "
                f"success="
                f"{result['successes']}/"
                f"{result['trials']} "
                f"({rate:.1f}%) | "
                f"descent_err="
                f"{result['mean_descent_error']:.4f} | "
                f"inward_err="
                f"{result['mean_inward_error']:.4f} | "
                f"lift="
                f"{result['mean_object_lift']:.4f}"
            )

        # ========================================================
        # Rank
        # ========================================================

        results = sorted(
            results,
            key=lambda x: (
                -x[
                    "success_rate"
                ],
                x[
                    "mean_inward_error"
                ],
            ),
        )

        print(
            "\n"
            "============================================================"
        )

        print(
            "STAGED APPROACH RESULTS"
        )

        print(
            "============================================================"
        )

        print(
            f"{'safe_x':>8} "
            f"{'success':>10} "
            f"{'rate':>10} "
            f"{'desc_err':>10} "
            f"{'in_err':>10} "
            f"{'lift':>10}"
        )

        for item in results:

            print(
                f"{item['safe_x']:>8.3f} "
                f"{item['successes']:>4}/"
                f"{item['trials']:<5} "
                f"{item['success_rate']:>9.1f}% "
                f"{item['mean_descent_error']:>10.4f} "
                f"{item['mean_inward_error']:>10.4f} "
                f"{item['mean_object_lift']:>10.4f}"
            )

        best = results[0]

        print(
            "\nBEST SAFE-X:"
        )

        print(
            f"safe_x = "
            f"{best['safe_x']:.3f} m"
        )

        print(
            f"success = "
            f"{best['successes']}/"
            f"{best['trials']} "
            f"({best['success_rate']:.1f}%)"
        )

        print(
            "\nCSV:"
        )

        print(
            OUTPUT_CSV
        )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
