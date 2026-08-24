# -*- coding: utf-8 -*-

"""
sweep_grasp_corrections.py

Systematically test grasp-target corrections on the exact object
positions that failed in the 100-episode manipulation benchmark.

Pipeline for each test:

    fixed failed object position
            ↓
    hybrid SAC pre-grasp
            ↓
    precision XY alignment
            ↓
    corrected grasp target:
        x = object_x + dx
        y = object_y + dy
        z = object_z + base_offset_z + dz
            ↓
    close gripper
            ↓
    lift
            ↓
    verify object lift

The goal is to find a correction that solves the difficult
low-X / negative-Y grasp region instead of manually guessing.

Input:
    models/hybrid_sac_grasp_100/failure_analysis/grasp_failures.csv

Output:
    models/hybrid_sac_grasp_100/grasp_correction_sweep/
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch


# ================================================================
# Project
# ================================================================

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
    / "grasp_correction_sweep"
)

RAW_OUTPUT = (
    OUTPUT_DIR
    / "all_trials.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "correction_summary.csv"
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
# General configuration
# ================================================================

ROUTER_Y_THRESHOLD = -0.10

MAX_RL_STEPS = 120

BASE_GRASP_OFFSET_Z = 0.115

LIFT_DISTANCE = 0.18

ALIGN_SEGMENTS = 30
DESCEND_SEGMENTS = 45
LIFT_SEGMENTS = 45

PHYSICS_STEPS_PER_SEGMENT = 20

GRIPPER_SETTLE_STEPS = 250
FINAL_SETTLE_STEPS = 150

MIN_OBJECT_LIFT = 0.05


# ================================================================
# Correction search space
# ================================================================
#
# Failures occur at low X.
# We test both directions, but give more resolution toward +X.
#
# All values are metres.
# ================================================================

DX_VALUES = [
    -0.005,
    0.000,
    0.005,
    0.010,
    0.015,
]

DY_VALUES = [
    -0.005,
    0.000,
    0.005,
]

DZ_VALUES = [
    0.000,
    0.005,
    0.010,
]


# ================================================================
# SAC loader
# ================================================================

def load_agent(
    model_path: Path,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> SACAgent:

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
# Physics helper
# ================================================================

def physics_step(
    env: PandaObjectReachEnv,
    steps: int,
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
# Force exact object position
# ================================================================

def set_object_xy(
    env: PandaObjectReachEnv,
    x: float,
    y: float,
):

    joint_id = env.object_free_joint_id

    if joint_id is None:

        raise RuntimeError(
            "Object has no free joint."
        )

    qpos_address = int(
        env.robot.model.jnt_qposadr[
            joint_id
        ]
    )

    z = float(
        env.robot.data.qpos[
            qpos_address + 2
        ]
    )

    env.robot.data.qpos[
        qpos_address
    ] = float(x)

    env.robot.data.qpos[
        qpos_address + 1
    ] = float(y)

    env.robot.data.qpos[
        qpos_address + 2
    ] = z

    # identity quaternion
    env.robot.data.qpos[
        qpos_address + 3
    ] = 1.0

    env.robot.data.qpos[
        qpos_address + 4
    ] = 0.0

    env.robot.data.qpos[
        qpos_address + 5
    ] = 0.0

    env.robot.data.qpos[
        qpos_address + 6
    ] = 0.0

    dof_address = int(
        env.robot.model.jnt_dofadr[
            joint_id
        ]
    )

    env.robot.data.qvel[
        dof_address:
        dof_address + 6
    ] = 0.0

    mujoco.mj_forward(
        env.robot.model,
        env.robot.data,
    )


# ================================================================
# Smooth movement
# ================================================================

def move_hand_smooth(
    env: PandaObjectReachEnv,
    target: np.ndarray,
    segments: int,
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
            PHYSICS_STEPS_PER_SEGMENT,
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

    return (
        final,
        error,
    )


# ================================================================
# Fixed-position hybrid pre-grasp
# ================================================================

def run_pregrasp(
    env,
    original_agent,
    targeted_agent,
    object_x,
    object_y,
):

    state, info = env.reset()

    # ------------------------------------------------------------
    # Override random object placement
    # ------------------------------------------------------------

    set_object_xy(
        env,
        object_x,
        object_y,
    )

    ee = env._ee_position()

    env.robot.target_pos = (
        ee.astype(float)
    )

    env.robot.target_quat = (
        env.robot.home_quat.copy()
    )

    goal = env._goal_position()

    distance = float(
        np.linalg.norm(
            goal - ee
        )
    )

    env.previous_distance = (
        distance
    )

    state = env._observation()

    # ------------------------------------------------------------
    # Policy router
    # ------------------------------------------------------------

    if object_y < ROUTER_Y_THRESHOLD:

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

    if final_info is None:

        raise RuntimeError(
            "Pre-grasp rollout failed to execute."
        )

    return {
        "success":
            success,

        "policy":
            policy,

        "steps":
            step,

        "final_distance":
            float(
                final_info[
                    "distance"
                ]
            ),
    }


# ================================================================
# Corrected grasp
# ================================================================

def run_grasp(
    env,
    dx,
    dy,
    dz,
):

    object_initial = (
        env._object_position()
        .astype(float)
    )

    hand_initial = (
        env._ee_position()
        .astype(float)
    )

    # ============================================================
    # A. Align over object
    # ============================================================

    alignment_target = np.array(
        [
            object_initial[0],
            object_initial[1],
            hand_initial[2],
        ],
        dtype=float,
    )

    (
        aligned_hand,
        alignment_error,
    ) = move_hand_smooth(
        env,
        alignment_target,
        ALIGN_SEGMENTS,
    )

    # ============================================================
    # B. Corrected descent target
    # ============================================================

    object_now = (
        env._object_position()
        .astype(float)
    )

    grasp_target = np.array(
        [
            object_now[0] + dx,
            object_now[1] + dy,
            object_now[2]
            + BASE_GRASP_OFFSET_Z
            + dz,
        ],
        dtype=float,
    )

    (
        hand_after_descent,
        descent_error,
    ) = move_hand_smooth(
        env,
        grasp_target,
        DESCEND_SEGMENTS,
    )

    # ------------------------------------------------------------
    # Actual tracking residual vector
    # ------------------------------------------------------------

    residual = (
        grasp_target
        - hand_after_descent
    )

    # ============================================================
    # C. Close
    # ============================================================

    env.robot.gripper(
        False
    )

    physics_step(
        env,
        GRIPPER_SETTLE_STEPS,
    )

    # ============================================================
    # D. Lift
    # ============================================================

    current_hand = (
        env._ee_position()
        .astype(float)
    )

    lift_target = (
        current_hand.copy()
    )

    lift_target[2] += (
        LIFT_DISTANCE
    )

    (
        _,
        lift_tracking_error,
    ) = move_hand_smooth(
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

    hand_final = (
        env._ee_position()
        .astype(float)
    )

    object_lift = float(
        object_final[2]
        - object_initial[2]
    )

    xy_motion = float(
        np.linalg.norm(
            object_final[:2]
            - object_initial[:2]
        )
    )

    hand_object_distance = float(
        np.linalg.norm(
            hand_final
            - object_final
        )
    )

    success = bool(
        object_lift
        >= MIN_OBJECT_LIFT
    )

    return {
        "success":
            success,

        "alignment_error":
            alignment_error,

        "descent_error":
            descent_error,

        "residual_x":
            float(
                residual[0]
            ),

        "residual_y":
            float(
                residual[1]
            ),

        "residual_z":
            float(
                residual[2]
            ),

        "lift_tracking_error":
            lift_tracking_error,

        "object_lift":
            object_lift,

        "xy_motion":
            xy_motion,

        "hand_object_distance":
            hand_object_distance,
    }


# ================================================================
# Main
# ================================================================

def main():

    print(
        "\n"
        "=============================================================="
    )

    print(
        "Failed-Region Grasp Correction Sweep"
    )

    print(
        "=============================================================="
    )

    if not FAILURE_CSV.exists():

        raise FileNotFoundError(
            f"Failure CSV missing:\n"
            f"{FAILURE_CSV}"
        )

    failures = pd.read_csv(
        FAILURE_CSV
    )

    print(
        "\nFailed object placements:",
        len(failures),
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

    corrections = []

    for dx in DX_VALUES:

        for dy in DY_VALUES:

            for dz in DZ_VALUES:

                corrections.append(
                    (
                        dx,
                        dy,
                        dz,
                    )
                )

    print(
        "\nCorrections tested:",
        len(corrections),
    )

    print(
        "Trials:",
        len(corrections)
        * len(failures),
    )

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ============================================================
    # Environment
    # ============================================================

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_RL_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=4040,
    )

    print(
        "\nLoading SAC models..."
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
        RAW_OUTPUT,
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

            "dx",
            "dy",
            "dz",

            "pregrasp_success",
            "pregrasp_distance",

            "grasp_success",

            "alignment_error",
            "descent_error",

            "descent_residual_x",
            "descent_residual_y",
            "descent_residual_z",

            "lift_tracking_error",

            "object_lift",
            "object_xy_motion",
            "hand_object_distance",
        ]
    )

    start_time = time.time()

    total_trials = (
        len(corrections)
        * len(failures)
    )

    trial_index = 0

    try:

        # ========================================================
        # Correction loop
        # ========================================================

        for correction_index, (
            dx,
            dy,
            dz,
        ) in enumerate(
            corrections,
            start=1,
        ):

            correction_results = []

            for _, row in failures.iterrows():

                trial_index += 1

                x = float(
                    row[
                        "object_x"
                    ]
                )

                y = float(
                    row[
                        "object_y"
                    ]
                )

                source_episode = int(
                    row[
                        "episode"
                    ]
                )

                # ------------------------------------------------
                # Learned pre-grasp
                # ------------------------------------------------

                pregrasp = run_pregrasp(
                    env,
                    original_agent,
                    targeted_agent,
                    x,
                    y,
                )

                if not pregrasp[
                    "success"
                ]:

                    print(
                        "\nWARNING: pre-grasp failed "
                        f"at source episode "
                        f"{source_episode}"
                    )

                    continue

                # ------------------------------------------------
                # Corrected grasp
                # ------------------------------------------------

                grasp = run_grasp(
                    env,
                    dx,
                    dy,
                    dz,
                )

                correction_results.append(
                    int(
                        grasp[
                            "success"
                        ]
                    )
                )

                writer.writerow(
                    [
                        source_episode,
                        x,
                        y,
                        pregrasp[
                            "policy"
                        ],

                        dx,
                        dy,
                        dz,

                        int(
                            pregrasp[
                                "success"
                            ]
                        ),

                        pregrasp[
                            "final_distance"
                        ],

                        int(
                            grasp[
                                "success"
                            ]
                        ),

                        grasp[
                            "alignment_error"
                        ],

                        grasp[
                            "descent_error"
                        ],

                        grasp[
                            "residual_x"
                        ],

                        grasp[
                            "residual_y"
                        ],

                        grasp[
                            "residual_z"
                        ],

                        grasp[
                            "lift_tracking_error"
                        ],

                        grasp[
                            "object_lift"
                        ],

                        grasp[
                            "xy_motion"
                        ],

                        grasp[
                            "hand_object_distance"
                        ],
                    ]
                )

                csv_file.flush()

            success_rate = (
                100.0
                * np.mean(
                    correction_results
                )
                if correction_results
                else 0.0
            )

            print(
                f"Correction "
                f"{correction_index:02d}/"
                f"{len(corrections)} | "
                f"dx={dx:+.3f} "
                f"dy={dy:+.3f} "
                f"dz={dz:+.3f} | "
                f"success="
                f"{int(np.sum(correction_results))}/"
                f"{len(correction_results)} "
                f"({success_rate:.1f}%)"
            )

        # ========================================================
        # Analyze results
        # ========================================================

        results = pd.read_csv(
            RAW_OUTPUT
        )

        summary = (
            results
            .groupby(
                [
                    "dx",
                    "dy",
                    "dz",
                ],
                as_index=False,
            )
            .agg(
                trials=(
                    "grasp_success",
                    "count",
                ),

                successes=(
                    "grasp_success",
                    "sum",
                ),

                mean_descent_error=(
                    "descent_error",
                    "mean",
                ),

                mean_abs_residual_x=(
                    "descent_residual_x",
                    lambda x:
                    np.mean(
                        np.abs(x)
                    ),
                ),

                mean_abs_residual_y=(
                    "descent_residual_y",
                    lambda x:
                    np.mean(
                        np.abs(x)
                    ),
                ),

                mean_abs_residual_z=(
                    "descent_residual_z",
                    lambda x:
                    np.mean(
                        np.abs(x)
                    ),
                ),

                mean_object_lift=(
                    "object_lift",
                    "mean",
                ),
            )
        )

        summary[
            "success_rate_percent"
        ] = (
            100.0
            * summary[
                "successes"
            ]
            / summary[
                "trials"
            ]
        )

        # Prefer:
        # 1. highest success
        # 2. smallest correction magnitude
        # 3. smallest descent error
        summary[
            "correction_magnitude"
        ] = np.sqrt(
            summary["dx"] ** 2
            + summary["dy"] ** 2
            + summary["dz"] ** 2
        )

        summary = summary.sort_values(
            [
                "success_rate_percent",
                "correction_magnitude",
                "mean_descent_error",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )

        summary.to_csv(
            SUMMARY_OUTPUT,
            index=False,
        )

        # ========================================================
        # Final output
        # ========================================================

        print(
            "\n"
            "=============================================================="
        )

        print(
            "TOP GRASP CORRECTIONS"
        )

        print(
            "=============================================================="
        )

        columns = [
            "dx",
            "dy",
            "dz",
            "successes",
            "trials",
            "success_rate_percent",
            "mean_descent_error",
            "mean_object_lift",
            "correction_magnitude",
        ]

        print(
            summary[
                columns
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

        best = (
            summary.iloc[0]
        )

        print(
            "\n"
            "=============================================================="
        )

        print(
            "BEST CORRECTION"
        )

        print(
            "=============================================================="
        )

        print(
            f"dx: "
            f"{best['dx']:+.4f} m"
        )

        print(
            f"dy: "
            f"{best['dy']:+.4f} m"
        )

        print(
            f"dz: "
            f"{best['dz']:+.4f} m"
        )

        print(
            f"Success: "
            f"{int(best['successes'])}/"
            f"{int(best['trials'])}"
        )

        print(
            f"Success rate: "
            f"{best['success_rate_percent']:.1f}%"
        )

        print(
            f"Mean descent error: "
            f"{best['mean_descent_error']:.4f} m"
        )

        print(
            f"Mean object lift: "
            f"{best['mean_object_lift']:.4f} m"
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"\nElapsed: "
            f"{elapsed / 60.0:.1f} min"
        )

        print(
            "\nRaw trials:"
        )

        print(
            RAW_OUTPUT
        )

        print(
            "\nSummary:"
        )

        print(
            SUMMARY_OUTPUT
        )

        print(
            "==============================================================\n"
        )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
