# -*- coding: utf-8 -*-

"""
evaluate_hybrid_sac_grasp_100.py

Benchmark the complete manipulation pipeline:

    object position
        ↓
    failure-aware router
        ↓
    original / targeted SAC
        ↓
    learned pre-grasp
        ↓
    scripted XY alignment
        ↓
    scripted descent
        ↓
    close gripper
        ↓
    lift
        ↓
    grasp verification

Evaluation:
    100 randomized object placements

Records:
    - selected SAC policy
    - pre-grasp success
    - grasp success
    - full-pipeline success
    - pre-grasp final error
    - object lift distance
    - final hand-object distance
    - object x/y
    - number of RL steps
    - per-route performance
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch


# ================================================================
# Project imports
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100

EVAL_SEED = 3030

MAX_RL_STEPS = 120

ROUTER_Y_THRESHOLD = -0.10


# ------------------------------------------------
# Grasp parameters
# ------------------------------------------------

GRASP_HAND_OFFSET_Z = 0.115

LIFT_DISTANCE = 0.18

ALIGN_SEGMENTS = 30

DESCEND_SEGMENTS = 45

LIFT_SEGMENTS = 45

PHYSICS_STEPS_PER_SEGMENT = 20

GRIPPER_SETTLE_STEPS = 250

FINAL_SETTLE_STEPS = 150

MIN_OBJECT_LIFT = 0.05


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
# Output
# ================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid_sac_grasp_100"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "hybrid_grasp_100.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.txt"
)


# ================================================================
# Agent loader
# ================================================================

def load_agent(
    model_path: Path,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> SACAgent:

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

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
    env: PandaObjectReachEnv,
    steps: int,
) -> None:

    for _ in range(
        steps
    ):

        env.robot.control(
            env.robot.target_pos,
            env.robot.target_quat,
        )

        mujoco.mj_step(
            env.robot.model,
            env.robot.data,
        )


# ================================================================
# Smooth Cartesian movement
# ================================================================

def move_hand_smooth(
    env: PandaObjectReachEnv,
    target_position: np.ndarray,
    segments: int,
    physics_steps_per_segment: int,
) -> float:

    start = (
        env._ee_position()
        .astype(float)
    )

    target = np.asarray(
        target_position,
        dtype=float,
    )

    for index in range(
        1,
        segments + 1,
    ):

        alpha = (
            index
            / segments
        )

        desired = (
            start
            + alpha
            * (
                target
                - start
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
            physics_steps_per_segment,
        )

    final = (
        env._ee_position()
        .astype(float)
    )

    error = float(
        np.linalg.norm(
            target
            - final
        )
    )

    return error


# ================================================================
# Hybrid pre-grasp
# ================================================================

def run_hybrid_pregrasp(
    env: PandaObjectReachEnv,
    original_agent: SACAgent,
    targeted_agent: SACAgent,
):

    state, info = (
        env.reset()
    )

    object_position = np.asarray(
        info[
            "object_position"
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Router
    # ------------------------------------------------------------

    if (
        object_position[1]
        < ROUTER_Y_THRESHOLD
    ):

        agent = targeted_agent

        policy_name = (
            "targeted"
        )

    else:

        agent = original_agent

        policy_name = (
            "original"
        )

    initial_distance = float(
        info[
            "distance"
        ]
    )

    pregrasp_success = False

    final_info = info

    rl_steps = 0

    total_rl_reward = 0.0

    # ------------------------------------------------------------
    # Deterministic SAC rollout
    # ------------------------------------------------------------

    for step in range(
        1,
        MAX_RL_STEPS + 1,
    ):

        action = (
            agent.select_action(
                state,
                deterministic=True,
            )
        )

        (
            state,
            reward,
            terminated,
            truncated,
            final_info,
        ) = env.step(
            action
        )

        rl_steps = step

        total_rl_reward += reward

        if terminated:

            pregrasp_success = True

            break

        if truncated:

            break

    return {
        "success":
            pregrasp_success,

        "policy":
            policy_name,

        "initial_distance":
            initial_distance,

        "final_distance":
            float(
                final_info[
                    "distance"
                ]
            ),

        "rl_steps":
            rl_steps,

        "rl_reward":
            total_rl_reward,

        "object_position":
            np.asarray(
                final_info[
                    "object_position"
                ],
                dtype=np.float32,
            ),

        "goal_position":
            np.asarray(
                final_info[
                    "goal_position"
                ],
                dtype=np.float32,
            ),

        "final_ee":
            np.asarray(
                final_info[
                    "ee_position"
                ],
                dtype=np.float32,
            ),
    }


# ================================================================
# Scripted grasp
# ================================================================

def execute_scripted_grasp(
    env: PandaObjectReachEnv,
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
    # A. Precision XY alignment
    # ============================================================

    alignment_target = np.array(
        [
            object_initial[0],
            object_initial[1],
            hand_initial[2],
        ],
        dtype=float,
    )

    alignment_error = (
        move_hand_smooth(
            env=env,
            target_position=alignment_target,
            segments=ALIGN_SEGMENTS,
            physics_steps_per_segment=(
                PHYSICS_STEPS_PER_SEGMENT
            ),
        )
    )

    # ============================================================
    # B. Vertical descent
    # ============================================================

    object_before_descent = (
        env._object_position()
        .astype(float)
    )

    grasp_target = np.array(
        [
            object_before_descent[0],
            object_before_descent[1],
            object_before_descent[2]
            + GRASP_HAND_OFFSET_Z,
        ],
        dtype=float,
    )

    descent_error = (
        move_hand_smooth(
            env=env,
            target_position=grasp_target,
            segments=DESCEND_SEGMENTS,
            physics_steps_per_segment=(
                PHYSICS_STEPS_PER_SEGMENT
            ),
        )
    )

    # ============================================================
    # C. Close gripper
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

    hand_after_close = (
        env._ee_position()
        .astype(float)
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

    lift_tracking_error = (
        move_hand_smooth(
            env=env,
            target_position=lift_target,
            segments=LIFT_SEGMENTS,
            physics_steps_per_segment=(
                PHYSICS_STEPS_PER_SEGMENT
            ),
        )
    )

    # ------------------------------------------------------------
    # Final settle
    # ------------------------------------------------------------

    physics_step(
        env,
        FINAL_SETTLE_STEPS,
    )

    # ============================================================
    # Verification
    # ============================================================

    object_final = (
        env._object_position()
        .astype(float)
    )

    hand_final = (
        env._ee_position()
        .astype(float)
    )

    object_vertical_motion = float(
        object_final[2]
        - object_initial[2]
    )

    object_xy_motion = float(
        np.linalg.norm(
            object_final[:2]
            - object_initial[:2]
        )
    )

    final_hand_object_distance = float(
        np.linalg.norm(
            hand_final
            - object_final
        )
    )

    grasp_success = bool(
        object_vertical_motion
        >= MIN_OBJECT_LIFT
    )

    return {
        "success":
            grasp_success,

        "alignment_error":
            alignment_error,

        "descent_error":
            descent_error,

        "lift_tracking_error":
            lift_tracking_error,

        "object_initial":
            object_initial,

        "object_after_close":
            object_after_close,

        "object_final":
            object_final,

        "hand_after_close":
            hand_after_close,

        "hand_final":
            hand_final,

        "object_vertical_motion":
            object_vertical_motion,

        "object_xy_motion":
            object_xy_motion,

        "final_hand_object_distance":
            final_hand_object_distance,
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
        "100-Episode Hybrid SAC + Grasp Benchmark"
    )

    print(
        "=============================================================="
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    np.random.seed(
        EVAL_SEED
    )

    torch.manual_seed(
        EVAL_SEED
    )

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_RL_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=EVAL_SEED,
    )

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    print(
        "\nLoading original SAC..."
    )

    original_agent = (
        load_agent(
            ORIGINAL_MODEL,
            env.observation_dim,
            env.action_dim,
            device,
        )
    )

    print(
        "Loading targeted SAC..."
    )

    targeted_agent = (
        load_agent(
            TARGETED_MODEL,
            env.observation_dim,
            env.action_dim,
            device,
        )
    )

    # ============================================================
    # Storage
    # ============================================================

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
            "episode",
            "policy",
            "object_x",
            "object_y",
            "object_z",

            "pregrasp_success",
            "initial_pregrasp_distance",
            "final_pregrasp_distance",
            "rl_steps",
            "rl_reward",

            "grasp_attempted",
            "grasp_success",
            "full_pipeline_success",

            "alignment_error",
            "descent_error",
            "lift_tracking_error",

            "object_lift_m",
            "object_xy_motion_m",
            "final_hand_object_distance_m",

            "object_final_x",
            "object_final_y",
            "object_final_z",
        ]
    )

    # ============================================================
    # Metrics
    # ============================================================

    pregrasp_successes = []

    grasp_successes = []

    full_successes = []

    original_route_results = []

    targeted_route_results = []

    lift_values = []

    pregrasp_errors = []

    rl_steps_success = []

    start_time = time.time()

    try:

        # ========================================================
        # Evaluation loop
        # ========================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            # ----------------------------------------------------
            # RL pre-grasp
            # ----------------------------------------------------

            pregrasp = (
                run_hybrid_pregrasp(
                    env,
                    original_agent,
                    targeted_agent,
                )
            )

            object_position = (
                pregrasp[
                    "object_position"
                ]
            )

            policy_name = (
                pregrasp[
                    "policy"
                ]
            )

            pregrasp_success = bool(
                pregrasp[
                    "success"
                ]
            )

            pregrasp_successes.append(
                1
                if pregrasp_success
                else 0
            )

            pregrasp_errors.append(
                pregrasp[
                    "final_distance"
                ]
            )

            if pregrasp_success:

                rl_steps_success.append(
                    pregrasp[
                        "rl_steps"
                    ]
                )

            # ----------------------------------------------------
            # Grasp
            # ----------------------------------------------------

            grasp_attempted = False

            grasp_success = False

            grasp = None

            if pregrasp_success:

                grasp_attempted = True

                grasp = (
                    execute_scripted_grasp(
                        env
                    )
                )

                grasp_success = bool(
                    grasp[
                        "success"
                    ]
                )

                grasp_successes.append(
                    1
                    if grasp_success
                    else 0
                )

                lift_values.append(
                    grasp[
                        "object_vertical_motion"
                    ]
                )

            # ----------------------------------------------------
            # Full pipeline
            # ----------------------------------------------------

            full_success = bool(
                pregrasp_success
                and grasp_success
            )

            full_successes.append(
                1
                if full_success
                else 0
            )

            if policy_name == "original":

                original_route_results.append(
                    1
                    if full_success
                    else 0
                )

            else:

                targeted_route_results.append(
                    1
                    if full_success
                    else 0
                )

            # ----------------------------------------------------
            # CSV values
            # ----------------------------------------------------

            if grasp is not None:

                alignment_error = (
                    grasp[
                        "alignment_error"
                    ]
                )

                descent_error = (
                    grasp[
                        "descent_error"
                    ]
                )

                lift_tracking_error = (
                    grasp[
                        "lift_tracking_error"
                    ]
                )

                object_lift = (
                    grasp[
                        "object_vertical_motion"
                    ]
                )

                object_xy_motion = (
                    grasp[
                        "object_xy_motion"
                    ]
                )

                hand_object_distance = (
                    grasp[
                        "final_hand_object_distance"
                    ]
                )

                object_final = (
                    grasp[
                        "object_final"
                    ]
                )

            else:

                alignment_error = np.nan

                descent_error = np.nan

                lift_tracking_error = np.nan

                object_lift = 0.0

                object_xy_motion = np.nan

                hand_object_distance = np.nan

                object_final = (
                    object_position
                )

            writer.writerow(
                [
                    episode,
                    policy_name,

                    float(
                        object_position[0]
                    ),

                    float(
                        object_position[1]
                    ),

                    float(
                        object_position[2]
                    ),

                    int(
                        pregrasp_success
                    ),

                    pregrasp[
                        "initial_distance"
                    ],

                    pregrasp[
                        "final_distance"
                    ],

                    pregrasp[
                        "rl_steps"
                    ],

                    pregrasp[
                        "rl_reward"
                    ],

                    int(
                        grasp_attempted
                    ),

                    int(
                        grasp_success
                    ),

                    int(
                        full_success
                    ),

                    alignment_error,

                    descent_error,

                    lift_tracking_error,

                    object_lift,

                    object_xy_motion,

                    hand_object_distance,

                    float(
                        object_final[0]
                    ),

                    float(
                        object_final[1]
                    ),

                    float(
                        object_final[2]
                    ),
                ]
            )

            csv_file.flush()

            # ----------------------------------------------------
            # Running metrics
            # ----------------------------------------------------

            running_pregrasp = (
                100.0
                * np.mean(
                    pregrasp_successes
                )
            )

            running_full = (
                100.0
                * np.mean(
                    full_successes
                )
            )

            if grasp_successes:

                running_grasp_given_reach = (
                    100.0
                    * np.mean(
                        grasp_successes
                    )
                )

            else:

                running_grasp_given_reach = (
                    0.0
                )

            if (
                episode <= 10
                or episode % 10 == 0
                or not full_success
            ):

                print(
                    f"Episode "
                    f"{episode:03d}/{NUM_EPISODES} | "
                    f"policy="
                    f"{policy_name:8s} | "
                    f"xy="
                    f"[{object_position[0]:.3f}, "
                    f"{object_position[1]:.3f}] | "
                    f"pregrasp="
                    f"{pregrasp_success} | "
                    f"grasp="
                    f"{grasp_success} | "
                    f"full="
                    f"{full_success} | "
                    f"pregrasp_err="
                    f"{pregrasp['final_distance']:.4f} | "
                    f"lift="
                    f"{object_lift:.4f} | "
                    f"running="
                    f"{running_full:.1f}%"
                )

        # ========================================================
        # Final statistics
        # ========================================================

        pregrasp_np = np.asarray(
            pregrasp_successes,
            dtype=np.float32,
        )

        full_np = np.asarray(
            full_successes,
            dtype=np.float32,
        )

        pregrasp_error_np = np.asarray(
            pregrasp_errors,
            dtype=np.float32,
        )

        pregrasp_rate = float(
            pregrasp_np.mean()
            * 100.0
        )

        full_rate = float(
            full_np.mean()
            * 100.0
        )

        # --------------------------------------------------------
        # Grasp success conditional on pre-grasp success
        # --------------------------------------------------------

        if grasp_successes:

            grasp_given_reach_rate = float(
                np.mean(
                    grasp_successes
                )
                * 100.0
            )

        else:

            grasp_given_reach_rate = 0.0

        # --------------------------------------------------------
        # Route performance
        # --------------------------------------------------------

        original_route_rate = (
            float(
                np.mean(
                    original_route_results
                )
                * 100.0
            )
            if original_route_results
            else 0.0
        )

        targeted_route_rate = (
            float(
                np.mean(
                    targeted_route_results
                )
                * 100.0
            )
            if targeted_route_results
            else 0.0
        )

        # --------------------------------------------------------
        # Other metrics
        # --------------------------------------------------------

        mean_pregrasp_error = float(
            pregrasp_error_np.mean()
        )

        worst_pregrasp_error = float(
            pregrasp_error_np.max()
        )

        if lift_values:

            lift_np = np.asarray(
                lift_values,
                dtype=np.float32,
            )

            mean_lift = float(
                lift_np.mean()
            )

            min_lift = float(
                lift_np.min()
            )

            max_lift = float(
                lift_np.max()
            )

        else:

            mean_lift = 0.0
            min_lift = 0.0
            max_lift = 0.0

        mean_rl_steps_success = (
            float(
                np.mean(
                    rl_steps_success
                )
            )
            if rl_steps_success
            else float("nan")
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ========================================================
        # Print
        # ========================================================

        print(
            "\n"
            "=============================================================="
        )

        print(
            "FULL MANIPULATION BENCHMARK RESULTS"
        )

        print(
            "=============================================================="
        )

        print(
            f"Episodes:                         "
            f"{NUM_EPISODES}"
        )

        print(
            f"\nPre-grasp successes:              "
            f"{int(pregrasp_np.sum())}/"
            f"{NUM_EPISODES}"
        )

        print(
            f"Pre-grasp success rate:           "
            f"{pregrasp_rate:.1f}%"
        )

        print(
            f"Mean pre-grasp final error:       "
            f"{mean_pregrasp_error:.4f} m"
        )

        print(
            f"Worst pre-grasp final error:      "
            f"{worst_pregrasp_error:.4f} m"
        )

        print(
            f"Mean RL steps on success:         "
            f"{mean_rl_steps_success:.1f}"
        )

        print(
            f"\nGrasp attempts:                   "
            f"{len(grasp_successes)}"
        )

        print(
            f"Successful grasps:                "
            f"{int(np.sum(grasp_successes))}"
        )

        print(
            f"Grasp success given pre-grasp:    "
            f"{grasp_given_reach_rate:.1f}%"
        )

        print(
            f"\nFull pipeline successes:          "
            f"{int(full_np.sum())}/"
            f"{NUM_EPISODES}"
        )

        print(
            f"FULL PIPELINE SUCCESS RATE:       "
            f"{full_rate:.1f}%"
        )

        print(
            "\nRoute performance:"
        )

        print(
            f"Original-route episodes:          "
            f"{len(original_route_results)}"
        )

        print(
            f"Original-route full success:      "
            f"{original_route_rate:.1f}%"
        )

        print(
            f"Targeted-route episodes:          "
            f"{len(targeted_route_results)}"
        )

        print(
            f"Targeted-route full success:      "
            f"{targeted_route_rate:.1f}%"
        )

        print(
            "\nLift statistics:"
        )

        print(
            f"Mean object lift:                 "
            f"{mean_lift:.4f} m"
        )

        print(
            f"Minimum object lift:              "
            f"{min_lift:.4f} m"
        )

        print(
            f"Maximum object lift:              "
            f"{max_lift:.4f} m"
        )

        print(
            f"\nElapsed:                          "
            f"{elapsed / 60.0:.1f} min"
        )

        print(
            "\nCSV:"
        )

        print(
            OUTPUT_CSV
        )

        print(
            "==============================================================\n"
        )

        # ========================================================
        # Save summary
        # ========================================================

        with open(
            SUMMARY_PATH,
            "w",
        ) as file:

            file.write(
                "Hybrid SAC + Scripted Grasp Benchmark\n"
            )

            file.write(
                "====================================\n\n"
            )

            file.write(
                f"Episodes: "
                f"{NUM_EPISODES}\n\n"
            )

            file.write(
                f"Pre-grasp success: "
                f"{pregrasp_rate:.1f}%\n"
            )

            file.write(
                f"Mean pre-grasp error: "
                f"{mean_pregrasp_error:.4f} m\n"
            )

            file.write(
                f"Worst pre-grasp error: "
                f"{worst_pregrasp_error:.4f} m\n\n"
            )

            file.write(
                f"Grasp success given pre-grasp: "
                f"{grasp_given_reach_rate:.1f}%\n"
            )

            file.write(
                f"Full pipeline success: "
                f"{full_rate:.1f}%\n\n"
            )

            file.write(
                f"Original-route success: "
                f"{original_route_rate:.1f}%\n"
            )

            file.write(
                f"Targeted-route success: "
                f"{targeted_route_rate:.1f}%\n\n"
            )

            file.write(
                f"Mean object lift: "
                f"{mean_lift:.4f} m\n"
            )

            file.write(
                f"Minimum object lift: "
                f"{min_lift:.4f} m\n"
            )

            file.write(
                f"Maximum object lift: "
                f"{max_lift:.4f} m\n"
            )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
