# -*- coding: utf-8 -*-

"""
evaluate_sac_object_reach.py

Deterministic evaluation for the trained object-relative SAC policy.

Task:
    Move the Panda end-effector to a pre-grasp point above a
    randomly positioned real red_box in MuJoCo.

Reports:
    - success rate
    - mean final pre-grasp error
    - median error
    - standard deviation
    - best / worst error
    - mean steps
    - mean steps on successful trials
    - mean reward

The object positions are unseen because evaluation uses a
different random seed from training.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


# ================================================================
# Project import
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100

# Different from training seed 77.
EVAL_SEED = 2027

MAX_STEPS = 120

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "sac_object_reach_best.pt"
)

FALLBACK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "sac_object_reach_final.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "evaluation_100_unseen_objects.csv"
)


# ================================================================
# Main
# ================================================================

def main() -> None:

    print(
        "\n"
        "=============================================\n"
        "SAC Object Pre-Grasp Deterministic Evaluation\n"
        "=============================================\n"
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
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ------------------------------------------------------------
    # Select model
    # ------------------------------------------------------------

    if MODEL_PATH.exists():

        model_path = MODEL_PATH

    elif FALLBACK_MODEL_PATH.exists():

        model_path = FALLBACK_MODEL_PATH

        print(
            "\nBest model not found."
        )

        print(
            "Using final model instead."
        )

    else:

        raise FileNotFoundError(
            "No object-reaching SAC model found.\n"
            f"{MODEL_PATH}\n"
            f"{FALLBACK_MODEL_PATH}"
        )

    print(
        "Model:",
        model_path,
    )

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=EVAL_SEED,
    )

    # ------------------------------------------------------------
    # SAC agent
    # ------------------------------------------------------------

    config = SACConfig(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
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

    # ============================================================
    # Metrics
    # ============================================================

    successes = []

    initial_distances = []

    final_distances = []

    episode_steps = []

    successful_steps = []

    episode_rewards = []

    object_positions = []

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = open(
        OUTPUT_PATH,
        "w",
        newline="",
    )

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "episode",
            "success",
            "initial_distance_m",
            "final_distance_m",
            "steps",
            "episode_reward",
            "object_x",
            "object_y",
            "object_z",
            "goal_x",
            "goal_y",
            "goal_z",
            "final_ee_x",
            "final_ee_y",
            "final_ee_z",
        ]
    )

    try:

        # ========================================================
        # Episodes
        # ========================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            state, reset_info = env.reset()

            initial_distance = float(
                reset_info[
                    "distance"
                ]
            )

            total_reward = 0.0

            success = False

            steps_taken = 0

            final_info = reset_info

            # ----------------------------------------------------
            # Deterministic rollout
            # ----------------------------------------------------

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                action = agent.select_action(
                    state,
                    deterministic=True,
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                state = next_state

                total_reward += reward

                steps_taken = step

                final_info = info

                if terminated:

                    success = True
                    break

                if truncated:

                    break

            # ----------------------------------------------------
            # Episode metrics
            # ----------------------------------------------------

            final_distance = float(
                final_info[
                    "distance"
                ]
            )

            obj = np.asarray(
                final_info[
                    "object_position"
                ],
                dtype=np.float32,
            )

            goal = np.asarray(
                final_info[
                    "goal_position"
                ],
                dtype=np.float32,
            )

            ee = np.asarray(
                final_info[
                    "ee_position"
                ],
                dtype=np.float32,
            )

            successes.append(
                1 if success else 0
            )

            initial_distances.append(
                initial_distance
            )

            final_distances.append(
                final_distance
            )

            episode_steps.append(
                steps_taken
            )

            episode_rewards.append(
                total_reward
            )

            object_positions.append(
                obj.copy()
            )

            if success:

                successful_steps.append(
                    steps_taken
                )

            writer.writerow(
                [
                    episode,
                    int(success),
                    initial_distance,
                    final_distance,
                    steps_taken,
                    total_reward,
                    float(obj[0]),
                    float(obj[1]),
                    float(obj[2]),
                    float(goal[0]),
                    float(goal[1]),
                    float(goal[2]),
                    float(ee[0]),
                    float(ee[1]),
                    float(ee[2]),
                ]
            )

            csv_file.flush()

            # ----------------------------------------------------
            # Progress output
            # ----------------------------------------------------

            if (
                episode <= 10
                or episode % 10 == 0
            ):

                running_success = (
                    100.0
                    * np.mean(
                        successes
                    )
                )

                print(
                    f"Episode "
                    f"{episode:03d}/{NUM_EPISODES} | "
                    f"success={success} | "
                    f"object="
                    f"{np.round(obj[:2], 3)} | "
                    f"initial="
                    f"{initial_distance:.4f} m | "
                    f"final="
                    f"{final_distance:.4f} m | "
                    f"steps="
                    f"{steps_taken:03d} | "
                    f"running="
                    f"{running_success:.1f}%"
                )

        # ========================================================
        # Aggregate statistics
        # ========================================================

        successes_np = np.asarray(
            successes,
            dtype=np.float32,
        )

        initial_np = np.asarray(
            initial_distances,
            dtype=np.float32,
        )

        final_np = np.asarray(
            final_distances,
            dtype=np.float32,
        )

        steps_np = np.asarray(
            episode_steps,
            dtype=np.float32,
        )

        rewards_np = np.asarray(
            episode_rewards,
            dtype=np.float32,
        )

        objects_np = np.asarray(
            object_positions,
            dtype=np.float32,
        )

        success_rate = float(
            successes_np.mean()
            * 100.0
        )

        mean_initial = float(
            initial_np.mean()
        )

        mean_final = float(
            final_np.mean()
        )

        median_final = float(
            np.median(
                final_np
            )
        )

        std_final = float(
            final_np.std()
        )

        best_final = float(
            final_np.min()
        )

        worst_final = float(
            final_np.max()
        )

        mean_steps_all = float(
            steps_np.mean()
        )

        if successful_steps:

            mean_steps_success = float(
                np.mean(
                    successful_steps
                )
            )

        else:

            mean_steps_success = float(
                "nan"
            )

        mean_reward = float(
            rewards_np.mean()
        )

        # ========================================================
        # Print results
        # ========================================================

        print(
            "\n"
            "=============================================\n"
            "Object Pre-Grasp Evaluation Results\n"
            "============================================="
        )

        print(
            f"Episodes:                 "
            f"{NUM_EPISODES}"
        )

        print(
            f"Successes:                "
            f"{int(successes_np.sum())}"
        )

        print(
            f"Failures:                 "
            f"{NUM_EPISODES - int(successes_np.sum())}"
        )

        print(
            f"Success rate:             "
            f"{success_rate:.1f}%"
        )

        print(
            f"Mean initial distance:    "
            f"{mean_initial:.4f} m"
        )

        print(
            f"Mean final distance:      "
            f"{mean_final:.4f} m"
        )

        print(
            f"Median final distance:    "
            f"{median_final:.4f} m"
        )

        print(
            f"Std final distance:       "
            f"{std_final:.4f} m"
        )

        print(
            f"Best final distance:      "
            f"{best_final:.4f} m"
        )

        print(
            f"Worst final distance:     "
            f"{worst_final:.4f} m"
        )

        print(
            f"Mean steps (all):         "
            f"{mean_steps_all:.1f}"
        )

        print(
            f"Mean steps (success):     "
            f"{mean_steps_success:.1f}"
        )

        print(
            f"Mean episode reward:      "
            f"{mean_reward:.3f}"
        )

        print(
            "\nObject XY evaluation range:"
        )

        print(
            f"X: "
            f"{objects_np[:, 0].min():.4f} "
            f"to "
            f"{objects_np[:, 0].max():.4f} m"
        )

        print(
            f"Y: "
            f"{objects_np[:, 1].min():.4f} "
            f"to "
            f"{objects_np[:, 1].max():.4f} m"
        )

        print(
            f"\nSuccess threshold: "
            f"{env.success_threshold:.3f} m"
        )

        print(
            f"Pre-grasp height: "
            f"{env.pregrasp_height:.3f} m"
        )

        print(
            "\nResults saved:"
        )

        print(
            OUTPUT_PATH
        )

        print(
            "=============================================\n"
        )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
