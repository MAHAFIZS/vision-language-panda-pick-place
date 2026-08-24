# -*- coding: utf-8 -*-

"""
evaluate_sac_targeted.py

Evaluate the failure-aware targeted SAC model on
uniform unseen Panda reaching targets.

This intentionally uses the ORIGINAL uniform environment.

Goal:
Compare targeted retraining against the original SAC baseline.

Original baseline:
    83 / 100 success
    83.0%
    mean final distance = 0.0399 m
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# IMPORTANT:
# Use original uniform environment for fair comparison.
from rl.panda_rl_env import PandaReachEnv

from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Evaluation configuration
# ================================================================

NUM_EPISODES = 100

# Use the SAME seed as the previous baseline evaluation.
#
# This means both models see the exact same target sequence.
EVAL_SEED = 2026

MAX_STEPS = 120


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_reach_targeted"
    / "sac_targeted_best.pt"
)


FALLBACK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_reach_targeted"
    / "sac_targeted_final.pt"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_reach_targeted"
    / "evaluation_uniform_100.csv"
)


# ================================================================
# Main
# ================================================================

def main() -> None:

    print(
        "\n"
        "=============================================\n"
        "Targeted SAC - Uniform Unseen Evaluation\n"
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
            "\nBest targeted model not found."
        )

        print(
            "Using final targeted model."
        )

    else:

        raise FileNotFoundError(
            "No targeted SAC model found.\n"
            f"{MODEL_PATH}\n"
            f"{FALLBACK_MODEL_PATH}"
        )

    print(
        "Model:",
        model_path,
    )

    # ------------------------------------------------------------
    # ORIGINAL uniform environment
    # ------------------------------------------------------------

    env = PandaReachEnv(
        max_steps=MAX_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.035,
        seed=EVAL_SEED,
    )

    # ------------------------------------------------------------
    # SAC
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

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    successes = []

    final_distances = []

    initial_distances = []

    episode_steps = []

    successful_steps = []

    rewards = []

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
            "target_x",
            "target_y",
            "target_z",
            "final_ee_x",
            "final_ee_y",
            "final_ee_z",
        ]
    )

    try:

        # ========================================================
        # Evaluation
        # ========================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            state, reset_info = (
                env.reset()
            )

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
            # Episode
            # ----------------------------------------------------

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                # Deterministic SAC policy
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

            target = np.asarray(
                final_info[
                    "target"
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

            rewards.append(
                total_reward
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
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    float(ee[0]),
                    float(ee[1]),
                    float(ee[2]),
                ]
            )

            csv_file.flush()

            # ----------------------------------------------------
            # Progress
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
                    f"initial="
                    f"{initial_distance:.4f} m | "
                    f"final="
                    f"{final_distance:.4f} m | "
                    f"steps="
                    f"{steps_taken:03d} | "
                    f"running success="
                    f"{running_success:.1f}%"
                )

        # ========================================================
        # Summary
        # ========================================================

        successes_np = np.asarray(
            successes,
            dtype=np.float32,
        )

        final_distances_np = np.asarray(
            final_distances,
            dtype=np.float32,
        )

        initial_distances_np = np.asarray(
            initial_distances,
            dtype=np.float32,
        )

        steps_np = np.asarray(
            episode_steps,
            dtype=np.float32,
        )

        rewards_np = np.asarray(
            rewards,
            dtype=np.float32,
        )

        success_rate = float(
            successes_np.mean()
            * 100.0
        )

        mean_final_distance = float(
            final_distances_np.mean()
        )

        median_final_distance = float(
            np.median(
                final_distances_np
            )
        )

        std_final_distance = float(
            final_distances_np.std()
        )

        best_distance = float(
            final_distances_np.min()
        )

        worst_distance = float(
            final_distances_np.max()
        )

        mean_initial_distance = float(
            initial_distances_np.mean()
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
        # Print result
        # ========================================================

        print(
            "\n"
            "=============================================\n"
            "Targeted SAC Uniform Evaluation Results\n"
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
            f"{mean_initial_distance:.4f} m"
        )

        print(
            f"Mean final distance:      "
            f"{mean_final_distance:.4f} m"
        )

        print(
            f"Median final distance:    "
            f"{median_final_distance:.4f} m"
        )

        print(
            f"Std final distance:       "
            f"{std_final_distance:.4f} m"
        )

        print(
            f"Best final distance:      "
            f"{best_distance:.4f} m"
        )

        print(
            f"Worst final distance:     "
            f"{worst_distance:.4f} m"
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
            f"\nSuccess threshold: "
            f"{env.success_threshold:.3f} m"
        )

        # --------------------------------------------------------
        # Direct comparison to old model
        # --------------------------------------------------------

        print(
            "\n"
            "---------------------------------------------\n"
            "Comparison with Original SAC\n"
            "---------------------------------------------"
        )

        print(
            "Original SAC success:      "
            "83.0%"
        )

        print(
            "Targeted SAC success:      "
            f"{success_rate:.1f}%"
        )

        improvement = (
            success_rate
            - 83.0
        )

        print(
            "Absolute improvement:      "
            f"{improvement:+.1f} percentage points"
        )

        print(
            "\nOriginal mean error:       "
            "0.0399 m"
        )

        print(
            "Targeted mean error:       "
            f"{mean_final_distance:.4f} m"
        )

        error_improvement = (
            0.0399
            - mean_final_distance
        )

        print(
            "Mean error improvement:    "
            f"{error_improvement:+.4f} m"
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
