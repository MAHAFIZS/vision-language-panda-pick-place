# -*- coding: utf-8 -*-

"""
evaluate_sac.py

Deterministic evaluation for the trained SAC Panda reaching policy.

Evaluates the learned policy on unseen randomly sampled targets and reports:

- success rate
- mean final distance
- median final distance
- standard deviation
- minimum / maximum final distance
- mean steps to success
- mean episode reward

The evaluation uses deterministic actor actions:
    action = tanh(mean)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_rl_env import PandaReachEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100

# Use a different seed from training.
# Training used 42, so this gives us another deterministic target set.
EVAL_SEED = 2026

MAX_STEPS = 120

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_reach"
    / "sac_reach_best.pt"
)

# Fallback if best checkpoint is unavailable.
FALLBACK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_reach"
    / "sac_reach_final.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "sac_reach"
    / "evaluation_100_unseen.csv"
)


# ================================================================
# Main
# ================================================================

def main() -> None:

    print(
        "\n"
        "=======================================\n"
        "SAC Panda Reach Deterministic Evaluation\n"
        "=======================================\n"
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

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
            "\nBest checkpoint not found."
        )

        print(
            "Using final checkpoint instead."
        )

    else:
        raise FileNotFoundError(
            "No trained SAC model found.\n"
            f"Checked:\n"
            f"  {MODEL_PATH}\n"
            f"  {FALLBACK_MODEL_PATH}"
        )

    print(
        "Model:",
        model_path,
    )

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    env = PandaReachEnv(
        max_steps=MAX_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.035,
        seed=EVAL_SEED,
    )

    # ------------------------------------------------------------
    # Agent
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
    # Result storage
    # ------------------------------------------------------------

    successes = []
    final_distances = []
    episode_steps_list = []
    successful_steps = []
    episode_rewards = []
    initial_distances = []

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
        # Evaluation episodes
        # ========================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            state, reset_info = env.reset()

            initial_distance = float(
                reset_info["distance"]
            )

            initial_distances.append(
                initial_distance
            )

            total_reward = 0.0

            success = False

            final_info = reset_info

            steps_taken = 0

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                # ------------------------------------------------
                # Deterministic SAC action
                # ------------------------------------------------

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

                total_reward += reward

                state = next_state

                final_info = info
                steps_taken = step

                if terminated:
                    success = True
                    break

                if truncated:
                    break

            # ----------------------------------------------------
            # Episode result
            # ----------------------------------------------------

            final_distance = float(
                final_info["distance"]
            )

            target = np.asarray(
                final_info["target"],
                dtype=np.float32,
            )

            final_ee = np.asarray(
                final_info["ee_position"],
                dtype=np.float32,
            )

            successes.append(
                1 if success else 0
            )

            final_distances.append(
                final_distance
            )

            episode_steps_list.append(
                steps_taken
            )

            episode_rewards.append(
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
                    float(final_ee[0]),
                    float(final_ee[1]),
                    float(final_ee[2]),
                ]
            )

            csv_file.flush()

            # ----------------------------------------------------
            # Progress display
            # ----------------------------------------------------

            if (
                episode <= 10
                or episode % 10 == 0
            ):

                running_success_rate = (
                    100.0
                    * np.mean(successes)
                )

                print(
                    f"Episode {episode:03d}/{NUM_EPISODES} | "
                    f"success={success} | "
                    f"initial={initial_distance:.4f} m | "
                    f"final={final_distance:.4f} m | "
                    f"steps={steps_taken:03d} | "
                    f"running success={running_success_rate:.1f}%"
                )

        # ========================================================
        # Summary
        # ========================================================

        successes_np = np.asarray(
            successes,
            dtype=np.float32,
        )

        distances_np = np.asarray(
            final_distances,
            dtype=np.float32,
        )

        rewards_np = np.asarray(
            episode_rewards,
            dtype=np.float32,
        )

        episode_steps_np = np.asarray(
            episode_steps_list,
            dtype=np.float32,
        )

        initial_distances_np = np.asarray(
            initial_distances,
            dtype=np.float32,
        )

        success_rate = float(
            successes_np.mean()
            * 100.0
        )

        mean_final_distance = float(
            distances_np.mean()
        )

        median_final_distance = float(
            np.median(
                distances_np
            )
        )

        std_final_distance = float(
            distances_np.std()
        )

        min_final_distance = float(
            distances_np.min()
        )

        max_final_distance = float(
            distances_np.max()
        )

        mean_episode_reward = float(
            rewards_np.mean()
        )

        mean_steps_all = float(
            episode_steps_np.mean()
        )

        mean_initial_distance = float(
            initial_distances_np.mean()
        )

        if len(successful_steps) > 0:

            mean_steps_success = float(
                np.mean(
                    successful_steps
                )
            )

        else:

            mean_steps_success = float(
                "nan"
            )

        print(
            "\n"
            "=======================================\n"
            "Evaluation Results\n"
            "======================================="
        )

        print(
            f"Episodes:                 {NUM_EPISODES}"
        )

        print(
            f"Successes:                {int(successes_np.sum())}"
        )

        print(
            f"Failures:                 {NUM_EPISODES - int(successes_np.sum())}"
        )

        print(
            f"Success rate:             {success_rate:.1f}%"
        )

        print(
            f"Mean initial distance:    {mean_initial_distance:.4f} m"
        )

        print(
            f"Mean final distance:      {mean_final_distance:.4f} m"
        )

        print(
            f"Median final distance:    {median_final_distance:.4f} m"
        )

        print(
            f"Std final distance:       {std_final_distance:.4f} m"
        )

        print(
            f"Best final distance:      {min_final_distance:.4f} m"
        )

        print(
            f"Worst final distance:     {max_final_distance:.4f} m"
        )

        print(
            f"Mean steps (all):         {mean_steps_all:.1f}"
        )

        print(
            f"Mean steps (success):     {mean_steps_success:.1f}"
        )

        print(
            f"Mean episode reward:      {mean_episode_reward:.3f}"
        )

        print(
            "\nSuccess threshold:",
            f"{env.success_threshold:.3f} m"
        )

        print(
            "\nResults saved to:"
        )

        print(
            OUTPUT_PATH
        )

        print(
            "=======================================\n"
        )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
