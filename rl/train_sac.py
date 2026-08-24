# -*- coding: utf-8 -*-

"""
train_sac.py

Train Soft Actor-Critic on the PandaReachEnv.

Pipeline:

PandaReachEnv
    ↓
SAC Actor
    ↓
MuJoCo interaction
    ↓
Replay Buffer
    ↓
SAC updates on GPU
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_rl_env import PandaReachEnv
from rl.replay_buffer import ReplayBuffer
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Training configuration
# ================================================================

SEED = 42

TOTAL_STEPS = 30_000

RANDOM_STEPS = 1_000

BATCH_SIZE = 256

REPLAY_CAPACITY = 200_000

UPDATES_PER_STEP = 1

LOG_EVERY_EPISODES = 10

SAVE_EVERY_STEPS = 5_000


# ================================================================
# Helpers
# ================================================================

def moving_average(
    values,
    window=20,
):

    if len(values) == 0:
        return 0.0

    values = values[
        -window:
    ]

    return float(
        np.mean(values)
    )


# ================================================================
# Main
# ================================================================

def main():

    # ------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n--- SAC Panda Reach Training ---\n"
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
    # Output paths
    # ------------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "models"
        / "sac_reach"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "sac_reach_latest.pt"
    )

    best_checkpoint_path = (
        output_dir
        / "sac_reach_best.pt"
    )

    csv_path = (
        output_dir
        / "training_log.csv"
    )

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    env = PandaReachEnv(
        max_steps=120,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.035,
        seed=SEED,
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

    # ------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------

    replay_buffer = ReplayBuffer(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        capacity=REPLAY_CAPACITY,
        device=device,
        seed=SEED,
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    best_success_rate = 0.0

    global_step = 0
    episode_index = 0

    start_time = time.time()

    # ------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------

    csv_file = open(
        csv_path,
        "w",
        newline="",
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        [
            "episode",
            "global_step",
            "episode_reward",
            "episode_length",
            "success",
            "final_distance",
            "moving_reward_20",
            "success_rate_20",
            "alpha",
            "critic_loss",
            "actor_loss",
        ]
    )

    try:

        # ========================================================
        # Training loop
        # ========================================================

        while global_step < TOTAL_STEPS:

            episode_index += 1

            state, reset_info = env.reset()

            episode_reward = 0.0
            episode_steps = 0

            success = False

            last_metrics = {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
            }

            # ----------------------------------------------------
            # Episode
            # ----------------------------------------------------

            while True:

                # ------------------------------------------------
                # Exploration
                # ------------------------------------------------

                if global_step < RANDOM_STEPS:

                    action = (
                        env.sample_random_action()
                    )

                else:

                    action = agent.select_action(
                        state,
                        deterministic=False,
                    )

                # ------------------------------------------------
                # Environment interaction
                # ------------------------------------------------

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                done = bool(
                    terminated
                )

                # ------------------------------------------------
                # Replay storage
                #
                # IMPORTANT:
                #
                # For bootstrapping, true task success is terminal.
                # Time-limit truncation is not treated as an
                # environment terminal state.
                # ------------------------------------------------

                replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                state = next_state

                episode_reward += reward
                episode_steps += 1
                global_step += 1

                # ------------------------------------------------
                # SAC update
                # ------------------------------------------------

                if (
                    global_step
                    >= RANDOM_STEPS
                    and replay_buffer.can_sample(
                        BATCH_SIZE
                    )
                ):

                    for _ in range(
                        UPDATES_PER_STEP
                    ):

                        batch = (
                            replay_buffer.sample(
                                BATCH_SIZE
                            )
                        )

                        last_metrics = (
                            agent.update(
                                batch
                            )
                        )

                # ------------------------------------------------
                # Checkpoint
                # ------------------------------------------------

                if (
                    global_step
                    % SAVE_EVERY_STEPS
                    == 0
                ):

                    agent.save(
                        checkpoint_path
                    )

                    print(
                        f"\n[checkpoint] "
                        f"step={global_step} "
                        f"saved={checkpoint_path}"
                    )

                # ------------------------------------------------
                # Episode end
                # ------------------------------------------------

                if terminated:

                    success = True
                    break

                if truncated:
                    break

                if global_step >= TOTAL_STEPS:
                    break

            # ====================================================
            # Episode metrics
            # ====================================================

            episode_rewards.append(
                episode_reward
            )

            episode_successes.append(
                1.0 if success else 0.0
            )

            episode_lengths.append(
                episode_steps
            )

            avg_reward_20 = moving_average(
                episode_rewards,
                window=20,
            )

            success_rate_20 = moving_average(
                episode_successes,
                window=20,
            )

            final_distance = float(
                info["distance"]
            )

            alpha_value = float(
                agent.alpha.detach()
                .cpu()
                .item()
            )

            critic_loss = float(
                last_metrics.get(
                    "critic_loss",
                    0.0,
                )
            )

            actor_loss = float(
                last_metrics.get(
                    "actor_loss",
                    0.0,
                )
            )

            csv_writer.writerow(
                [
                    episode_index,
                    global_step,
                    episode_reward,
                    episode_steps,
                    int(success),
                    final_distance,
                    avg_reward_20,
                    success_rate_20,
                    alpha_value,
                    critic_loss,
                    actor_loss,
                ]
            )

            csv_file.flush()

            # ====================================================
            # Best checkpoint
            # ====================================================

            if (
                len(
                    episode_successes
                )
                >= 20
                and success_rate_20
                > best_success_rate
            ):

                best_success_rate = (
                    success_rate_20
                )

                agent.save(
                    best_checkpoint_path
                )

            # ====================================================
            # Logging
            # ====================================================

            if (
                episode_index
                % LOG_EVERY_EPISODES
                == 0
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    "\n"
                    f"Episode: {episode_index}\n"
                    f"Global step: {global_step}/{TOTAL_STEPS}\n"
                    f"Episode reward: {episode_reward:.3f}\n"
                    f"Episode length: {episode_steps}\n"
                    f"Success: {success}\n"
                    f"Final distance: {final_distance:.4f} m\n"
                    f"Mean reward (20): {avg_reward_20:.3f}\n"
                    f"Success rate (20): "
                    f"{success_rate_20 * 100:.1f}%\n"
                    f"Replay size: {len(replay_buffer)}\n"
                    f"Alpha: {alpha_value:.4f}\n"
                    f"Critic loss: {critic_loss:.5f}\n"
                    f"Actor loss: {actor_loss:.5f}\n"
                    f"Elapsed: {elapsed / 60.0:.1f} min"
                )

            # ====================================================
            # Training finished
            # ====================================================

            if global_step >= TOTAL_STEPS:
                break

        # ========================================================
        # Final save
        # ========================================================

        final_path = (
            output_dir
            / "sac_reach_final.pt"
        )

        agent.save(
            final_path
        )

        print(
            "\n"
            "======================================="
        )

        print(
            "Training completed."
        )

        print(
            "Total environment steps:",
            global_step,
        )

        print(
            "Episodes:",
            episode_index,
        )

        print(
            "Final 20-episode success rate:",
            f"{moving_average(episode_successes, 20) * 100:.1f}%"
        )

        print(
            "Best 20-episode success rate:",
            f"{best_success_rate * 100:.1f}%"
        )

        print(
            "Final model:",
            final_path,
        )

        print(
            "Training log:",
            csv_path,
        )

        print(
            "=======================================\n"
        )

    finally:

        csv_file.close()
        env.close()


if __name__ == "__main__":
    main()
