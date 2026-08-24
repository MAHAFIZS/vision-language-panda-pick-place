# -*- coding: utf-8 -*-

"""
train_sac_object_reach.py

Train Soft Actor-Critic for real-object pre-grasp reaching.

Observation:
    26 dimensions

Action:
    [dx, dy, dz]

Task:
    Move Panda EE to a safe pre-grasp goal above a randomly
    positioned real red_box in MuJoCo.
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

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.replay_buffer import ReplayBuffer
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

SEED = 77

TOTAL_STEPS = 30_000

RANDOM_STEPS = 1_000

BATCH_SIZE = 256

REPLAY_CAPACITY = 200_000

UPDATES_PER_STEP = 1

LOG_EVERY_EPISODES = 10

SAVE_EVERY_STEPS = 5_000


OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
)


# ================================================================
# Helpers
# ================================================================

def moving_average(
    values,
    window=20,
):

    if len(values) == 0:
        return 0.0

    return float(
        np.mean(
            values[-window:]
        )
    )


# ================================================================
# Main
# ================================================================

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        "=======================================\n"
        "SAC Object Pre-Grasp Training\n"
        "=======================================\n"
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

    # ============================================================
    # Output
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        OUTPUT_DIR
        / "sac_object_reach_latest.pt"
    )

    best_path = (
        OUTPUT_DIR
        / "sac_object_reach_best.pt"
    )

    final_path = (
        OUTPUT_DIR
        / "sac_object_reach_final.pt"
    )

    csv_path = (
        OUTPUT_DIR
        / "training_log.csv"
    )

    # ============================================================
    # Environment
    # ============================================================

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=120,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=SEED,
    )

    print(
        "Observation dimension:",
        env.observation_dim,
    )

    print(
        "Action dimension:",
        env.action_dim,
    )

    # ============================================================
    # SAC
    # ============================================================

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

    # ============================================================
    # Replay buffer
    # ============================================================

    replay = ReplayBuffer(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        capacity=REPLAY_CAPACITY,
        device=device,
        seed=SEED,
    )

    # ============================================================
    # Metrics
    # ============================================================

    global_step = 0
    episode_index = 0

    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    best_success_rate = 0.0

    start_time = time.time()

    csv_file = open(
        csv_path,
        "w",
        newline="",
    )

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "episode",
            "global_step",
            "episode_reward",
            "episode_length",
            "success",
            "initial_distance",
            "final_distance",
            "moving_reward_20",
            "success_rate_20",
            "alpha",
            "critic_loss",
            "actor_loss",
            "object_x",
            "object_y",
            "object_z",
        ]
    )

    try:

        # ========================================================
        # Training
        # ========================================================

        while global_step < TOTAL_STEPS:

            episode_index += 1

            state, reset_info = (
                env.reset()
            )

            initial_distance = float(
                reset_info[
                    "distance"
                ]
            )

            episode_reward = 0.0
            episode_steps = 0

            success = False

            last_metrics = {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
            }

            info = reset_info

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

                    action = (
                        agent.select_action(
                            state,
                            deterministic=False,
                        )
                    )

                # ------------------------------------------------
                # Environment
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

                # ------------------------------------------------
                # Replay
                # ------------------------------------------------

                replay.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=terminated,
                )

                state = next_state

                episode_reward += reward
                episode_steps += 1
                global_step += 1

                # ------------------------------------------------
                # SAC update
                # ------------------------------------------------

                if (
                    global_step >= RANDOM_STEPS
                    and replay.can_sample(
                        BATCH_SIZE
                    )
                ):

                    for _ in range(
                        UPDATES_PER_STEP
                    ):

                        batch = replay.sample(
                            BATCH_SIZE
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
                        latest_path
                    )

                    print(
                        f"\n[checkpoint] "
                        f"step={global_step}"
                    )

                # ------------------------------------------------
                # Done
                # ------------------------------------------------

                if terminated:

                    success = True
                    break

                if truncated:
                    break

                if global_step >= TOTAL_STEPS:
                    break

            # ====================================================
            # Metrics
            # ====================================================

            episode_rewards.append(
                episode_reward
            )

            episode_successes.append(
                1.0
                if success
                else 0.0
            )

            episode_lengths.append(
                episode_steps
            )

            avg_reward_20 = moving_average(
                episode_rewards,
                20,
            )

            success_rate_20 = moving_average(
                episode_successes,
                20,
            )

            final_distance = float(
                info[
                    "distance"
                ]
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

            obj = np.asarray(
                info[
                    "object_position"
                ],
                dtype=np.float32,
            )

            writer.writerow(
                [
                    episode_index,
                    global_step,
                    episode_reward,
                    episode_steps,
                    int(success),
                    initial_distance,
                    final_distance,
                    avg_reward_20,
                    success_rate_20,
                    alpha_value,
                    critic_loss,
                    actor_loss,
                    float(obj[0]),
                    float(obj[1]),
                    float(obj[2]),
                ]
            )

            csv_file.flush()

            # ====================================================
            # Best model
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
                    best_path
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
                    f"Step: {global_step}/{TOTAL_STEPS}\n"
                    f"Reward: {episode_reward:.3f}\n"
                    f"Length: {episode_steps}\n"
                    f"Success: {success}\n"
                    f"Initial distance: "
                    f"{initial_distance:.4f} m\n"
                    f"Final distance: "
                    f"{final_distance:.4f} m\n"
                    f"Mean reward (20): "
                    f"{avg_reward_20:.3f}\n"
                    f"Success rate (20): "
                    f"{success_rate_20 * 100:.1f}%\n"
                    f"Replay size: "
                    f"{len(replay)}\n"
                    f"Alpha: "
                    f"{alpha_value:.5f}\n"
                    f"Critic loss: "
                    f"{critic_loss:.5f}\n"
                    f"Actor loss: "
                    f"{actor_loss:.5f}\n"
                    f"Elapsed: "
                    f"{elapsed / 60:.1f} min"
                )

            if global_step >= TOTAL_STEPS:
                break

        # ========================================================
        # Save final
        # ========================================================

        agent.save(
            final_path
        )

        print(
            "\n"
            "======================================="
        )

        print(
            "Object-reach SAC training completed."
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
            "Final success rate (20):",
            f"{moving_average(episode_successes, 20) * 100:.1f}%"
        )

        print(
            "Best success rate (20):",
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
