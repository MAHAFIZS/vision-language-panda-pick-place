# -*- coding: utf-8 -*-

"""
train_sac_targeted.py

Failure-aware continuation training for Panda SAC.

Starts from the previously trained best SAC model and oversamples
difficult regions discovered during evaluation.

Sampling:
    70% failure-focused targets
    30% normal uniform targets
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


SEED = 123

TOTAL_STEPS = 15_000

BATCH_SIZE = 256

REPLAY_CAPACITY = 150_000

UPDATES_PER_STEP = 1

LOG_EVERY_EPISODES = 10

SAVE_EVERY_STEPS = 5_000


SOURCE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_reach"
    / "sac_reach_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_reach_targeted"
)


def moving_average(
    values,
    window=20,
):

    if not values:
        return 0.0

    return float(
        np.mean(
            values[-window:]
        )
    )


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
        "Failure-Aware SAC Targeted Retraining\n"
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

    if not SOURCE_MODEL.exists():

        raise FileNotFoundError(
            f"Source model not found:\n"
            f"{SOURCE_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        OUTPUT_DIR
        / "sac_targeted_latest.pt"
    )

    best_path = (
        OUTPUT_DIR
        / "sac_targeted_best.pt"
    )

    final_path = (
        OUTPUT_DIR
        / "sac_targeted_final.pt"
    )

    csv_path = (
        OUTPUT_DIR
        / "targeted_training_log.csv"
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

    env.curriculum_mode = True

    env.hard_target_probability = 0.70

    # ------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------

    config = SACConfig(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-4,
        alpha_lr=1e-4,
        initial_alpha=0.2,
    )

    agent = SACAgent(
        config=config,
        device=device,
    )

    print(
        "\nLoading pretrained model:"
    )

    print(
        SOURCE_MODEL
    )

    agent.load(
        SOURCE_MODEL
    )

    # Lower LR after loading optimizer states.
    for group in agent.actor_optimizer.param_groups:
        group["lr"] = 1e-4

    for group in agent.critic_optimizer.param_groups:
        group["lr"] = 1e-4

    for group in agent.alpha_optimizer.param_groups:
        group["lr"] = 1e-4

    # ------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------

    replay = ReplayBuffer(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        capacity=REPLAY_CAPACITY,
        device=device,
        seed=SEED,
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    global_step = 0
    episode_index = 0

    rewards_history = []
    success_history = []

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
            "reward",
            "length",
            "success",
            "final_distance",
            "success_rate_20",
            "mean_reward_20",
            "alpha",
        ]
    )

    try:

        while global_step < TOTAL_STEPS:

            episode_index += 1

            state, reset_info = env.reset()

            episode_reward = 0.0
            episode_steps = 0
            success = False

            while True:

                action = agent.select_action(
                    state,
                    deterministic=False,
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

                if replay.can_sample(
                    BATCH_SIZE
                ):

                    for _ in range(
                        UPDATES_PER_STEP
                    ):

                        batch = replay.sample(
                            BATCH_SIZE
                        )

                        agent.update(
                            batch
                        )

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

                if terminated:

                    success = True
                    break

                if truncated:
                    break

                if global_step >= TOTAL_STEPS:
                    break

            rewards_history.append(
                episode_reward
            )

            success_history.append(
                1.0 if success else 0.0
            )

            success_rate_20 = moving_average(
                success_history,
                20,
            )

            reward_20 = moving_average(
                rewards_history,
                20,
            )

            alpha = float(
                agent.alpha.detach()
                .cpu()
                .item()
            )

            writer.writerow(
                [
                    episode_index,
                    global_step,
                    episode_reward,
                    episode_steps,
                    int(success),
                    info["distance"],
                    success_rate_20,
                    reward_20,
                    alpha,
                ]
            )

            csv_file.flush()

            if (
                len(success_history) >= 20
                and success_rate_20
                > best_success_rate
            ):

                best_success_rate = (
                    success_rate_20
                )

                agent.save(
                    best_path
                )

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
                    f"Final distance: {info['distance']:.4f} m\n"
                    f"Success rate (20): "
                    f"{success_rate_20 * 100:.1f}%\n"
                    f"Mean reward (20): "
                    f"{reward_20:.3f}\n"
                    f"Alpha: {alpha:.5f}\n"
                    f"Elapsed: "
                    f"{elapsed / 60:.1f} min"
                )

            if global_step >= TOTAL_STEPS:
                break

        agent.save(
            final_path
        )

        print(
            "\n"
            "======================================="
        )

        print(
            "Targeted retraining completed."
        )

        print(
            "Environment steps:",
            global_step,
        )

        print(
            "Episodes:",
            episode_index,
        )

        print(
            "Final success rate (20):",
            f"{moving_average(success_history, 20) * 100:.1f}%"
        )

        print(
            "Best success rate (20):",
            f"{best_success_rate * 100:.1f}%"
        )

        print(
            "Final model:",
            final_path
        )

        print(
            "=======================================\n"
        )

    finally:

        csv_file.close()
        env.close()


if __name__ == "__main__":
    main()
