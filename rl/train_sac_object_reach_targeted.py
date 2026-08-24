# -*- coding: utf-8 -*-

"""
train_sac_object_reach_targeted.py

Failure-aware continuation training for object pre-grasp SAC.

Starts from:
    models/sac_object_reach/sac_object_reach_best.pt

Uses:
    PandaObjectReachTargetedEnv

Sampling:
    70% hard negative-Y object placements
    30% uniform placements

Goal:
    Improve reliability in the failure region while preserving
    general object-position performance.
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

from rl.panda_object_reach_env_targeted import PandaObjectReachTargetedEnv
from rl.replay_buffer import ReplayBuffer
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

SEED = 808

TOTAL_STEPS = 15_000

BATCH_SIZE = 256

REPLAY_CAPACITY = 150_000

UPDATES_PER_STEP = 1

LOG_EVERY_EPISODES = 10

SAVE_EVERY_STEPS = 5_000


SOURCE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "sac_object_reach_best.pt"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
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
        "\n"
        "=============================================\n"
        "Failure-Aware Object-Reach SAC Retraining\n"
        "=============================================\n"
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
    # Check source model
    # ------------------------------------------------------------

    if not SOURCE_MODEL.exists():

        raise FileNotFoundError(
            f"Source model not found:\n"
            f"{SOURCE_MODEL}"
        )

    # ------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        OUTPUT_DIR
        / "sac_object_targeted_latest.pt"
    )

    best_path = (
        OUTPUT_DIR
        / "sac_object_targeted_best.pt"
    )

    final_path = (
        OUTPUT_DIR
        / "sac_object_targeted_final.pt"
    )

    csv_path = (
        OUTPUT_DIR
        / "targeted_training_log.csv"
    )

    # ============================================================
    # Environment
    # ============================================================

    env = PandaObjectReachTargetedEnv(
        object_name="red_box",
        max_steps=120,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        hard_target_probability=0.70,
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

        # Lower continuation learning rates.
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
        "\nLoading pretrained object-reaching model:"
    )

    print(
        SOURCE_MODEL
    )

    agent.load(
        SOURCE_MODEL
    )

    # ------------------------------------------------------------
    # Important:
    # Loading checkpoint also restores old optimizer LR.
    # Force continuation LR back to 1e-4.
    # ------------------------------------------------------------

    for group in (
        agent.actor_optimizer.param_groups
    ):

        group["lr"] = 1e-4

    for group in (
        agent.critic_optimizer.param_groups
    ):

        group["lr"] = 1e-4

    for group in (
        agent.alpha_optimizer.param_groups
    ):

        group["lr"] = 1e-4

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

    hard_successes = []

    uniform_successes = []

    best_success_rate = 0.0

    start_time = time.time()

    # ============================================================
    # CSV
    # ============================================================

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
            "mode",
            "reward",
            "length",
            "success",
            "initial_distance",
            "final_distance",
            "success_rate_20",
            "mean_reward_20",
            "hard_success_rate",
            "uniform_success_rate",
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
        # Training loop
        # ========================================================

        while global_step < TOTAL_STEPS:

            episode_index += 1

            state, reset_info = (
                env.reset()
            )

            mode = reset_info[
                "object_mode"
            ]

            initial_distance = float(
                reset_info[
                    "distance"
                ]
            )

            episode_reward = 0.0

            episode_steps = 0

            success = False

            info = reset_info

            last_metrics = {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
            }

            # ====================================================
            # Episode
            # ====================================================

            while True:

                # ------------------------------------------------
                # Continue with stochastic SAC policy
                # ------------------------------------------------

                action = (
                    agent.select_action(
                        state,
                        deterministic=False,
                    )
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

                # ------------------------------------------------
                # Replay
                # ------------------------------------------------

                replay.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,

                    # Time-limit truncation does not become
                    # true terminal state.
                    done=terminated,
                )

                state = next_state

                episode_reward += reward

                episode_steps += 1

                global_step += 1

                # ------------------------------------------------
                # SAC updates
                # ------------------------------------------------

                if replay.can_sample(
                    BATCH_SIZE
                ):

                    for _ in range(
                        UPDATES_PER_STEP
                    ):

                        batch = (
                            replay.sample(
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
                        latest_path
                    )

                    print(
                        f"\n[checkpoint] "
                        f"step={global_step}"
                    )

                # ------------------------------------------------
                # End conditions
                # ------------------------------------------------

                if terminated:

                    success = True

                    break

                if truncated:

                    break

                if global_step >= TOTAL_STEPS:

                    break

            # ====================================================
            # Episode statistics
            # ====================================================

            episode_rewards.append(
                episode_reward
            )

            episode_successes.append(
                1.0
                if success
                else 0.0
            )

            if (
                mode
                == "hard_negative_y"
            ):

                hard_successes.append(
                    1.0
                    if success
                    else 0.0
                )

            else:

                uniform_successes.append(
                    1.0
                    if success
                    else 0.0
                )

            success_rate_20 = (
                moving_average(
                    episode_successes,
                    20,
                )
            )

            reward_20 = (
                moving_average(
                    episode_rewards,
                    20,
                )
            )

            hard_rate = (
                float(
                    np.mean(
                        hard_successes
                    )
                )
                if hard_successes
                else 0.0
            )

            uniform_rate = (
                float(
                    np.mean(
                        uniform_successes
                    )
                )
                if uniform_successes
                else 0.0
            )

            final_distance = float(
                info[
                    "distance"
                ]
            )

            alpha = float(
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

            # ====================================================
            # CSV row
            # ====================================================

            writer.writerow(
                [
                    episode_index,
                    global_step,
                    mode,
                    episode_reward,
                    episode_steps,
                    int(success),
                    initial_distance,
                    final_distance,
                    success_rate_20,
                    reward_20,
                    hard_rate,
                    uniform_rate,
                    alpha,
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
                    f"Step: "
                    f"{global_step}/{TOTAL_STEPS}\n"
                    f"Mode: {mode}\n"
                    f"Reward: "
                    f"{episode_reward:.3f}\n"
                    f"Length: "
                    f"{episode_steps}\n"
                    f"Success: "
                    f"{success}\n"
                    f"Initial distance: "
                    f"{initial_distance:.4f} m\n"
                    f"Final distance: "
                    f"{final_distance:.4f} m\n"
                    f"Success rate (20): "
                    f"{success_rate_20 * 100:.1f}%\n"
                    f"Hard cumulative success: "
                    f"{hard_rate * 100:.1f}%\n"
                    f"Uniform cumulative success: "
                    f"{uniform_rate * 100:.1f}%\n"
                    f"Mean reward (20): "
                    f"{reward_20:.3f}\n"
                    f"Replay size: "
                    f"{len(replay)}\n"
                    f"Alpha: "
                    f"{alpha:.5f}\n"
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
        # Final save
        # ========================================================

        agent.save(
            final_path
        )

        print(
            "\n"
            "============================================="
        )

        print(
            "Targeted object-reach retraining completed."
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
            f"{moving_average(episode_successes, 20) * 100:.1f}%"
        )

        print(
            "Best success rate (20):",
            f"{best_success_rate * 100:.1f}%"
        )

        if hard_successes:

            print(
                "Hard cumulative success:",
                f"{np.mean(hard_successes) * 100:.1f}%"
            )

        if uniform_successes:

            print(
                "Uniform cumulative success:",
                f"{np.mean(uniform_successes) * 100:.1f}%"
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
            "=============================================\n"
        )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
