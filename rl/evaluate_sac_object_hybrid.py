# -*- coding: utf-8 -*-

"""
evaluate_sac_object_hybrid.py

Hybrid failure-aware policy router.

Routing rule:
    object_y < -0.10  -> targeted SAC
    object_y >= -0.10 -> original SAC

Evaluation:
    100 identical uniform unseen object placements

Goal:
    Test whether combining the original general policy with the
    failure-specialized policy improves overall performance.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100
EVAL_SEED = 2027
MAX_STEPS = 120

ROUTER_Y_THRESHOLD = -0.10

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
    / "hybrid"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "hybrid_uniform_100.csv"
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
# Generate exact uniform targets
# ================================================================

def generate_uniform_targets(
    count: int,
    seed: int,
):

    rng = np.random.default_rng(seed)

    targets = []

    for _ in range(count):

        x = rng.uniform(
            0.40,
            0.60,
        )

        y = rng.uniform(
            -0.20,
            0.20,
        )

        targets.append(
            (
                float(x),
                float(y),
            )
        )

    return targets


# ================================================================
# Force object XY
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

    current_z = float(
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
    ] = current_z

    # Identity quaternion
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
# Main
# ================================================================

def main():

    print(
        "\n"
        "============================================================\n"
        "Failure-Aware Hybrid SAC Evaluation\n"
        "============================================================\n"
    )

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

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=12345,
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

    print(
        "\nOriginal model:",
        ORIGINAL_MODEL,
    )

    print(
        "Targeted model:",
        TARGETED_MODEL,
    )

    print(
        f"\nRouter rule: "
        f"object_y < {ROUTER_Y_THRESHOLD:.2f} "
        f"-> targeted SAC"
    )

    print(
        f"             "
        f"object_y >= {ROUTER_Y_THRESHOLD:.2f} "
        f"-> original SAC"
    )

    targets = generate_uniform_targets(
        NUM_EPISODES,
        EVAL_SEED,
    )

    successes = []
    final_distances = []
    initial_distances = []
    episode_steps = []
    successful_steps = []
    episode_rewards = []

    targeted_count = 0
    original_count = 0

    targeted_successes = []
    original_successes = []

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
            "target_x",
            "target_y",
            "selected_policy",
            "success",
            "initial_distance",
            "final_distance",
            "steps",
            "reward",
            "final_ee_x",
            "final_ee_y",
            "final_ee_z",
        ]
    )

    try:

        for episode, (
            target_x,
            target_y,
        ) in enumerate(
            targets,
            start=1,
        ):

            state, info = env.reset()

            set_object_xy(
                env,
                target_x,
                target_y,
            )

            ee = env._ee_position()

            env.robot.target_pos = (
                ee.astype(float)
            )

            env.robot.target_quat = (
                env.robot.home_quat.copy()
            )

            goal = env._goal_position()

            initial_distance = float(
                np.linalg.norm(
                    goal - ee
                )
            )

            env.previous_distance = (
                initial_distance
            )

            state = env._observation()

            # ====================================================
            # Failure-aware routing
            # ====================================================

            if target_y < ROUTER_Y_THRESHOLD:

                agent = targeted_agent
                policy_name = "targeted"

                targeted_count += 1

            else:

                agent = original_agent
                policy_name = "original"

                original_count += 1

            total_reward = 0.0
            success = False
            steps_taken = 0

            final_info = None

            # ====================================================
            # Deterministic rollout
            # ====================================================

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

            if final_info is None:

                raise RuntimeError(
                    "No environment step executed."
                )

            final_distance = float(
                final_info[
                    "distance"
                ]
            )

            final_ee = np.asarray(
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

            if success:
                successful_steps.append(
                    steps_taken
                )

            if policy_name == "targeted":

                targeted_successes.append(
                    1 if success else 0
                )

            else:

                original_successes.append(
                    1 if success else 0
                )

            writer.writerow(
                [
                    episode,
                    target_x,
                    target_y,
                    policy_name,
                    int(success),
                    initial_distance,
                    final_distance,
                    steps_taken,
                    total_reward,
                    float(final_ee[0]),
                    float(final_ee[1]),
                    float(final_ee[2]),
                ]
            )

            csv_file.flush()

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
                    f"policy={policy_name:8s} | "
                    f"xy="
                    f"[{target_x:.3f}, "
                    f"{target_y:.3f}] | "
                    f"success="
                    f"{success} | "
                    f"final="
                    f"{final_distance:.4f} | "
                    f"steps="
                    f"{steps_taken:03d} | "
                    f"running="
                    f"{running_success:.1f}%"
                )

        # ========================================================
        # Statistics
        # ========================================================

        successes_np = np.asarray(
            successes,
            dtype=np.float32,
        )

        final_np = np.asarray(
            final_distances,
            dtype=np.float32,
        )

        initial_np = np.asarray(
            initial_distances,
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

        success_rate = float(
            successes_np.mean()
            * 100.0
        )

        mean_final = float(
            final_np.mean()
        )

        median_final = float(
            np.median(
                final_np
            )
        )

        worst_final = float(
            final_np.max()
        )

        best_final = float(
            final_np.min()
        )

        mean_initial = float(
            initial_np.mean()
        )

        mean_steps_all = float(
            steps_np.mean()
        )

        mean_steps_success = (
            float(
                np.mean(
                    successful_steps
                )
            )
            if successful_steps
            else float("nan")
        )

        mean_reward = float(
            rewards_np.mean()
        )

        targeted_rate = (
            100.0
            * np.mean(
                targeted_successes
            )
            if targeted_successes
            else 0.0
        )

        original_rate = (
            100.0
            * np.mean(
                original_successes
            )
            if original_successes
            else 0.0
        )

        # ========================================================
        # Results
        # ========================================================

        print(
            "\n"
            "============================================================"
        )

        print(
            "HYBRID POLICY RESULTS"
        )

        print(
            "============================================================"
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
            f"Best final distance:      "
            f"{best_final:.4f} m"
        )

        print(
            f"Worst final distance:     "
            f"{worst_final:.4f} m"
        )

        print(
            f"Mean steps all:           "
            f"{mean_steps_all:.1f}"
        )

        print(
            f"Mean steps success:       "
            f"{mean_steps_success:.1f}"
        )

        print(
            f"Mean episode reward:      "
            f"{mean_reward:.3f}"
        )

        print(
            "\nRouter usage:"
        )

        print(
            f"Targeted SAC episodes:    "
            f"{targeted_count}"
        )

        print(
            f"Original SAC episodes:    "
            f"{original_count}"
        )

        print(
            f"Targeted-route success:   "
            f"{targeted_rate:.1f}%"
        )

        print(
            f"Original-route success:   "
            f"{original_rate:.1f}%"
        )

        print(
            "\nReference:"
        )

        print(
            "Original-only uniform:    88.0%"
        )

        print(
            "Targeted-only uniform:    84.0%"
        )

        print(
            f"Hybrid uniform:           "
            f"{success_rate:.1f}%"
        )

        print(
            "\nResults saved:"
        )

        print(
            OUTPUT_CSV
        )

        print(
            "============================================================\n"
        )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
