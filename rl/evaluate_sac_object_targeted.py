# -*- coding: utf-8 -*-

"""
evaluate_sac_object_targeted.py

Fair deterministic comparison:

1. Original object SAC
2. Failure-aware targeted object SAC

Both policies are evaluated on:

A) The exact same 100 uniform unseen object placements
B) The exact same 100 hard negative-Y placements

Metrics:
    - success rate
    - mean final distance
    - median final distance
    - worst final distance
    - mean steps on successful episodes
    - mean reward

This gives a clean before-vs-after comparison.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


# ================================================================
# Project imports
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.panda_object_reach_env_targeted import PandaObjectReachTargetedEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100

UNIFORM_SEED = 2027

HARD_SEED = 9091

MAX_STEPS = 120


ORIGINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "sac_object_reach_best.pt"
)


TARGETED_BEST_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
    / "sac_object_targeted_best.pt"
)


TARGETED_FINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
    / "sac_object_targeted_final.pt"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
    / "comparison"
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
# Generate exact target list
# ================================================================

def generate_uniform_targets(
    count: int,
    seed: int,
):

    rng = np.random.default_rng(
        seed
    )

    targets = []

    for _ in range(
        count
    ):

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

def generate_hard_targets(
    count: int,
    seed: int,
):

    rng = np.random.default_rng(
        seed
    )

    targets = []

    for _ in range(
        count
    ):

        x = rng.uniform(
            0.44,
            0.59,
        )

        y = rng.uniform(
            -0.20,
            -0.14,
        )

        targets.append(
            (
                float(x),
                float(y),
            )
        )

    return targets


# ================================================================
# Force object position
# ================================================================

def set_object_xy(
    env,
    x: float,
    y: float,
):

    joint_id = (
        env.object_free_joint_id
    )

    if joint_id is None:

        raise RuntimeError(
            "Object has no free joint."
        )

    qpos_address = int(
        env.robot.model
        .jnt_qposadr[
            joint_id
        ]
    )

    current_z = float(
        env.robot.data
        .qpos[
            qpos_address + 2
        ]
    )

    env.robot.data.qpos[
        qpos_address
    ] = x

    env.robot.data.qpos[
        qpos_address + 1
    ] = y

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
        env.robot.model
        .jnt_dofadr[
            joint_id
        ]
    )

    env.robot.data.qvel[
        dof_address:
        dof_address + 6
    ] = 0.0

    import mujoco

    mujoco.mj_forward(
        env.robot.model,
        env.robot.data,
    )


# ================================================================
# Evaluate one model
# ================================================================

def evaluate_model(
    name: str,
    model_path: Path,
    targets,
    device: torch.device,
    output_csv: Path,
):

    print(
        "\n"
        "============================================="
    )

    print(
        f"Evaluating: {name}"
    )

    print(
        "Model:",
        model_path,
    )

    print(
        "Episodes:",
        len(targets),
    )

    print(
        "============================================="
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

    agent = load_agent(
        model_path=model_path,
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        device=device,
    )

    successes = []

    initial_distances = []

    final_distances = []

    rewards = []

    steps_all = []

    steps_success = []

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = open(
        output_csv,
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
            "target_x",
            "target_y",
            "object_x",
            "object_y",
            "initial_distance",
            "final_distance",
            "steps",
            "reward",
            "final_ee_x",
            "final_ee_y",
            "final_ee_z",
            "goal_x",
            "goal_y",
            "goal_z",
        ]
    )

    try:

        for episode_index, (
            target_x,
            target_y,
        ) in enumerate(
            targets,
            start=1,
        ):

            # ----------------------------------------------------
            # Normal reset
            # ----------------------------------------------------

            state, info = (
                env.reset()
            )

            # ----------------------------------------------------
            # Override randomized object with exact target
            # ----------------------------------------------------

            set_object_xy(
                env,
                target_x,
                target_y,
            )

            # ----------------------------------------------------
            # Recompute controller target after force placement
            # ----------------------------------------------------

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

            state = (
                env._observation()
            )

            total_reward = 0.0

            success = False

            steps_taken = 0

            final_info = None

            # ----------------------------------------------------
            # Deterministic rollout
            # ----------------------------------------------------

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                action = (
                    agent.select_action(
                        state,
                        deterministic=True,
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

                state = next_state

                total_reward += (
                    reward
                )

                steps_taken = step

                final_info = info

                if terminated:

                    success = True

                    break

                if truncated:

                    break

            if final_info is None:

                raise RuntimeError(
                    "Episode did not execute."
                )

            final_distance = float(
                final_info[
                    "distance"
                ]
            )

            object_position = (
                np.asarray(
                    final_info[
                        "object_position"
                    ],
                    dtype=np.float32,
                )
            )

            final_ee = (
                np.asarray(
                    final_info[
                        "ee_position"
                    ],
                    dtype=np.float32,
                )
            )

            goal = (
                np.asarray(
                    final_info[
                        "goal_position"
                    ],
                    dtype=np.float32,
                )
            )

            successes.append(
                1
                if success
                else 0
            )

            initial_distances.append(
                initial_distance
            )

            final_distances.append(
                final_distance
            )

            rewards.append(
                total_reward
            )

            steps_all.append(
                steps_taken
            )

            if success:

                steps_success.append(
                    steps_taken
                )

            writer.writerow(
                [
                    episode_index,
                    int(success),
                    target_x,
                    target_y,
                    float(
                        object_position[0]
                    ),
                    float(
                        object_position[1]
                    ),
                    initial_distance,
                    final_distance,
                    steps_taken,
                    total_reward,
                    float(
                        final_ee[0]
                    ),
                    float(
                        final_ee[1]
                    ),
                    float(
                        final_ee[2]
                    ),
                    float(
                        goal[0]
                    ),
                    float(
                        goal[1]
                    ),
                    float(
                        goal[2]
                    ),
                ]
            )

            csv_file.flush()

            if (
                episode_index <= 10
                or episode_index % 10 == 0
            ):

                running = (
                    100.0
                    * np.mean(
                        successes
                    )
                )

                print(
                    f"Episode "
                    f"{episode_index:03d}/"
                    f"{len(targets)} | "
                    f"success="
                    f"{success} | "
                    f"xy="
                    f"[{target_x:.3f}, "
                    f"{target_y:.3f}] | "
                    f"final="
                    f"{final_distance:.4f} | "
                    f"steps="
                    f"{steps_taken:03d} | "
                    f"running="
                    f"{running:.1f}%"
                )

        # ========================================================
        # Summary
        # ========================================================

        success_np = np.asarray(
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

        rewards_np = np.asarray(
            rewards,
            dtype=np.float32,
        )

        steps_np = np.asarray(
            steps_all,
            dtype=np.float32,
        )

        result = {
            "name":
                name,

            "episodes":
                len(targets),

            "successes":
                int(
                    success_np.sum()
                ),

            "success_rate":
                float(
                    success_np.mean()
                    * 100.0
                ),

            "mean_initial_distance":
                float(
                    initial_np.mean()
                ),

            "mean_final_distance":
                float(
                    final_np.mean()
                ),

            "median_final_distance":
                float(
                    np.median(
                        final_np
                    )
                ),

            "std_final_distance":
                float(
                    final_np.std()
                ),

            "best_final_distance":
                float(
                    final_np.min()
                ),

            "worst_final_distance":
                float(
                    final_np.max()
                ),

            "mean_steps_all":
                float(
                    steps_np.mean()
                ),

            "mean_steps_success":
                (
                    float(
                        np.mean(
                            steps_success
                        )
                    )
                    if steps_success
                    else float("nan")
                ),

            "mean_reward":
                float(
                    rewards_np.mean()
                ),
        }

        print(
            "\n"
            f"{name} RESULTS"
        )

        print(
            f"Success: "
            f"{result['successes']}/"
            f"{result['episodes']} "
            f"({result['success_rate']:.1f}%)"
        )

        print(
            f"Mean final distance: "
            f"{result['mean_final_distance']:.4f} m"
        )

        print(
            f"Median final distance: "
            f"{result['median_final_distance']:.4f} m"
        )

        print(
            f"Worst final distance: "
            f"{result['worst_final_distance']:.4f} m"
        )

        print(
            f"Mean steps success: "
            f"{result['mean_steps_success']:.1f}"
        )

        return result

    finally:

        csv_file.close()

        env.close()


# ================================================================
# Comparison printer
# ================================================================

def print_comparison(
    title: str,
    original,
    targeted,
):

    success_improvement = (
        targeted[
            "success_rate"
        ]
        - original[
            "success_rate"
        ]
    )

    error_improvement = (
        original[
            "mean_final_distance"
        ]
        - targeted[
            "mean_final_distance"
        ]
    )

    worst_improvement = (
        original[
            "worst_final_distance"
        ]
        - targeted[
            "worst_final_distance"
        ]
    )

    print(
        "\n"
        "=============================================================="
    )

    print(
        title
    )

    print(
        "=============================================================="
    )

    print(
        f"{'Metric':<30}"
        f"{'Original':>12}"
        f"{'Targeted':>12}"
        f"{'Change':>12}"
    )

    print(
        "-" * 66
    )

    print(
        f"{'Success rate':<30}"
        f"{original['success_rate']:>11.1f}%"
        f"{targeted['success_rate']:>11.1f}%"
        f"{success_improvement:>+11.1f}pp"
    )

    print(
        f"{'Mean final error [m]':<30}"
        f"{original['mean_final_distance']:>12.4f}"
        f"{targeted['mean_final_distance']:>12.4f}"
        f"{-error_improvement:>+12.4f}"
    )

    print(
        f"{'Median final error [m]':<30}"
        f"{original['median_final_distance']:>12.4f}"
        f"{targeted['median_final_distance']:>12.4f}"
    )

    print(
        f"{'Worst final error [m]':<30}"
        f"{original['worst_final_distance']:>12.4f}"
        f"{targeted['worst_final_distance']:>12.4f}"
        f"{-worst_improvement:>+12.4f}"
    )

    print(
        f"{'Mean steps success':<30}"
        f"{original['mean_steps_success']:>12.1f}"
        f"{targeted['mean_steps_success']:>12.1f}"
    )

    print(
        "\nImprovement:"
    )

    print(
        f"Success rate: "
        f"{success_improvement:+.1f} percentage points"
    )

    print(
        f"Mean error reduction: "
        f"{error_improvement:.4f} m"
    )

    print(
        f"Worst error reduction: "
        f"{worst_improvement:.4f} m"
    )


# ================================================================
# Main
# ================================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        "==============================================================\n"
        "Original vs Targeted Object-Reach SAC Evaluation\n"
        "==============================================================\n"
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
    # Targeted checkpoint selection
    # ------------------------------------------------------------

    if TARGETED_BEST_MODEL.exists():

        targeted_model = (
            TARGETED_BEST_MODEL
        )

    elif TARGETED_FINAL_MODEL.exists():

        targeted_model = (
            TARGETED_FINAL_MODEL
        )

        print(
            "\nTargeted best model not found."
        )

        print(
            "Using targeted final model."
        )

    else:

        raise FileNotFoundError(
            "No targeted model found."
        )

    if not ORIGINAL_MODEL.exists():

        raise FileNotFoundError(
            f"Original model missing:\n"
            f"{ORIGINAL_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Generate identical target lists
    # ============================================================

    uniform_targets = (
        generate_uniform_targets(
            NUM_EPISODES,
            UNIFORM_SEED,
        )
    )

    hard_targets = (
        generate_hard_targets(
            NUM_EPISODES,
            HARD_SEED,
        )
    )

    # ============================================================
    # Uniform benchmark
    # ============================================================

    print(
        "\n\n"
        "##############################################################"
    )

    print(
        "BENCHMARK 1: 100 IDENTICAL UNIFORM UNSEEN OBJECT POSITIONS"
    )

    print(
        "##############################################################"
    )

    original_uniform = (
        evaluate_model(
            name="ORIGINAL SAC - UNIFORM",
            model_path=ORIGINAL_MODEL,
            targets=uniform_targets,
            device=device,
            output_csv=(
                OUTPUT_DIR
                / "original_uniform_100.csv"
            ),
        )
    )

    targeted_uniform = (
        evaluate_model(
            name="TARGETED SAC - UNIFORM",
            model_path=targeted_model,
            targets=uniform_targets,
            device=device,
            output_csv=(
                OUTPUT_DIR
                / "targeted_uniform_100.csv"
            ),
        )
    )

    print_comparison(
        "UNIFORM UNSEEN OBJECT COMPARISON",
        original_uniform,
        targeted_uniform,
    )

    # ============================================================
    # Hard benchmark
    # ============================================================

    print(
        "\n\n"
        "##############################################################"
    )

    print(
        "BENCHMARK 2: 100 IDENTICAL HARD NEGATIVE-Y OBJECT POSITIONS"
    )

    print(
        "##############################################################"
    )

    original_hard = (
        evaluate_model(
            name="ORIGINAL SAC - HARD",
            model_path=ORIGINAL_MODEL,
            targets=hard_targets,
            device=device,
            output_csv=(
                OUTPUT_DIR
                / "original_hard_100.csv"
            ),
        )
    )

    targeted_hard = (
        evaluate_model(
            name="TARGETED SAC - HARD",
            model_path=targeted_model,
            targets=hard_targets,
            device=device,
            output_csv=(
                OUTPUT_DIR
                / "targeted_hard_100.csv"
            ),
        )
    )

    print_comparison(
        "HARD NEGATIVE-Y COMPARISON",
        original_hard,
        targeted_hard,
    )

    # ============================================================
    # Save text summary
    # ============================================================

    summary_path = (
        OUTPUT_DIR
        / "comparison_summary.txt"
    )

    with open(
        summary_path,
        "w",
    ) as file:

        file.write(
            "Object-Reach SAC Comparison\n"
        )

        file.write(
            "===========================\n\n"
        )

        file.write(
            "UNIFORM BENCHMARK\n"
        )

        file.write(
            f"Original success: "
            f"{original_uniform['success_rate']:.1f}%\n"
        )

        file.write(
            f"Targeted success: "
            f"{targeted_uniform['success_rate']:.1f}%\n"
        )

        file.write(
            f"Improvement: "
            f"{targeted_uniform['success_rate'] - original_uniform['success_rate']:+.1f} pp\n"
        )

        file.write(
            f"Original mean error: "
            f"{original_uniform['mean_final_distance']:.4f} m\n"
        )

        file.write(
            f"Targeted mean error: "
            f"{targeted_uniform['mean_final_distance']:.4f} m\n\n"
        )

        file.write(
            "HARD NEGATIVE-Y BENCHMARK\n"
        )

        file.write(
            f"Original success: "
            f"{original_hard['success_rate']:.1f}%\n"
        )

        file.write(
            f"Targeted success: "
            f"{targeted_hard['success_rate']:.1f}%\n"
        )

        file.write(
            f"Improvement: "
            f"{targeted_hard['success_rate'] - original_hard['success_rate']:+.1f} pp\n"
        )

        file.write(
            f"Original mean error: "
            f"{original_hard['mean_final_distance']:.4f} m\n"
        )

        file.write(
            f"Targeted mean error: "
            f"{targeted_hard['mean_final_distance']:.4f} m\n"
        )

    print(
        "\n"
        "=============================================================="
    )

    print(
        "Evaluation complete."
    )

    print(
        "Results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )

    print(
        "==============================================================\n"
    )


if __name__ == "__main__":
    main()
