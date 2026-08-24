# -*- coding: utf-8 -*-

"""
evaluate_sac_hard_targets.py

Compare:

1. Original SAC model
2. Failure-aware targeted SAC model

on the SAME hard-target distribution.

Hard target modes:
    - low X
    - negative Y
    - high Z

This gives a direct benchmark of whether targeted retraining
actually improved performance in the regions that originally failed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


# ================================================================
# Project paths
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_rl_env_targeted import PandaReachTargetedEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100

EVAL_SEED = 9090

MAX_STEPS = 120

# Force 100% hard targets.
HARD_TARGET_PROBABILITY = 1.0


ORIGINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_reach"
    / "sac_reach_best.pt"
)


TARGETED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_reach_targeted"
    / "sac_targeted_best.pt"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_hard_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# Create SAC agent
# ================================================================

def create_agent(
    model_path: Path,
    device: torch.device,
) -> SACAgent:

    config = SACConfig(
        state_dim=23,
        action_dim=3,
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
# Generate fixed hard targets
# ================================================================

def generate_hard_targets(
    num_episodes: int,
    seed: int,
):

    """
    Generate one fixed sequence of hard targets.

    Both models will evaluate on exactly these targets.
    """

    env = PandaReachTargetedEnv(
        max_steps=MAX_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.035,
        hard_target_probability=HARD_TARGET_PROBABILITY,
        seed=seed,
    )

    targets = []

    try:

        for _ in range(
            num_episodes
        ):

            _, info = env.reset()

            targets.append(
                {
                    "target":
                        info["target"].copy(),

                    "target_mode":
                        info["target_mode"],
                }
            )

    finally:

        env.close()

    return targets


# ================================================================
# Force target into environment
# ================================================================

def set_fixed_target(
    env: PandaReachTargetedEnv,
    target: np.ndarray,
):

    env.target = np.asarray(
        target,
        dtype=np.float32,
    )

    ee = env._ee_position()

    env.previous_distance = float(
        np.linalg.norm(
            env.target - ee
        )
    )

    obs = env._observation()

    info = {
        "target":
            env.target.copy(),

        "ee_position":
            ee.copy(),

        "distance":
            env.previous_distance,
    }

    return obs, info


# ================================================================
# Evaluate one model
# ================================================================

def evaluate_model(
    model_name: str,
    model_path: Path,
    targets,
    device: torch.device,
):

    print(
        "\n"
        "=============================================\n"
        f"Evaluating: {model_name}\n"
        "=============================================\n"
    )

    print(
        "Model:",
        model_path,
    )

    agent = create_agent(
        model_path=model_path,
        device=device,
    )

    env = PandaReachTargetedEnv(
        max_steps=MAX_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.035,

        # reset target will be overridden,
        # but keep hard mode enabled.
        hard_target_probability=1.0,

        seed=12345,
    )

    results = []

    try:

        for episode_index, target_entry in enumerate(
            targets,
            start=1,
        ):

            # ----------------------------------------------------
            # Reset robot
            # ----------------------------------------------------

            _, _ = env.reset()

            # ----------------------------------------------------
            # Replace random reset target with fixed benchmark target
            # ----------------------------------------------------

            state, initial_info = set_fixed_target(
                env,
                target_entry["target"],
            )

            initial_distance = float(
                initial_info["distance"]
            )

            total_reward = 0.0

            success = False

            final_info = initial_info

            steps_taken = 0

            # ----------------------------------------------------
            # Episode
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
            # Save result
            # ----------------------------------------------------

            final_distance = float(
                final_info["distance"]
            )

            target = np.asarray(
                target_entry["target"],
                dtype=np.float32,
            )

            ee = np.asarray(
                final_info["ee_position"],
                dtype=np.float32,
            )

            result = {
                "episode":
                    episode_index,

                "model":
                    model_name,

                "target_mode":
                    target_entry[
                        "target_mode"
                    ],

                "success":
                    int(success),

                "initial_distance_m":
                    initial_distance,

                "final_distance_m":
                    final_distance,

                "steps":
                    steps_taken,

                "episode_reward":
                    total_reward,

                "target_x":
                    float(
                        target[0]
                    ),

                "target_y":
                    float(
                        target[1]
                    ),

                "target_z":
                    float(
                        target[2]
                    ),

                "final_ee_x":
                    float(
                        ee[0]
                    ),

                "final_ee_y":
                    float(
                        ee[1]
                    ),

                "final_ee_z":
                    float(
                        ee[2]
                    ),
            }

            results.append(
                result
            )

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (
                episode_index <= 10
                or episode_index % 10 == 0
            ):

                running_success = (
                    100.0
                    * np.mean(
                        [
                            r["success"]
                            for r in results
                        ]
                    )
                )

                print(
                    f"Episode "
                    f"{episode_index:03d}/{NUM_EPISODES} | "
                    f"mode="
                    f"{target_entry['target_mode']:16s} | "
                    f"success="
                    f"{success} | "
                    f"final="
                    f"{final_distance:.4f} m | "
                    f"steps="
                    f"{steps_taken:03d} | "
                    f"running="
                    f"{running_success:.1f}%"
                )

    finally:

        env.close()

    return results


# ================================================================
# Summary
# ================================================================

def summarize(
    results,
):

    successes = np.asarray(
        [
            r["success"]
            for r in results
        ],
        dtype=np.float32,
    )

    final_distances = np.asarray(
        [
            r["final_distance_m"]
            for r in results
        ],
        dtype=np.float32,
    )

    steps = np.asarray(
        [
            r["steps"]
            for r in results
        ],
        dtype=np.float32,
    )

    rewards = np.asarray(
        [
            r["episode_reward"]
            for r in results
        ],
        dtype=np.float32,
    )

    successful_steps = [
        r["steps"]
        for r in results
        if r["success"] == 1
    ]

    summary = {
        "episodes":
            len(results),

        "successes":
            int(
                successes.sum()
            ),

        "failures":
            len(results)
            - int(
                successes.sum()
            ),

        "success_rate":
            float(
                successes.mean()
                * 100.0
            ),

        "mean_final_distance":
            float(
                final_distances.mean()
            ),

        "median_final_distance":
            float(
                np.median(
                    final_distances
                )
            ),

        "std_final_distance":
            float(
                final_distances.std()
            ),

        "best_final_distance":
            float(
                final_distances.min()
            ),

        "worst_final_distance":
            float(
                final_distances.max()
            ),

        "mean_steps_all":
            float(
                steps.mean()
            ),

        "mean_steps_success":
            (
                float(
                    np.mean(
                        successful_steps
                    )
                )
                if successful_steps
                else float("nan")
            ),

        "mean_reward":
            float(
                rewards.mean()
            ),
    }

    return summary


# ================================================================
# Per-region summary
# ================================================================

def summarize_by_mode(
    results,
):

    modes = [
        "hard_low_x",
        "hard_negative_y",
        "hard_high_z",
    ]

    summaries = {}

    for mode in modes:

        mode_results = [
            r
            for r in results
            if r["target_mode"] == mode
        ]

        if not mode_results:
            continue

        summaries[mode] = summarize(
            mode_results
        )

    return summaries


# ================================================================
# Save CSV
# ================================================================

def save_results_csv(
    results,
    path: Path,
):

    fieldnames = [
        "episode",
        "model",
        "target_mode",
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

    with open(
        path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ================================================================
# Print comparison
# ================================================================

def print_summary(
    name,
    summary,
):

    print(
        f"\n{name}"
    )

    print(
        "-" * 50
    )

    print(
        f"Episodes:              "
        f"{summary['episodes']}"
    )

    print(
        f"Successes:             "
        f"{summary['successes']}"
    )

    print(
        f"Failures:              "
        f"{summary['failures']}"
    )

    print(
        f"Success rate:          "
        f"{summary['success_rate']:.1f}%"
    )

    print(
        f"Mean final distance:   "
        f"{summary['mean_final_distance']:.4f} m"
    )

    print(
        f"Median final distance: "
        f"{summary['median_final_distance']:.4f} m"
    )

    print(
        f"Worst final distance:  "
        f"{summary['worst_final_distance']:.4f} m"
    )

    print(
        f"Mean steps success:    "
        f"{summary['mean_steps_success']:.1f}"
    )

    print(
        f"Mean reward:           "
        f"{summary['mean_reward']:.3f}"
    )


# ================================================================
# Main
# ================================================================

def main():

    print(
        "\n"
        "=============================================\n"
        "SAC Hard-Target Benchmark\n"
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
    # Check models
    # ------------------------------------------------------------

    if not ORIGINAL_MODEL.exists():

        raise FileNotFoundError(
            f"Original SAC model not found:\n"
            f"{ORIGINAL_MODEL}"
        )

    if not TARGETED_MODEL.exists():

        raise FileNotFoundError(
            f"Targeted SAC model not found:\n"
            f"{TARGETED_MODEL}"
        )

    # ------------------------------------------------------------
    # Generate ONE shared hard-target benchmark
    # ------------------------------------------------------------

    print(
        "\nGenerating fixed hard-target benchmark..."
    )

    targets = generate_hard_targets(
        num_episodes=NUM_EPISODES,
        seed=EVAL_SEED,
    )

    mode_counts = {}

    for entry in targets:

        mode = entry[
            "target_mode"
        ]

        mode_counts[
            mode
        ] = (
            mode_counts.get(
                mode,
                0,
            )
            + 1
        )

    print(
        "\nHard-target distribution:"
    )

    for mode, count in mode_counts.items():

        print(
            f"{mode:18s}: {count}"
        )

    # ------------------------------------------------------------
    # Original SAC
    # ------------------------------------------------------------

    original_results = evaluate_model(
        model_name="original_sac",
        model_path=ORIGINAL_MODEL,
        targets=targets,
        device=device,
    )

    # ------------------------------------------------------------
    # Targeted SAC
    # ------------------------------------------------------------

    targeted_results = evaluate_model(
        model_name="targeted_sac",
        model_path=TARGETED_MODEL,
        targets=targets,
        device=device,
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    original_csv = (
        OUTPUT_DIR
        / "original_sac_hard_100.csv"
    )

    targeted_csv = (
        OUTPUT_DIR
        / "targeted_sac_hard_100.csv"
    )

    save_results_csv(
        original_results,
        original_csv,
    )

    save_results_csv(
        targeted_results,
        targeted_csv,
    )

    # ------------------------------------------------------------
    # Overall summaries
    # ------------------------------------------------------------

    original_summary = summarize(
        original_results
    )

    targeted_summary = summarize(
        targeted_results
    )

    print(
        "\n"
        "=============================================\n"
        "Overall Hard-Target Results\n"
        "============================================="
    )

    print_summary(
        "Original SAC",
        original_summary,
    )

    print_summary(
        "Failure-Aware Targeted SAC",
        targeted_summary,
    )

    # ------------------------------------------------------------
    # Improvement
    # ------------------------------------------------------------

    success_improvement = (
        targeted_summary[
            "success_rate"
        ]
        - original_summary[
            "success_rate"
        ]
    )

    distance_improvement = (
        original_summary[
            "mean_final_distance"
        ]
        - targeted_summary[
            "mean_final_distance"
        ]
    )

    worst_improvement = (
        original_summary[
            "worst_final_distance"
        ]
        - targeted_summary[
            "worst_final_distance"
        ]
    )

    print(
        "\n"
        "=============================================\n"
        "Improvement\n"
        "============================================="
    )

    print(
        f"Success-rate improvement: "
        f"{success_improvement:+.1f} percentage points"
    )

    print(
        f"Mean-error improvement:   "
        f"{distance_improvement:+.4f} m"
    )

    print(
        f"Worst-error improvement:  "
        f"{worst_improvement:+.4f} m"
    )

    # ------------------------------------------------------------
    # Per-mode results
    # ------------------------------------------------------------

    original_modes = summarize_by_mode(
        original_results
    )

    targeted_modes = summarize_by_mode(
        targeted_results
    )

    print(
        "\n"
        "=============================================\n"
        "Per Hard-Target Region\n"
        "============================================="
    )

    for mode in [
        "hard_low_x",
        "hard_negative_y",
        "hard_high_z",
    ]:

        if (
            mode not in original_modes
            or mode not in targeted_modes
        ):

            continue

        old = original_modes[
            mode
        ]

        new = targeted_modes[
            mode
        ]

        print(
            f"\n{mode}"
        )

        print(
            "-" * 50
        )

        print(
            f"Samples: "
            f"{old['episodes']}"
        )

        print(
            f"Original success: "
            f"{old['success_rate']:.1f}%"
        )

        print(
            f"Targeted success: "
            f"{new['success_rate']:.1f}%"
        )

        print(
            f"Improvement: "
            f"{new['success_rate'] - old['success_rate']:+.1f} pp"
        )

        print(
            f"Original mean error: "
            f"{old['mean_final_distance']:.4f} m"
        )

        print(
            f"Targeted mean error: "
            f"{new['mean_final_distance']:.4f} m"
        )

    # ------------------------------------------------------------
    # Save text summary
    # ------------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "hard_target_comparison.txt"
    )

    with open(
        summary_path,
        "w",
    ) as file:

        file.write(
            "SAC Hard-Target Benchmark\n"
        )

        file.write(
            "=========================\n\n"
        )

        file.write(
            f"Original success rate: "
            f"{original_summary['success_rate']:.1f}%\n"
        )

        file.write(
            f"Targeted success rate: "
            f"{targeted_summary['success_rate']:.1f}%\n"
        )

        file.write(
            f"Improvement: "
            f"{success_improvement:+.1f} percentage points\n\n"
        )

        file.write(
            f"Original mean final distance: "
            f"{original_summary['mean_final_distance']:.4f} m\n"
        )

        file.write(
            f"Targeted mean final distance: "
            f"{targeted_summary['mean_final_distance']:.4f} m\n"
        )

        file.write(
            f"Mean error improvement: "
            f"{distance_improvement:+.4f} m\n\n"
        )

        for mode in [
            "hard_low_x",
            "hard_negative_y",
            "hard_high_z",
        ]:

            if (
                mode not in original_modes
                or mode not in targeted_modes
            ):

                continue

            old = original_modes[
                mode
            ]

            new = targeted_modes[
                mode
            ]

            file.write(
                f"{mode}\n"
            )

            file.write(
                f"  Original success: "
                f"{old['success_rate']:.1f}%\n"
            )

            file.write(
                f"  Targeted success: "
                f"{new['success_rate']:.1f}%\n"
            )

            file.write(
                f"  Improvement: "
                f"{new['success_rate'] - old['success_rate']:+.1f} pp\n\n"
            )

    print(
        "\n"
        "=============================================\n"
        "Saved Results\n"
        "============================================="
    )

    print(
        original_csv
    )

    print(
        targeted_csv
    )

    print(
        summary_path
    )

    print(
        "\nBenchmark complete.\n"
    )


if __name__ == "__main__":
    main()
