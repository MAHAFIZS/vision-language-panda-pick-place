# -*- coding: utf-8 -*-

"""
evaluate_hybrid_sac_grasp_adaptive_100.py

Final adaptive manipulation benchmark.

Architecture:

    object position
          |
          v
    failure-aware SAC router
       /            \
 original SAC    targeted SAC
       \            /
          pre-grasp
              |
              v
       grasp risk classifier
        /             \
 normal region       low-X risky region
      |                       |
 direct grasp          staged approach
                             |
                       safe_x = 0.450
                             |
                      vertical descent
                             |
                      horizontal inward
        \                    /
             close gripper
                  |
                 lift
                  |
               verify

Uses exactly the same environment seed as the previous
100-episode benchmark so the result can be compared against
the previous 94% baseline.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ================================================================
# Project
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from rl.panda_object_reach_env import PandaObjectReachEnv

from rl.sac_agent import (
    SACAgent,
    SACConfig,
)

from rl.evaluate_hybrid_sac_grasp_100 import (
    load_agent,
    run_hybrid_pregrasp,
    execute_scripted_grasp,
)

from rl.sweep_staged_grasp_approach import (
    staged_grasp,
)


# ================================================================
# Configuration
# ================================================================

NUM_EPISODES = 100

# IMPORTANT:
# Same seed as previous 94% benchmark.
EVAL_SEED = 3030

MAX_RL_STEPS = 120


# ================================================================
# Adaptive grasp region
# ================================================================

LOW_X_THRESHOLD = 0.45

RISK_Y_LOW = -0.13

RISK_Y_HIGH = 0.00

SAFE_X = 0.450


# ================================================================
# Models
# ================================================================

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


# ================================================================
# Output
# ================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid_sac_grasp_adaptive_100"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "adaptive_grasp_100.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "summary.txt"
)


# ================================================================
# Risk classifier
# ================================================================

def is_risky_grasp_region(
    object_position: np.ndarray,
) -> bool:

    x = float(
        object_position[0]
    )

    y = float(
        object_position[1]
    )

    return bool(
        x <= LOW_X_THRESHOLD
        and
        RISK_Y_LOW
        <= y
        <= RISK_Y_HIGH
    )


# ================================================================
# Main
# ================================================================

def main():

    print(
        "\n"
        "=============================================================="
    )

    print(
        "Adaptive Failure-Aware Manipulation Benchmark"
    )

    print(
        "=============================================================="
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
        "\nDevice:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    np.random.seed(
        EVAL_SEED
    )

    torch.manual_seed(
        EVAL_SEED
    )

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_RL_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=EVAL_SEED,
    )

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    print(
        "\nLoading original SAC..."
    )

    original_agent = load_agent(
        ORIGINAL_MODEL,
        env.observation_dim,
        env.action_dim,
        device,
    )

    print(
        "Loading targeted SAC..."
    )

    targeted_agent = load_agent(
        TARGETED_MODEL,
        env.observation_dim,
        env.action_dim,
        device,
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

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

            "reach_policy",

            "object_x",
            "object_y",
            "object_z",

            "pregrasp_success",
            "pregrasp_error",
            "rl_steps",

            "risk_region",
            "grasp_strategy",

            "grasp_success",
            "full_success",

            "object_lift",

            "direct_or_staged",
        ]
    )

    # ============================================================
    # Metrics
    # ============================================================

    pregrasp_results = []

    grasp_results = []

    full_results = []

    direct_results = []

    staged_results = []

    original_route_results = []

    targeted_route_results = []

    risky_count = 0

    start_time = time.time()

    try:

        # ========================================================
        # Evaluation
        # ========================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            # ====================================================
            # 1. Failure-aware SAC reach
            # ====================================================

            pregrasp = run_hybrid_pregrasp(
                env,
                original_agent,
                targeted_agent,
            )

            object_position = np.asarray(
                pregrasp[
                    "object_position"
                ],
                dtype=float,
            )

            reach_policy = (
                pregrasp[
                    "policy"
                ]
            )

            pregrasp_success = bool(
                pregrasp[
                    "success"
                ]
            )

            pregrasp_results.append(
                int(
                    pregrasp_success
                )
            )

            # ----------------------------------------------------
            # If reach failed, do not grasp.
            # ----------------------------------------------------

            if not pregrasp_success:

                grasp_success = False

                full_success = False

                risk_region = False

                strategy = (
                    "not_attempted"
                )

                object_lift = 0.0

            else:

                # =================================================
                # 2. Grasp risk classifier
                # =================================================

                risk_region = (
                    is_risky_grasp_region(
                        object_position
                    )
                )

                # =================================================
                # 3A. Staged correction
                # =================================================

                if risk_region:

                    risky_count += 1

                    strategy = (
                        "staged"
                    )

                    grasp = staged_grasp(
                        env,
                        safe_x=SAFE_X,
                    )

                    grasp_success = bool(
                        grasp[
                            "success"
                        ]
                    )

                    object_lift = float(
                        grasp[
                            "object_lift"
                        ]
                    )

                    staged_results.append(
                        int(
                            grasp_success
                        )
                    )

                # =================================================
                # 3B. Standard grasp
                # =================================================

                else:

                    strategy = (
                        "direct"
                    )

                    grasp = (
                        execute_scripted_grasp(
                            env
                        )
                    )

                    grasp_success = bool(
                        grasp[
                            "success"
                        ]
                    )

                    object_lift = float(
                        grasp[
                            "object_vertical_motion"
                        ]
                    )

                    direct_results.append(
                        int(
                            grasp_success
                        )
                    )

                full_success = bool(
                    pregrasp_success
                    and
                    grasp_success
                )

            # ====================================================
            # Metrics
            # ====================================================

            grasp_results.append(
                int(
                    grasp_success
                )
            )

            full_results.append(
                int(
                    full_success
                )
            )

            if reach_policy == "original":

                original_route_results.append(
                    int(
                        full_success
                    )
                )

            else:

                targeted_route_results.append(
                    int(
                        full_success
                    )
                )

            # ====================================================
            # CSV
            # ====================================================

            writer.writerow(
                [
                    episode,

                    reach_policy,

                    float(
                        object_position[0]
                    ),

                    float(
                        object_position[1]
                    ),

                    float(
                        object_position[2]
                    ),

                    int(
                        pregrasp_success
                    ),

                    float(
                        pregrasp[
                            "final_distance"
                        ]
                    ),

                    int(
                        pregrasp[
                            "rl_steps"
                        ]
                    ),

                    int(
                        risk_region
                    ),

                    strategy,

                    int(
                        grasp_success
                    ),

                    int(
                        full_success
                    ),

                    object_lift,

                    strategy,
                ]
            )

            csv_file.flush()

            # ====================================================
            # Running result
            # ====================================================

            running_full = (
                100.0
                * np.mean(
                    full_results
                )
            )

            if (
                episode <= 10
                or
                episode % 10 == 0
                or
                not full_success
            ):

                print(
                    f"Episode "
                    f"{episode:03d}/"
                    f"{NUM_EPISODES} | "
                    f"reach="
                    f"{reach_policy:8s} | "
                    f"xy="
                    f"["
                    f"{object_position[0]:.3f}, "
                    f"{object_position[1]:.3f}"
                    f"] | "
                    f"grasp="
                    f"{strategy:6s} | "
                    f"pre="
                    f"{pregrasp_success} | "
                    f"grasp_ok="
                    f"{grasp_success} | "
                    f"lift="
                    f"{object_lift:.4f} | "
                    f"full="
                    f"{full_success} | "
                    f"running="
                    f"{running_full:.1f}%"
                )

        # ========================================================
        # Statistics
        # ========================================================

        pregrasp_rate = (
            100.0
            * np.mean(
                pregrasp_results
            )
        )

        grasp_rate = (
            100.0
            * np.mean(
                grasp_results
            )
        )

        full_rate = (
            100.0
            * np.mean(
                full_results
            )
        )

        direct_rate = (
            100.0
            * np.mean(
                direct_results
            )
            if direct_results
            else 0.0
        )

        staged_rate = (
            100.0
            * np.mean(
                staged_results
            )
            if staged_results
            else 0.0
        )

        original_route_rate = (
            100.0
            * np.mean(
                original_route_results
            )
            if original_route_results
            else 0.0
        )

        targeted_route_rate = (
            100.0
            * np.mean(
                targeted_route_results
            )
            if targeted_route_results
            else 0.0
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ========================================================
        # Final output
        # ========================================================

        print(
            "\n"
            "=============================================================="
        )

        print(
            "FINAL ADAPTIVE MANIPULATION RESULTS"
        )

        print(
            "=============================================================="
        )

        print(
            f"Episodes:                         "
            f"{NUM_EPISODES}"
        )

        print(
            f"\nPre-grasp success:                "
            f"{int(np.sum(pregrasp_results))}/"
            f"{NUM_EPISODES} "
            f"({pregrasp_rate:.1f}%)"
        )

        print(
            f"Grasp success:                    "
            f"{int(np.sum(grasp_results))}/"
            f"{NUM_EPISODES} "
            f"({grasp_rate:.1f}%)"
        )

        print(
            f"\nFULL PIPELINE SUCCESS:            "
            f"{int(np.sum(full_results))}/"
            f"{NUM_EPISODES} "
            f"({full_rate:.1f}%)"
        )

        print(
            "\nGrasp strategy:"
        )

        print(
            f"Direct grasp trials:              "
            f"{len(direct_results)}"
        )

        print(
            f"Direct grasp success:             "
            f"{direct_rate:.1f}%"
        )

        print(
            f"Staged grasp trials:              "
            f"{len(staged_results)}"
        )

        print(
            f"Staged grasp success:             "
            f"{staged_rate:.1f}%"
        )

        print(
            f"Risky-region detections:          "
            f"{risky_count}"
        )

        print(
            "\nReach-route full success:"
        )

        print(
            f"Original SAC route:               "
            f"{original_route_rate:.1f}%"
        )

        print(
            f"Targeted SAC route:               "
            f"{targeted_route_rate:.1f}%"
        )

        print(
            "\nBaseline comparison:"
        )

        print(
            "Previous full pipeline:           "
            "94.0%"
        )

        print(
            f"Adaptive full pipeline:           "
            f"{full_rate:.1f}%"
        )

        print(
            f"Improvement:                      "
            f"{full_rate - 94.0:+.1f} percentage points"
        )

        print(
            f"\nElapsed:                          "
            f"{elapsed / 60.0:.1f} min"
        )

        print(
            "\nCSV:"
        )

        print(
            OUTPUT_CSV
        )

        print(
            "==============================================================\n"
        )

        # ========================================================
        # Summary file
        # ========================================================

        with open(
            SUMMARY_FILE,
            "w",
        ) as file:

            file.write(
                "Adaptive Failure-Aware Manipulation Benchmark\n"
            )

            file.write(
                "============================================\n\n"
            )

            file.write(
                f"Episodes: "
                f"{NUM_EPISODES}\n"
            )

            file.write(
                f"Pre-grasp success: "
                f"{pregrasp_rate:.1f}%\n"
            )

            file.write(
                f"Grasp success: "
                f"{grasp_rate:.1f}%\n"
            )

            file.write(
                f"Full pipeline success: "
                f"{full_rate:.1f}%\n\n"
            )

            file.write(
                f"Direct grasp success: "
                f"{direct_rate:.1f}%\n"
            )

            file.write(
                f"Staged grasp success: "
                f"{staged_rate:.1f}%\n"
            )

            file.write(
                f"Staged trials: "
                f"{len(staged_results)}\n\n"
            )

            file.write(
                f"Original route full success: "
                f"{original_route_rate:.1f}%\n"
            )

            file.write(
                f"Targeted route full success: "
                f"{targeted_route_rate:.1f}%\n\n"
            )

            file.write(
                "Previous full pipeline: "
                "94.0%\n"
            )

            file.write(
                f"Adaptive full pipeline: "
                f"{full_rate:.1f}%\n"
            )

            file.write(
                f"Improvement: "
                f"{full_rate - 94.0:+.1f} pp\n"
            )

    finally:

        csv_file.close()

        env.close()


if __name__ == "__main__":
    main()
