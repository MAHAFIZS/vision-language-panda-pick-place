# -*- coding: utf-8 -*-

"""
analyze_grasp_failures.py

Analyze grasp-stage failures from:

    models/hybrid_sac_grasp_100/hybrid_grasp_100.csv

Purpose:
    - isolate the failed grasp cases
    - compare successful vs failed object positions
    - compare alignment/descent errors
    - compare route type
    - identify failure concentration in XY
    - generate candidate correction ranges for adaptive grasp retry

Outputs:
    models/hybrid_sac_grasp_100/failure_analysis/

        grasp_failures.csv
        grasp_successes.csv
        summary.txt
        xy_region_stats.csv
        route_stats.csv

        grasp_xy.png
        x_vs_lift.png
        y_vs_lift.png
        alignment_vs_lift.png
        descent_vs_lift.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# Paths
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "models"
    / "hybrid_sac_grasp_100"
    / "hybrid_grasp_100.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "hybrid_sac_grasp_100"
    / "failure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# Helpers
# ================================================================

def section(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 72
    )

    print(title)

    print(
        "=" * 72
    )


def stats(
    series: pd.Series,
) -> str:

    values = (
        series
        .dropna()
    )

    if len(values) == 0:
        return "N/A"

    return (
        f"mean={values.mean():.4f}, "
        f"median={values.median():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}, "
        f"std={values.std():.4f}"
    )


# ================================================================
# Main
# ================================================================

def main() -> None:

    section(
        "Hybrid SAC Grasp Failure Analysis"
    )

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"CSV not found:\n"
            f"{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        "\nLoaded:",
        INPUT_CSV
    )

    print(
        "Episodes:",
        len(df)
    )

    # ============================================================
    # Basic cleanup
    # ============================================================

    df[
        "pregrasp_success"
    ] = (
        df[
            "pregrasp_success"
        ]
        .astype(int)
    )

    df[
        "grasp_success"
    ] = (
        df[
            "grasp_success"
        ]
        .astype(int)
    )

    df[
        "full_pipeline_success"
    ] = (
        df[
            "full_pipeline_success"
        ]
        .astype(int)
    )

    # Only analyze actual grasp attempts.
    attempts = df[
        df[
            "grasp_attempted"
        ] == 1
    ].copy()

    successes = attempts[
        attempts[
            "grasp_success"
        ] == 1
    ].copy()

    failures = attempts[
        attempts[
            "grasp_success"
        ] == 0
    ].copy()

    # ============================================================
    # Overall
    # ============================================================

    section(
        "Overall Grasp Performance"
    )

    print(
        f"Grasp attempts: "
        f"{len(attempts)}"
    )

    print(
        f"Successful grasps: "
        f"{len(successes)}"
    )

    print(
        f"Failed grasps: "
        f"{len(failures)}"
    )

    if len(attempts) > 0:

        grasp_rate = (
            100.0
            * len(successes)
            / len(attempts)
        )

    else:

        grasp_rate = 0.0

    print(
        f"Grasp success rate: "
        f"{grasp_rate:.1f}%"
    )

    # ============================================================
    # Failed cases
    # ============================================================

    section(
        "Failed Grasp Cases"
    )

    columns = [
        "episode",
        "policy",
        "object_x",
        "object_y",
        "object_z",
        "final_pregrasp_distance",
        "alignment_error",
        "descent_error",
        "lift_tracking_error",
        "object_lift_m",
        "object_xy_motion_m",
        "final_hand_object_distance_m",
    ]

    if len(failures) > 0:

        print(
            failures[
                columns
            ].to_string(
                index=False
            )
        )

    # ============================================================
    # Object position comparison
    # ============================================================

    section(
        "Object Position Comparison"
    )

    for coordinate in [
        "object_x",
        "object_y",
    ]:

        print(
            f"\n{coordinate}"
        )

        print(
            "  successful:",
            stats(
                successes[
                    coordinate
                ]
            )
        )

        print(
            "  failed:    ",
            stats(
                failures[
                    coordinate
                ]
            )
        )

    # ============================================================
    # Pre-grasp comparison
    # ============================================================

    section(
        "Pre-Grasp Error Comparison"
    )

    print(
        "Successful:",
        stats(
            successes[
                "final_pregrasp_distance"
            ]
        )
    )

    print(
        "Failed:    ",
        stats(
            failures[
                "final_pregrasp_distance"
            ]
        )
    )

    # ============================================================
    # Alignment comparison
    # ============================================================

    section(
        "Alignment Error Comparison"
    )

    print(
        "Successful:",
        stats(
            successes[
                "alignment_error"
            ]
        )
    )

    print(
        "Failed:    ",
        stats(
            failures[
                "alignment_error"
            ]
        )
    )

    # ============================================================
    # Descent comparison
    # ============================================================

    section(
        "Descent Error Comparison"
    )

    print(
        "Successful:",
        stats(
            successes[
                "descent_error"
            ]
        )
    )

    print(
        "Failed:    ",
        stats(
            failures[
                "descent_error"
            ]
        )
    )

    # ============================================================
    # Lift tracking comparison
    # ============================================================

    section(
        "Lift Tracking Error Comparison"
    )

    print(
        "Successful:",
        stats(
            successes[
                "lift_tracking_error"
            ]
        )
    )

    print(
        "Failed:    ",
        stats(
            failures[
                "lift_tracking_error"
            ]
        )
    )

    # ============================================================
    # Route analysis
    # ============================================================

    section(
        "Policy Route Analysis"
    )

    route_rows = []

    for policy_name, group in attempts.groupby(
        "policy"
    ):

        samples = len(group)

        success_count = int(
            group[
                "grasp_success"
            ].sum()
        )

        failure_count = (
            samples
            - success_count
        )

        rate = (
            100.0
            * success_count
            / samples
            if samples > 0
            else 0.0
        )

        route_rows.append(
            {
                "policy":
                    policy_name,

                "samples":
                    samples,

                "successes":
                    success_count,

                "failures":
                    failure_count,

                "success_rate_percent":
                    rate,
            }
        )

    route_stats = pd.DataFrame(
        route_rows
    )

    print(
        route_stats.to_string(
            index=False
        )
    )

    # ============================================================
    # XY region analysis
    # ============================================================

    section(
        "XY Region Analysis"
    )

    x_bins = np.linspace(
        0.40,
        0.60,
        5,
    )

    y_bins = np.linspace(
        -0.20,
        0.20,
        5,
    )

    attempts[
        "x_region"
    ] = pd.cut(
        attempts[
            "object_x"
        ],
        bins=x_bins,
        include_lowest=True,
    )

    attempts[
        "y_region"
    ] = pd.cut(
        attempts[
            "object_y"
        ],
        bins=y_bins,
        include_lowest=True,
    )

    region_rows = []

    # ------------------------------------------------------------
    # X regions
    # ------------------------------------------------------------

    for region, group in attempts.groupby(
        "x_region",
        observed=False,
    ):

        if len(group) == 0:
            continue

        success_count = int(
            group[
                "grasp_success"
            ].sum()
        )

        region_rows.append(
            {
                "axis":
                    "x",

                "region":
                    str(region),

                "samples":
                    len(group),

                "successes":
                    success_count,

                "failures":
                    len(group)
                    - success_count,

                "success_rate_percent":
                    100.0
                    * group[
                        "grasp_success"
                    ].mean(),

                "mean_object_lift":
                    group[
                        "object_lift_m"
                    ].mean(),
            }
        )

    # ------------------------------------------------------------
    # Y regions
    # ------------------------------------------------------------

    for region, group in attempts.groupby(
        "y_region",
        observed=False,
    ):

        if len(group) == 0:
            continue

        success_count = int(
            group[
                "grasp_success"
            ].sum()
        )

        region_rows.append(
            {
                "axis":
                    "y",

                "region":
                    str(region),

                "samples":
                    len(group),

                "successes":
                    success_count,

                "failures":
                    len(group)
                    - success_count,

                "success_rate_percent":
                    100.0
                    * group[
                        "grasp_success"
                    ].mean(),

                "mean_object_lift":
                    group[
                        "object_lift_m"
                    ].mean(),
            }
        )

    region_stats = pd.DataFrame(
        region_rows
    )

    print(
        region_stats.to_string(
            index=False
        )
    )

    # ============================================================
    # 2D grid
    # ============================================================

    section(
        "2D XY Failure Grid"
    )

    grid_rows = []

    for (
        x_region,
        y_region,
    ), group in attempts.groupby(
        [
            "x_region",
            "y_region",
        ],
        observed=False,
    ):

        if len(group) == 0:
            continue

        success_count = int(
            group[
                "grasp_success"
            ].sum()
        )

        grid_rows.append(
            {
                "x_region":
                    str(
                        x_region
                    ),

                "y_region":
                    str(
                        y_region
                    ),

                "samples":
                    len(group),

                "successes":
                    success_count,

                "failures":
                    len(group)
                    - success_count,

                "success_rate_percent":
                    100.0
                    * group[
                        "grasp_success"
                    ].mean(),

                "mean_object_lift":
                    group[
                        "object_lift_m"
                    ].mean(),
            }
        )

    grid_stats = pd.DataFrame(
        grid_rows
    )

    if len(grid_stats) > 0:

        grid_stats = (
            grid_stats
            .sort_values(
                [
                    "success_rate_percent",
                    "samples",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )

        print(
            grid_stats.to_string(
                index=False
            )
        )

    # ============================================================
    # Suggested correction region
    # ============================================================

    section(
        "Suggested Failure-Focused Grasp Region"
    )

    if len(failures) > 0:

        x_low = float(
            failures[
                "object_x"
            ].quantile(
                0.10
            )
        )

        x_high = float(
            failures[
                "object_x"
            ].quantile(
                0.90
            )
        )

        y_low = float(
            failures[
                "object_y"
            ].quantile(
                0.10
            )
        )

        y_high = float(
            failures[
                "object_y"
            ].quantile(
                0.90
            )
        )

        print(
            f"X: "
            f"[{x_low:.3f}, "
            f"{x_high:.3f}]"
        )

        print(
            f"Y: "
            f"[{y_low:.3f}, "
            f"{y_high:.3f}]"
        )

        print(
            "\nFailure center:"
        )

        print(
            f"X mean: "
            f"{failures['object_x'].mean():.3f}"
        )

        print(
            f"Y mean: "
            f"{failures['object_y'].mean():.3f}"
        )

    # ============================================================
    # Save CSVs
    # ============================================================

    failures.to_csv(
        OUTPUT_DIR
        / "grasp_failures.csv",
        index=False,
    )

    successes.to_csv(
        OUTPUT_DIR
        / "grasp_successes.csv",
        index=False,
    )

    region_stats.to_csv(
        OUTPUT_DIR
        / "xy_region_stats.csv",
        index=False,
    )

    grid_stats.to_csv(
        OUTPUT_DIR
        / "xy_grid_stats.csv",
        index=False,
    )

    route_stats.to_csv(
        OUTPUT_DIR
        / "route_stats.csv",
        index=False,
    )

    # ============================================================
    # Plot: XY
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        successes[
            "object_x"
        ],
        successes[
            "object_y"
        ],
        label="Success",
        alpha=0.7,
    )

    plt.scatter(
        failures[
            "object_x"
        ],
        failures[
            "object_y"
        ],
        marker="x",
        s=100,
        label="Failure",
    )

    plt.xlabel(
        "Object X [m]"
    )

    plt.ylabel(
        "Object Y [m]"
    )

    plt.title(
        "Hybrid SAC Grasp Outcomes"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "grasp_xy.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # X vs lift
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        attempts[
            "object_x"
        ],
        attempts[
            "object_lift_m"
        ],
        alpha=0.7,
    )

    plt.axhline(
        0.05,
        linestyle="--",
        label="Lift success threshold",
    )

    plt.xlabel(
        "Object X [m]"
    )

    plt.ylabel(
        "Object Lift [m]"
    )

    plt.title(
        "Object X vs Grasp Lift"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "x_vs_lift.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Y vs lift
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        attempts[
            "object_y"
        ],
        attempts[
            "object_lift_m"
        ],
        alpha=0.7,
    )

    plt.axhline(
        0.05,
        linestyle="--",
        label="Lift success threshold",
    )

    plt.xlabel(
        "Object Y [m]"
    )

    plt.ylabel(
        "Object Lift [m]"
    )

    plt.title(
        "Object Y vs Grasp Lift"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "y_vs_lift.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Alignment vs lift
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        attempts[
            "alignment_error"
        ],
        attempts[
            "object_lift_m"
        ],
        alpha=0.7,
    )

    plt.xlabel(
        "Alignment Error [m]"
    )

    plt.ylabel(
        "Object Lift [m]"
    )

    plt.title(
        "Alignment Error vs Object Lift"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "alignment_vs_lift.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Descent vs lift
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        attempts[
            "descent_error"
        ],
        attempts[
            "object_lift_m"
        ],
        alpha=0.7,
    )

    plt.xlabel(
        "Descent Error [m]"
    )

    plt.ylabel(
        "Object Lift [m]"
    )

    plt.title(
        "Descent Error vs Object Lift"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "descent_vs_lift.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Summary file
    # ============================================================

    summary_path = (
        OUTPUT_DIR
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
    ) as file:

        file.write(
            "Hybrid SAC Grasp Failure Analysis\n"
        )

        file.write(
            "=================================\n\n"
        )

        file.write(
            f"Attempts: "
            f"{len(attempts)}\n"
        )

        file.write(
            f"Successes: "
            f"{len(successes)}\n"
        )

        file.write(
            f"Failures: "
            f"{len(failures)}\n"
        )

        file.write(
            f"Success rate: "
            f"{grasp_rate:.1f}%\n\n"
        )

        if len(failures) > 0:

            file.write(
                "Failure object range:\n"
            )

            file.write(
                f"X: "
                f"{failures['object_x'].min():.4f} "
                f"to "
                f"{failures['object_x'].max():.4f}\n"
            )

            file.write(
                f"Y: "
                f"{failures['object_y'].min():.4f} "
                f"to "
                f"{failures['object_y'].max():.4f}\n\n"
            )

            file.write(
                "Failure pre-grasp error:\n"
            )

            file.write(
                stats(
                    failures[
                        "final_pregrasp_distance"
                    ]
                )
            )

            file.write(
                "\n\n"
            )

            file.write(
                "Failure alignment error:\n"
            )

            file.write(
                stats(
                    failures[
                        "alignment_error"
                    ]
                )
            )

            file.write(
                "\n\n"
            )

            file.write(
                "Failure descent error:\n"
            )

            file.write(
                stats(
                    failures[
                        "descent_error"
                    ]
                )
            )

            file.write(
                "\n"
            )

    # ============================================================
    # Done
    # ============================================================

    section(
        "Files Saved"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n- grasp_failures.csv"
    )

    print(
        "- grasp_successes.csv"
    )

    print(
        "- xy_region_stats.csv"
    )

    print(
        "- xy_grid_stats.csv"
    )

    print(
        "- route_stats.csv"
    )

    print(
        "- summary.txt"
    )

    print(
        "- grasp_xy.png"
    )

    print(
        "- x_vs_lift.png"
    )

    print(
        "- y_vs_lift.png"
    )

    print(
        "- alignment_vs_lift.png"
    )

    print(
        "- descent_vs_lift.png"
    )

    print(
        "\nGrasp failure analysis completed."
    )


if __name__ == "__main__":
    main()
