# -*- coding: utf-8 -*-

"""
analyze_object_reach_failures.py

Analyze failures from deterministic object pre-grasp evaluation.

Input:
    models/sac_object_reach/evaluation_100_unseen_objects.csv

Outputs:
    models/sac_object_reach/failure_analysis/

        failure_cases.csv
        success_cases.csv
        xy_region_stats.csv
        summary.txt

        object_xy.png
        object_x_vs_error.png
        object_y_vs_error.png
        final_distance_histogram.png
        residual_axis_errors.png

Goals:
    - identify where object-position failures occur
    - measure residual X/Y/Z error
    - detect difficult object XY regions
    - prepare a targeted curriculum for retraining
"""

from __future__ import annotations

from pathlib import Path
import sys

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
    / "sac_object_reach"
    / "evaluation_100_unseen_objects.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "failure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# Helpers
# ================================================================

def print_section(title: str) -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


def stats_string(values: pd.Series) -> str:

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

    print_section(
        "SAC Object Pre-Grasp Failure Analysis"
    )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Evaluation CSV not found:\n"
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

    df["success"] = (
        df["success"]
        .astype(int)
    )

    # ============================================================
    # Residual errors relative to pre-grasp goal
    # ============================================================

    df["error_x"] = (
        df["goal_x"]
        - df["final_ee_x"]
    )

    df["error_y"] = (
        df["goal_y"]
        - df["final_ee_y"]
    )

    df["error_z"] = (
        df["goal_z"]
        - df["final_ee_z"]
    )

    df["abs_error_x"] = (
        df["error_x"].abs()
    )

    df["abs_error_y"] = (
        df["error_y"].abs()
    )

    df["abs_error_z"] = (
        df["error_z"].abs()
    )

    successes = df[
        df["success"] == 1
    ].copy()

    failures = df[
        df["success"] == 0
    ].copy()

    success_rate = (
        100.0
        * df["success"].mean()
    )

    # ============================================================
    # Overall
    # ============================================================

    print_section(
        "Overall Performance"
    )

    print(
        f"Successes: {len(successes)}"
    )

    print(
        f"Failures:  {len(failures)}"
    )

    print(
        f"Success rate: {success_rate:.1f}%"
    )

    print(
        "\nFinal pre-grasp distance:"
    )

    print(
        stats_string(
            df["final_distance_m"]
        )
    )

    # ============================================================
    # Failed object locations
    # ============================================================

    print_section(
        "Failure Object Locations"
    )

    if len(failures) > 0:

        print(
            "\nObject X:"
        )

        print(
            stats_string(
                failures["object_x"]
            )
        )

        print(
            "\nObject Y:"
        )

        print(
            stats_string(
                failures["object_y"]
            )
        )

    # ============================================================
    # Success vs failure object locations
    # ============================================================

    print_section(
        "Successful vs Failed Object Positions"
    )

    for coordinate in [
        "object_x",
        "object_y",
    ]:

        print(
            f"\n{coordinate}"
        )

        print(
            "  success:",
            stats_string(
                successes[
                    coordinate
                ]
            )
        )

        print(
            "  failure:",
            stats_string(
                failures[
                    coordinate
                ]
            )
        )

    # ============================================================
    # Residual-axis analysis
    # ============================================================

    print_section(
        "Failure Residual Error"
    )

    if len(failures) > 0:

        mean_abs_x = float(
            failures[
                "abs_error_x"
            ].mean()
        )

        mean_abs_y = float(
            failures[
                "abs_error_y"
            ].mean()
        )

        mean_abs_z = float(
            failures[
                "abs_error_z"
            ].mean()
        )

        print(
            f"Mean |X error|: "
            f"{mean_abs_x:.4f} m"
        )

        print(
            f"Mean |Y error|: "
            f"{mean_abs_y:.4f} m"
        )

        print(
            f"Mean |Z error|: "
            f"{mean_abs_z:.4f} m"
        )

        axis_errors = {
            "X": mean_abs_x,
            "Y": mean_abs_y,
            "Z": mean_abs_z,
        }

        dominant_axis = max(
            axis_errors,
            key=axis_errors.get,
        )

        print(
            "\nDominant residual axis:",
            dominant_axis
        )

    # ============================================================
    # Hardest failures
    # ============================================================

    print_section(
        "10 Hardest Object Placements"
    )

    hardest = (
        failures
        .sort_values(
            "final_distance_m",
            ascending=False,
        )
        .head(10)
    )

    columns = [
        "episode",
        "object_x",
        "object_y",
        "initial_distance_m",
        "final_distance_m",
        "error_x",
        "error_y",
        "error_z",
    ]

    if len(hardest) > 0:

        print(
            hardest[
                columns
            ].to_string(
                index=False
            )
        )

    # ============================================================
    # XY region analysis
    # ============================================================

    print_section(
        "Object XY Region Analysis"
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

    df["x_region"] = pd.cut(
        df["object_x"],
        bins=x_bins,
        include_lowest=True,
    )

    df["y_region"] = pd.cut(
        df["object_y"],
        bins=y_bins,
        include_lowest=True,
    )

    region_rows = []

    # ------------------------------------------------------------
    # X-axis regions
    # ------------------------------------------------------------

    grouped_x = df.groupby(
        "x_region",
        observed=False,
    )

    for region, group in grouped_x:

        if len(group) == 0:
            continue

        region_rows.append(
            {
                "axis":
                    "x",

                "region":
                    str(region),

                "samples":
                    len(group),

                "successes":
                    int(
                        group[
                            "success"
                        ].sum()
                    ),

                "failures":
                    int(
                        (
                            1
                            - group[
                                "success"
                            ]
                        ).sum()
                    ),

                "success_rate_percent":
                    100.0
                    * group[
                        "success"
                    ].mean(),

                "mean_final_distance":
                    group[
                        "final_distance_m"
                    ].mean(),
            }
        )

    # ------------------------------------------------------------
    # Y-axis regions
    # ------------------------------------------------------------

    grouped_y = df.groupby(
        "y_region",
        observed=False,
    )

    for region, group in grouped_y:

        if len(group) == 0:
            continue

        region_rows.append(
            {
                "axis":
                    "y",

                "region":
                    str(region),

                "samples":
                    len(group),

                "successes":
                    int(
                        group[
                            "success"
                        ].sum()
                    ),

                "failures":
                    int(
                        (
                            1
                            - group[
                                "success"
                            ]
                        ).sum()
                    ),

                "success_rate_percent":
                    100.0
                    * group[
                        "success"
                    ].mean(),

                "mean_final_distance":
                    group[
                        "final_distance_m"
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
    # 2D XY grid analysis
    # ============================================================

    print_section(
        "2D Object XY Failure Grid"
    )

    grid_rows = []

    grouped_grid = df.groupby(
        [
            "x_region",
            "y_region",
        ],
        observed=False,
    )

    for (
        x_region,
        y_region,
    ), group in grouped_grid:

        if len(group) == 0:
            continue

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
                    int(
                        group[
                            "success"
                        ].sum()
                    ),

                "failures":
                    int(
                        (
                            1
                            - group[
                                "success"
                            ]
                        ).sum()
                    ),

                "success_rate_percent":
                    100.0
                    * group[
                        "success"
                    ].mean(),

                "mean_final_distance":
                    group[
                        "final_distance_m"
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
                    "mean_final_distance",
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
    # Suggested failure-focused range
    # ============================================================

    print_section(
        "Suggested Failure-Focused Object Range"
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
    # Save CSV outputs
    # ============================================================

    failures.to_csv(
        OUTPUT_DIR
        / "failure_cases.csv",
        index=False,
    )

    successes.to_csv(
        OUTPUT_DIR
        / "success_cases.csv",
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

    # ============================================================
    # Plot: object XY
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
        s=80,
        label="Failure",
    )

    plt.xlabel(
        "Object X [m]"
    )

    plt.ylabel(
        "Object Y [m]"
    )

    plt.title(
        "Object Pre-Grasp Evaluation – XY"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "object_xy.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Plot: object X vs error
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        df[
            "object_x"
        ],
        df[
            "final_distance_m"
        ],
        alpha=0.7,
    )

    plt.axhline(
        0.040,
        linestyle="--",
        label="Success threshold",
    )

    plt.xlabel(
        "Object X [m]"
    )

    plt.ylabel(
        "Final Pre-Grasp Error [m]"
    )

    plt.title(
        "Object X vs Final Error"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "object_x_vs_error.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Plot: object Y vs error
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.scatter(
        df[
            "object_y"
        ],
        df[
            "final_distance_m"
        ],
        alpha=0.7,
    )

    plt.axhline(
        0.040,
        linestyle="--",
        label="Success threshold",
    )

    plt.xlabel(
        "Object Y [m]"
    )

    plt.ylabel(
        "Final Pre-Grasp Error [m]"
    )

    plt.title(
        "Object Y vs Final Error"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "object_y_vs_error.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Histogram
    # ============================================================

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.hist(
        df[
            "final_distance_m"
        ],
        bins=20,
        alpha=0.8,
    )

    plt.axvline(
        0.040,
        linestyle="--",
        label="Success threshold",
    )

    plt.xlabel(
        "Final Pre-Grasp Error [m]"
    )

    plt.ylabel(
        "Episodes"
    )

    plt.title(
        "Object Pre-Grasp Final Error Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_distance_histogram.png",
        dpi=180,
    )

    plt.close()

    # ============================================================
    # Residual-axis plot
    # ============================================================

    axis_names = [
        "X",
        "Y",
        "Z",
    ]

    if len(failures) > 0:

        axis_values = [
            mean_abs_x,
            mean_abs_y,
            mean_abs_z,
        ]

        plt.figure(
            figsize=(
                7,
                5,
            )
        )

        plt.bar(
            axis_names,
            axis_values,
        )

        plt.ylabel(
            "Mean Absolute Error [m]"
        )

        plt.title(
            "Failed Episodes – Residual Axis Error"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / "residual_axis_errors.png",
            dpi=180,
        )

        plt.close()

    # ============================================================
    # Summary text
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
            "SAC Object Pre-Grasp Failure Analysis\n"
        )

        file.write(
            "=====================================\n\n"
        )

        file.write(
            f"Episodes: "
            f"{len(df)}\n"
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
            f"{success_rate:.1f}%\n\n"
        )

        file.write(
            "Final distance:\n"
        )

        file.write(
            stats_string(
                df[
                    "final_distance_m"
                ]
            )
        )

        file.write(
            "\n\n"
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
                "Failure residual error:\n"
            )

            file.write(
                f"X: "
                f"{mean_abs_x:.4f} m\n"
            )

            file.write(
                f"Y: "
                f"{mean_abs_y:.4f} m\n"
            )

            file.write(
                f"Z: "
                f"{mean_abs_z:.4f} m\n"
            )

            file.write(
                f"\nDominant residual axis: "
                f"{dominant_axis}\n"
            )

    # ============================================================
    # Done
    # ============================================================

    print_section(
        "Files Saved"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n- failure_cases.csv"
    )

    print(
        "- success_cases.csv"
    )

    print(
        "- xy_region_stats.csv"
    )

    print(
        "- xy_grid_stats.csv"
    )

    print(
        "- summary.txt"
    )

    print(
        "- object_xy.png"
    )

    print(
        "- object_x_vs_error.png"
    )

    print(
        "- object_y_vs_error.png"
    )

    print(
        "- final_distance_histogram.png"
    )

    print(
        "- residual_axis_errors.png"
    )

    print(
        "\nFailure analysis completed."
    )


if __name__ == "__main__":
    main()
