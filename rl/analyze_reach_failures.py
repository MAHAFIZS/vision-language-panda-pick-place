# -*- coding: utf-8 -*-

"""
analyze_reach_failures.py

Analyze deterministic SAC evaluation results.

Input:
    models/sac_reach/evaluation_100_unseen.csv

Outputs:
    models/sac_reach/failure_analysis/

        failure_cases.csv
        success_cases.csv
        target_region_stats.csv
        summary.txt

        target_xy.png
        target_xz.png
        target_yz.png
        final_distance_histogram.png
        initial_vs_final_distance.png

Goals:
    - determine where failures occur
    - detect workspace-edge problems
    - measure residual XYZ error
    - identify hard target regions
    - generate target ranges for targeted retraining
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
    / "sac_reach"
    / "evaluation_100_unseen.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sac_reach"
    / "failure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# Helper functions
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


def stats_string(
    values: pd.Series,
) -> str:

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
        "SAC Panda Reach Failure Analysis"
    )

    # ------------------------------------------------------------
    # Load CSV
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

    # ------------------------------------------------------------
    # Ensure success is integer/bool
    # ------------------------------------------------------------

    df["success"] = (
        df["success"]
        .astype(int)
    )

    successes = df[
        df["success"] == 1
    ].copy()

    failures = df[
        df["success"] == 0
    ].copy()

    # ------------------------------------------------------------
    # Residual errors
    # ------------------------------------------------------------

    df["error_x"] = (
        df["target_x"]
        - df["final_ee_x"]
    )

    df["error_y"] = (
        df["target_y"]
        - df["final_ee_y"]
    )

    df["error_z"] = (
        df["target_z"]
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

    # Recreate subsets after adding errors.

    successes = df[
        df["success"] == 1
    ].copy()

    failures = df[
        df["success"] == 0
    ].copy()

    # ------------------------------------------------------------
    # Basic performance
    # ------------------------------------------------------------

    success_rate = (
        100.0
        * df["success"].mean()
    )

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
        "\nFinal distance:"
    )

    print(
        stats_string(
            df["final_distance_m"]
        )
    )

    # ============================================================
    # Failure target coordinates
    # ============================================================

    print_section(
        "Failure Target Locations"
    )

    if len(failures) > 0:

        print(
            "\nTarget X:"
        )

        print(
            stats_string(
                failures["target_x"]
            )
        )

        print(
            "\nTarget Y:"
        )

        print(
            stats_string(
                failures["target_y"]
            )
        )

        print(
            "\nTarget Z:"
        )

        print(
            stats_string(
                failures["target_z"]
            )
        )

    # ============================================================
    # Success vs failure comparison
    # ============================================================

    print_section(
        "Successful vs Failed Target Comparison"
    )

    for coordinate in [
        "target_x",
        "target_y",
        "target_z",
    ]:

        print(
            f"\n{coordinate}"
        )

        print(
            "  success:",
            stats_string(
                successes[coordinate]
            )
        )

        print(
            "  failure:",
            stats_string(
                failures[coordinate]
            )
        )

    # ============================================================
    # Residual-axis error analysis
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

        worst_axis = max(
            axis_errors,
            key=axis_errors.get,
        )

        print(
            "\nDominant residual axis:",
            worst_axis
        )

    # ============================================================
    # Hardest failures
    # ============================================================

    print_section(
        "10 Hardest Failed Targets"
    )

    hardest = (
        failures
        .sort_values(
            "final_distance_m",
            ascending=False,
        )
        .head(10)
    )

    columns_to_print = [
        "episode",
        "target_x",
        "target_y",
        "target_z",
        "initial_distance_m",
        "final_distance_m",
        "error_x",
        "error_y",
        "error_z",
    ]

    if len(hardest) > 0:

        print(
            hardest[
                columns_to_print
            ].to_string(
                index=False
            )
        )

    # ============================================================
    # Workspace bins
    # ============================================================

    print_section(
        "Workspace Region Analysis"
    )

    # Same general ranges used by training environment.
    x_bins = np.linspace(
        0.35,
        0.65,
        4,
    )

    y_bins = np.linspace(
        -0.25,
        0.25,
        4,
    )

    z_bins = np.linspace(
        0.25,
        0.50,
        4,
    )

    df["x_region"] = pd.cut(
        df["target_x"],
        bins=x_bins,
        include_lowest=True,
    )

    df["y_region"] = pd.cut(
        df["target_y"],
        bins=y_bins,
        include_lowest=True,
    )

    df["z_region"] = pd.cut(
        df["target_z"],
        bins=z_bins,
        include_lowest=True,
    )

    region_rows = []

    # ------------------------------------------------------------
    # Individual-axis regions
    # ------------------------------------------------------------

    for axis_name in [
        "x_region",
        "y_region",
        "z_region",
    ]:

        grouped = df.groupby(
            axis_name,
            observed=False,
        )

        for region, group in grouped:

            if len(group) == 0:
                continue

            region_rows.append(
                {
                    "axis": axis_name,
                    "region": str(region),
                    "samples": len(group),
                    "successes": int(
                        group["success"].sum()
                    ),
                    "failures": int(
                        (
                            1
                            - group["success"]
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
    # Identify difficult coordinate ranges
    # ============================================================

    print_section(
        "Suggested Hard-Target Ranges"
    )

    if len(failures) > 0:

        # Use failure quantiles rather than min/max,
        # avoiding single extreme outliers.

        x_low = float(
            failures[
                "target_x"
            ].quantile(0.10)
        )

        x_high = float(
            failures[
                "target_x"
            ].quantile(0.90)
        )

        y_low = float(
            failures[
                "target_y"
            ].quantile(0.10)
        )

        y_high = float(
            failures[
                "target_y"
            ].quantile(0.90)
        )

        z_low = float(
            failures[
                "target_z"
            ].quantile(0.10)
        )

        z_high = float(
            failures[
                "target_z"
            ].quantile(0.90)
        )

        print(
            "Approximate failure-focused range:"
        )

        print(
            f"X: [{x_low:.3f}, "
            f"{x_high:.3f}]"
        )

        print(
            f"Y: [{y_low:.3f}, "
            f"{y_high:.3f}]"
        )

        print(
            f"Z: [{z_low:.3f}, "
            f"{z_high:.3f}]"
        )

    # ============================================================
    # Save cases
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
        / "target_region_stats.csv",
        index=False,
    )

    # ============================================================
    # Plots
    # ============================================================

    # ------------------------------------------------------------
    # XY
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        successes["target_x"],
        successes["target_y"],
        label="Success",
        alpha=0.7,
    )

    plt.scatter(
        failures["target_x"],
        failures["target_y"],
        marker="x",
        s=80,
        label="Failure",
    )

    plt.xlabel(
        "Target X [m]"
    )

    plt.ylabel(
        "Target Y [m]"
    )

    plt.title(
        "SAC Reach Evaluation – Target XY"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "target_xy.png",
        dpi=180,
    )

    plt.close()

    # ------------------------------------------------------------
    # XZ
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        successes["target_x"],
        successes["target_z"],
        label="Success",
        alpha=0.7,
    )

    plt.scatter(
        failures["target_x"],
        failures["target_z"],
        marker="x",
        s=80,
        label="Failure",
    )

    plt.xlabel(
        "Target X [m]"
    )

    plt.ylabel(
        "Target Z [m]"
    )

    plt.title(
        "SAC Reach Evaluation – Target XZ"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "target_xz.png",
        dpi=180,
    )

    plt.close()

    # ------------------------------------------------------------
    # YZ
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        successes["target_y"],
        successes["target_z"],
        label="Success",
        alpha=0.7,
    )

    plt.scatter(
        failures["target_y"],
        failures["target_z"],
        marker="x",
        s=80,
        label="Failure",
    )

    plt.xlabel(
        "Target Y [m]"
    )

    plt.ylabel(
        "Target Z [m]"
    )

    plt.title(
        "SAC Reach Evaluation – Target YZ"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "target_yz.png",
        dpi=180,
    )

    plt.close()

    # ------------------------------------------------------------
    # Final distance distribution
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        df[
            "final_distance_m"
        ],
        bins=20,
        alpha=0.8,
    )

    plt.axvline(
        0.035,
        linestyle="--",
        label="Success threshold 0.035 m",
    )

    plt.xlabel(
        "Final Distance [m]"
    )

    plt.ylabel(
        "Episodes"
    )

    plt.title(
        "Final Distance Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_distance_histogram.png",
        dpi=180,
    )

    plt.close()

    # ------------------------------------------------------------
    # Initial vs final distance
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        successes[
            "initial_distance_m"
        ],
        successes[
            "final_distance_m"
        ],
        label="Success",
        alpha=0.7,
    )

    plt.scatter(
        failures[
            "initial_distance_m"
        ],
        failures[
            "final_distance_m"
        ],
        marker="x",
        s=80,
        label="Failure",
    )

    plt.axhline(
        0.035,
        linestyle="--",
        label="Success threshold",
    )

    plt.xlabel(
        "Initial Distance [m]"
    )

    plt.ylabel(
        "Final Distance [m]"
    )

    plt.title(
        "Initial vs Final Distance"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "initial_vs_final_distance.png",
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
            "SAC Panda Reach Failure Analysis\n"
        )

        file.write(
            "===============================\n\n"
        )

        file.write(
            f"Episodes: {len(df)}\n"
        )

        file.write(
            f"Successes: {len(successes)}\n"
        )

        file.write(
            f"Failures: {len(failures)}\n"
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
                df["final_distance_m"]
            )
        )

        file.write(
            "\n\n"
        )

        if len(failures) > 0:

            file.write(
                "Failure target ranges:\n"
            )

            file.write(
                f"X: "
                f"{failures['target_x'].min():.4f} "
                f"to "
                f"{failures['target_x'].max():.4f}\n"
            )

            file.write(
                f"Y: "
                f"{failures['target_y'].min():.4f} "
                f"to "
                f"{failures['target_y'].max():.4f}\n"
            )

            file.write(
                f"Z: "
                f"{failures['target_z'].min():.4f} "
                f"to "
                f"{failures['target_z'].max():.4f}\n"
            )

            file.write(
                "\nMean absolute residual errors:\n"
            )

            file.write(
                f"X: {mean_abs_x:.4f} m\n"
            )

            file.write(
                f"Y: {mean_abs_y:.4f} m\n"
            )

            file.write(
                f"Z: {mean_abs_z:.4f} m\n"
            )

            file.write(
                f"\nDominant residual axis: "
                f"{worst_axis}\n"
            )

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
        "- target_region_stats.csv"
    )

    print(
        "- summary.txt"
    )

    print(
        "- target_xy.png"
    )

    print(
        "- target_xz.png"
    )

    print(
        "- target_yz.png"
    )

    print(
        "- final_distance_histogram.png"
    )

    print(
        "- initial_vs_final_distance.png"
    )

    print(
        "\nFailure analysis completed."
    )


if __name__ == "__main__":
    main()
