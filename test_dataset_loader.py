# -*- coding: utf-8 -*-
"""
test_dataset_loader.py

Quick test for Phase 7:
Load saved VLA demonstrations and convert them into
behavior-cloning-ready arrays.
"""

from dataset.dataset_loader import (
    load_all_demos,
    build_behavior_cloning_arrays,
    print_dataset_summary,
)


def main() -> None:
    demos = load_all_demos("datasets")
    print(f"Loaded {len(demos)} demos.")

    X, Y, _ = build_behavior_cloning_arrays("datasets")

    print("\nX:")
    print(X)

    print("\nY:")
    print(Y)

    print_dataset_summary("datasets")


if __name__ == "__main__":
    main()
