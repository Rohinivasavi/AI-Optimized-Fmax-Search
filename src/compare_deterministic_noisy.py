
import pandas as pd
import numpy as np

DETERMINISTIC_PATH = "outputs/ml_windowed_search_results.csv"
NOISY_PATH = "outputs/ml_windowed_search_noisy_results.csv"

def main():

    deterministic = pd.read_csv(DETERMINISTIC_PATH)
    noisy = pd.read_csv(NOISY_PATH)

    # Verify both experiments use the same chips
    if not deterministic["unit_id"].equals(noisy["unit_id"]):
        raise ValueError("Chip order or unit IDs do not match.")

    actual = noisy["true_fmax_mhz"]

    deterministic_error = (
        deterministic["measured_fmax_mhz"] - actual
    ).abs()

    noisy_error = (
        noisy["measured_fmax_mhz"] - actual
    ).abs()

    print("=" * 70)
    print("DETERMINISTIC VS NOISY SEARCH COMPARISON")
    print("=" * 70)

    print("\n1. TEST EFFICIENCY")
    print("-" * 50)

    print(
        "Deterministic average tests:",
        round(deterministic["tests_required"].mean(), 4)
    )

    print(
        "Noisy average tests:",
        round(noisy["tests_required"].mean(), 4)
    )

    print(
        "Deterministic maximum tests:",
        deterministic["tests_required"].max()
    )

    print(
        "Noisy maximum tests:",
        noisy["tests_required"].max()
    )

    print("\n2. FMAX ACCURACY")
    print("-" * 50)

    print(
        "Deterministic MAE:",
        round(deterministic_error.mean(), 4),
        "MHz"
    )

    print(
        "Noisy MAE:",
        round(noisy_error.mean(), 4),
        "MHz"
    )

    print(
        "Deterministic median error:",
        round(deterministic_error.median(), 4),
        "MHz"
    )

    print(
        "Noisy median error:",
        round(noisy_error.median(), 4),
        "MHz"
    )

    print("\n3. CHIP-BY-CHIP ACCURACY")
    print("-" * 50)

    print(
        "Deterministic result closer:",
        (deterministic_error < noisy_error).sum()
    )

    print(
        "Noisy result closer:",
        (noisy_error < deterministic_error).sum()
    )

    print(
        "Equal accuracy:",
        (noisy_error == deterministic_error).sum()
    )

    print("\n4. TEST COUNT DIFFERENCE")
    print("-" * 50)

    test_difference = (
        noisy["tests_required"]
        - deterministic["tests_required"]
    )

    print("Average additional tests:",
          round(test_difference.mean(), 4))

    print("Chips requiring more tests:",
          (test_difference > 0).sum())

    print("Chips requiring fewer tests:",
          (test_difference < 0).sum())

    print("Chips requiring equal tests:",
          (test_difference == 0).sum())

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
