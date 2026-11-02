
import pandas as pd
import numpy as np

DATA_PATH = "outputs/grid_alignment_detailed_fixed.csv"
OUTPUT_PATH = "outputs/final_floor_performance_report.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # Test count comparison
    binary_tests = df["tests_binary"]
    reference_windowed_tests = df["tests_windowed"]
    floor_tests = df["floor_tests"]

    # Accuracy comparison
    actual = df["fmax_true_mhz"]
    reference_fmax = df["fmax_measured_windowed_mhz"]
    floor_fmax = df["floor_measured_fmax"]

    reference_mae = (reference_fmax - actual).abs().mean()
    floor_mae = (floor_fmax - actual).abs().mean()

    test_reduction = (
        (binary_tests.mean() - floor_tests.mean())
        / binary_tests.mean()
    ) * 100

    reference_count_matches = (
        floor_tests == reference_windowed_tests
    ).sum()

    print("=" * 70)
    print("FINAL FLOOR SEARCH PERFORMANCE EVALUATION")
    print("=" * 70)

    print("\n1. TEST EFFICIENCY")
    print("-" * 50)

    print("Total chips:", len(df))
    print("Binary search average:", round(binary_tests.mean(), 4))
    print("Reference windowed average:",
          round(reference_windowed_tests.mean(), 4))
    print("Floor search average:", round(floor_tests.mean(), 4))

    print("Maximum Floor tests:", floor_tests.max())

    print("Test reduction vs binary search:",
          round(test_reduction, 2), "%")

    print("\n2. REFERENCE COUNT AGREEMENT")
    print("-" * 50)

    print("Matching counts:", reference_count_matches)
    print("Mismatching counts:",
          len(df) - reference_count_matches)

    print("Match percentage:",
          round(reference_count_matches / len(df) * 100, 2), "%")

    print("\n3. FMAX ACCURACY")
    print("-" * 50)

    print("Original reference MAE:",
          round(reference_mae, 4), "MHz")

    print("Floor algorithm MAE:",
          round(floor_mae, 4), "MHz")

    improvement = (
        (reference_mae - floor_mae) / reference_mae
    ) * 100

    print("MAE improvement:",
          round(improvement, 2), "%")

    print("\n4. FINAL SUMMARY")
    print("-" * 50)

    summary = pd.DataFrame([
        {
            "Method": "Binary Search",
            "Average Tests": binary_tests.mean(),
            "Fmax MAE (MHz)": np.nan
        },
        {
            "Method": "Reference Windowed Search",
            "Average Tests": reference_windowed_tests.mean(),
            "Fmax MAE (MHz)": reference_mae
        },
        {
            "Method": "Floor Alignment Search",
            "Average Tests": floor_tests.mean(),
            "Fmax MAE (MHz)": floor_mae
        }
    ])

    print(summary.round(4).to_string(index=False))

    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nReport saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
