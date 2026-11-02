
import pandas as pd
import numpy as np

DATA_PATH = "outputs/grid_alignment_detailed_fixed.csv"
OUTPUT_PATH = "outputs/grid_alignment_accuracy_report.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    reference_column = "fmax_measured_windowed_mhz"

    methods = ["ceiling", "floor", "nearest"]

    results = []

    print("=" * 75)
    print("GRID ALIGNMENT - MEASURED FMAX ACCURACY ANALYSIS")
    print("=" * 75)

    for method in methods:

        predicted_column = f"{method}_measured_fmax"

        actual = df[reference_column]
        predicted = df[predicted_column]

        valid = actual.notna() & predicted.notna()

        actual_valid = actual[valid]
        predicted_valid = predicted[valid]

        differences = predicted_valid - actual_valid
        absolute_errors = differences.abs()

        exact_matches = (differences == 0).sum()

        results.append({
            "Method": method.capitalize(),
            "Total Chips": len(df),
            "Valid Comparisons": valid.sum(),
            "Exact Fmax Matches": exact_matches,
            "Different Fmax Results": valid.sum() - exact_matches,
            "Mean Absolute Error (MHz)": absolute_errors.mean(),
            "Maximum Absolute Error (MHz)": absolute_errors.max()
        })

        print(f"\n{method.upper()} ALIGNMENT")
        print("-" * 50)
        print("Valid comparisons:", valid.sum())
        print("Exact matches:", exact_matches)
        print("Different results:", valid.sum() - exact_matches)
        print("Mean absolute error:", round(absolute_errors.mean(), 4))
        print("Maximum absolute error:", round(absolute_errors.max(), 4))

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("FINAL ACCURACY COMPARISON")
    print("=" * 75)

    print(result_df.to_string(index=False))

    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\nAccuracy report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
