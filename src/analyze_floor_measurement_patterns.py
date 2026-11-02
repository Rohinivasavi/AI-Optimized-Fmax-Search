
import pandas as pd
import numpy as np

DATA_PATH = "outputs/remaining_floor_mismatches.csv"
OUTPUT_PATH = "outputs/floor_measurement_patterns.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # Compare original recorded measurement with Floor result
    df["measurement_difference"] = (
        df["fmax_measured_windowed_mhz"]
        - df["floor_measured_fmax"]
    )

    df["absolute_difference"] = (
        df["measurement_difference"].abs()
    )

    print("=" * 70)
    print("FLOOR MEASUREMENT DIFFERENCE ANALYSIS")
    print("=" * 70)

    print("\n1. MEASUREMENT DIFFERENCE DISTRIBUTION")
    print("-" * 50)

    print(
        df["measurement_difference"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n2. DIFFERENCE STATISTICS")
    print("-" * 50)

    print("Mean difference:",
          round(df["measurement_difference"].mean(), 4), "MHz")

    print("Median difference:",
          round(df["measurement_difference"].median(), 4), "MHz")

    print("Mean absolute difference:",
          round(df["absolute_difference"].mean(), 4), "MHz")

    print("Maximum absolute difference:",
          round(df["absolute_difference"].max(), 4), "MHz")

    print("\n3. TEST COUNT DIFFERENCE VS MEASUREMENT DIFFERENCE")
    print("-" * 50)

    summary = df.groupby(
        "test_count_difference"
    ).agg(
        chip_count=("unit_id", "count"),
        mean_measurement_error=("absolute_difference", "mean"),
        exact_measurement_matches=(
            "absolute_difference",
            lambda x: (x == 0).sum()
        )
    )

    print(summary.round(3).to_string())

    print("\n4. FIRST 20 RECORDS")
    print("-" * 50)

    columns = [
        "unit_id",
        "tests_windowed",
        "floor_tests",
        "test_count_difference",
        "fmax_measured_windowed_mhz",
        "floor_measured_fmax",
        "measurement_difference"
    ]

    print(df[columns].head(20).to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDetailed report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
