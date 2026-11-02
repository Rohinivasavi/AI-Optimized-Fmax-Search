
import pandas as pd
import numpy as np

DATA_PATH = "outputs/grid_alignment_detailed_fixed.csv"
OUTPUT_PATH = "outputs/remaining_floor_mismatches.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # Identify records where Floor test counts differ
    df["test_count_difference"] = (
        df["floor_tests"] - df["tests_windowed"]
    )

    mismatches = df[
        df["test_count_difference"] != 0
    ].copy()

    # Compare measured Fmax values
    mismatches["fmax_difference"] = (
        mismatches["floor_measured_fmax"]
        - mismatches["fmax_measured_windowed_mhz"]
    )

    mismatches["absolute_fmax_difference"] = (
        mismatches["fmax_difference"].abs()
    )

    # Check whether actual Fmax is inside the continuous window
    mismatches["actual_inside_window"] = (
        (mismatches["fmax_true_mhz"] >= mismatches["window_low_mhz"])
        &
        (mismatches["fmax_true_mhz"] <= mismatches["window_high_mhz"])
    )

    mismatches["actual_position"] = np.select(
        [
            mismatches["fmax_true_mhz"] < mismatches["window_low_mhz"],
            mismatches["fmax_true_mhz"] > mismatches["window_high_mhz"]
        ],
        [
            "Below window",
            "Above window"
        ],
        default="Inside window"
    )

    print("=" * 70)
    print("REMAINING FLOOR ALIGNMENT MISMATCH ANALYSIS")
    print("=" * 70)

    print("\nTOTAL RECORDS:", len(df))
    print("MATCHING COUNTS:", (df["floor_tests"] == df["tests_windowed"]).sum())
    print("MISMATCHING COUNTS:", len(mismatches))

    print("\n1. TEST COUNT DIFFERENCE DISTRIBUTION")
    print("-" * 50)

    print(
        mismatches["test_count_difference"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n2. ACTUAL FMAX POSITION")
    print("-" * 50)

    print(
        mismatches["actual_position"]
        .value_counts()
        .to_string()
    )

    print("\n3. MEASURED FMAX COMPARISON")
    print("-" * 50)

    print(
        "Exact measured Fmax matches:",
        (mismatches["fmax_difference"] == 0).sum()
    )

    print(
        "Different measured Fmax results:",
        (mismatches["fmax_difference"] != 0).sum()
    )

    print(
        "Average absolute Fmax difference:",
        round(mismatches["absolute_fmax_difference"].mean(), 4),
        "MHz"
    )

    print("\n4. MISMATCHES WITH IDENTICAL MEASURED FMAX")
    print("-" * 50)

    identical = mismatches[
        mismatches["fmax_difference"] == 0
    ]

    print("Count:", len(identical))

    print("\n5. FIRST 20 MISMATCH RECORDS")
    print("-" * 50)

    columns = [
        "unit_id",
        "fmax_true_mhz",
        "window_low_mhz",
        "window_high_mhz",
        "tests_windowed",
        "floor_tests",
        "test_count_difference",
        "fmax_measured_windowed_mhz",
        "floor_measured_fmax",
        "actual_position"
    ]

    print(
        mismatches[columns]
        .head(20)
        .to_string(index=False)
    )

    mismatches.to_csv(OUTPUT_PATH, index=False)

    print("\nDetailed mismatch report saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
