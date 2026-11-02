
import pandas as pd
import numpy as np

from compare_grid_alignment import (
    FREQ_MIN,
    FREQ_MAX,
    STEP,
    get_aligned_indices,
    run_search
)

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

grid = np.arange(FREQ_MIN, FREQ_MAX + STEP, STEP)

floor_tests = []
floor_results = []

for _, row in df.iterrows():

    true_fmax = float(row["fmax_true_mhz"])

    aligned = get_aligned_indices(
        float(row["window_low_mhz"]),
        float(row["window_high_mhz"]),
        "Floor",
        grid
    )

    if aligned is None:
        best = None
        tests = 0
    else:
        low_idx, high_idx = aligned

        best, tests = run_search(
            grid,
            true_fmax,
            low_idx,
            high_idx
        )

    floor_tests.append(tests)
    floor_results.append(best)

df["floor_tests"] = floor_tests
df["floor_result_mhz"] = floor_results

df["count_difference"] = (
    df["floor_tests"] - df["tests_windowed"]
)

matched = df[df["count_difference"] == 0]
mismatched = df[df["count_difference"] != 0]

print("=" * 65)
print("FLOOR ALIGNMENT MISMATCH INVESTIGATION")
print("=" * 65)

print("\nTotal chips:", len(df))
print("Matching chips:", len(matched))
print("Mismatching chips:", len(mismatched))

print("\nMismatch direction:")
print(
    mismatched["count_difference"]
    .value_counts()
    .sort_index()
)

print("\nAverage tests:")
print("Reference:", df["tests_windowed"].mean())
print("Floor:", df["floor_tests"].mean())

print("\nMismatch analysis by actual window position:")

df["actual_inside_window"] = (
    (df["fmax_true_mhz"] >= df["window_low_mhz"])
    & (df["fmax_true_mhz"] <= df["window_high_mhz"])
)

print(
    df.groupby("actual_inside_window")[
        ["tests_windowed", "floor_tests"]
    ].mean()
)

print("\nFirst 20 mismatching chips:")

print(
    mismatched[
        [
            "unit_id",
            "fmax_true_mhz",
            "window_low_mhz",
            "window_high_mhz",
            "tests_windowed",
            "floor_tests",
            "count_difference",
            "fmax_measured_windowed_mhz",
            "floor_result_mhz"
        ]
    ].head(20).to_string(index=False)
)

output_path = "outputs/floor_alignment_mismatches.csv"
mismatched.to_csv(output_path, index=False)

print("\nMismatch details saved to:", output_path)
