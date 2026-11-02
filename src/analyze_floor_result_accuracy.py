
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"
MISMATCH_PATH = "outputs/floor_alignment_mismatches.csv"

df = pd.read_csv(DATA_PATH)
mismatches = pd.read_csv(MISMATCH_PATH)

# Reconstruct the full Floor result using the saved mismatch file
# and the original reference data.
from compare_grid_alignment import (
    FREQ_MIN,
    FREQ_MAX,
    STEP,
    get_aligned_indices,
    run_search
)

grid = np.arange(FREQ_MIN, FREQ_MAX + STEP, STEP)

floor_results = []

for _, row in df.iterrows():
    aligned = get_aligned_indices(
        float(row["window_low_mhz"]),
        float(row["window_high_mhz"]),
        "Floor",
        grid
    )

    if aligned is None:
        best = None
    else:
        low_idx, high_idx = aligned
        best, _ = run_search(
            grid,
            float(row["fmax_true_mhz"]),
            low_idx,
            high_idx
        )

    floor_results.append(best)

df["floor_result_mhz"] = floor_results

# Compare Floor result with reference measured Fmax
df["measurement_difference"] = (
    df["floor_result_mhz"]
    - df["fmax_measured_windowed_mhz"]
)

df["absolute_difference"] = (
    df["measurement_difference"].abs()
)

print("=" * 65)
print("FLOOR RESULT ACCURACY ANALYSIS")
print("=" * 65)

print("\nFloor result vs reference measured Fmax:")
print(
    df["absolute_difference"]
    .value_counts()
    .sort_index()
)

print("\nExact measured Fmax matches:",
      (df["absolute_difference"] == 0).sum())

print("Different measured Fmax results:",
      (df["absolute_difference"] > 0).sum())

print("\nAverage absolute measured Fmax difference:",
      df["absolute_difference"].mean(), "MHz")

print("\nResults for the 857 count mismatches:")

merged = df.merge(
    mismatches[["unit_id"]],
    on="unit_id",
    how="inner"
)

print(
    merged[
        [
            "unit_id",
            "fmax_true_mhz",
            "fmax_measured_windowed_mhz",
            "floor_result_mhz",
            "measurement_difference",
            "tests_windowed",
        ]
    ].head(20).to_string(index=False)
)

print("\nMismatch group measured Fmax agreement:")
print(
    merged["absolute_difference"]
    .value_counts()
    .sort_index()
)

print("\nAnalysis complete. Original dataset unchanged.")
