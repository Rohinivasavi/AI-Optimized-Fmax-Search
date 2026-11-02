
import pandas as pd
import numpy as np

from compare_grid_alignment import (
    FREQ_MIN,
    FREQ_MAX,
    STEP,
    get_aligned_indices,
    run_search
)

df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

grid = np.arange(
    FREQ_MIN,
    FREQ_MAX + STEP,
    STEP
)

missing_rows = []

for _, row in df.iterrows():

    aligned = get_aligned_indices(
        float(row["window_low_mhz"]),
        float(row["window_high_mhz"]),
        "Floor",
        grid
    )

    if aligned is None:
        missing_rows.append({
            "unit_id": row["unit_id"],
            "reason": "Invalid aligned window",
            "fmax_true_mhz": row["fmax_true_mhz"],
            "window_low_mhz": row["window_low_mhz"],
            "window_high_mhz": row["window_high_mhz"]
        })
        continue

    low_idx, high_idx = aligned

    best, tests = run_search(
        grid,
        float(row["fmax_true_mhz"]),
        low_idx,
        high_idx
    )

    if best is None:
        missing_rows.append({
            "unit_id": row["unit_id"],
            "reason": "No passing frequency found",
            "fmax_true_mhz": row["fmax_true_mhz"],
            "window_low_mhz": row["window_low_mhz"],
            "window_high_mhz": row["window_high_mhz"]
        })

missing_df = pd.DataFrame(missing_rows)

print("=" * 60)
print("FLOOR ALIGNMENT MISSING RESULT INVESTIGATION")
print("=" * 60)

print("\nTotal missing results:", len(missing_df))

if not missing_df.empty:
    print("\nReasons:")
    print(missing_df["reason"].value_counts())

    print("\nFirst 30 affected chips:")
    print(missing_df.head(30).to_string(index=False))

    missing_df.to_csv(
        "outputs/floor_missing_results.csv",
        index=False
    )

    print("\nSaved to outputs/floor_missing_results.csv")

else:
    print("No missing results found.")
