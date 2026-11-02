
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

STEP = 10

# Highest grid frequency not exceeding actual Fmax
df["grid_floor_fmax"] = (
    np.floor(df["fmax_true_mhz"] / STEP) * STEP
)

# Check whether the measured Fmax matches the expected grid floor
df["measured_matches_grid_floor"] = (
    df["fmax_measured_windowed_mhz"]
    == df["grid_floor_fmax"]
)

# Check whether the measured frequency is inside the prediction window
df["measured_inside_window"] = (
    (df["fmax_measured_windowed_mhz"] >= df["window_low_mhz"])
    & (df["fmax_measured_windowed_mhz"] <= df["window_high_mhz"])
)

# Count grid points in each reference window
df["grid_points_in_window"] = df.apply(
    lambda r: len(
        np.arange(
            np.ceil(r["window_low_mhz"] / STEP) * STEP,
            r["window_high_mhz"] + 0.001,
            STEP
        )
    ),
    axis=1
)

print("=" * 60)
print("REFERENCE COUNTING RULE INVESTIGATION")
print("=" * 60)

print("\nMeasured Fmax matches grid floor:")
print(df["measured_matches_grid_floor"].value_counts())

print("\nMeasured Fmax inside prediction window:")
print(df["measured_inside_window"].value_counts())

print("\nAverage reference tests by measured position:")
print(
    df.groupby("measured_inside_window")["tests_windowed"]
    .agg(["count", "mean", "min", "max"])
)

print("\nAverage tests by number of grid points in window:")
print(
    df.groupby("grid_points_in_window")["tests_windowed"]
    .agg(["count", "mean", "min", "max"])
)

print("\nExample chips:")
examples = ["U02001", "U00013", "U00014", "U00017"]

print(
    df[df["unit_id"].isin(examples)][
        [
            "unit_id",
            "fmax_true_mhz",
            "window_low_mhz",
            "window_high_mhz",
            "grid_floor_fmax",
            "fmax_measured_windowed_mhz",
            "measured_inside_window",
            "grid_points_in_window",
            "tests_windowed"
        ]
    ].to_string(index=False)
)
