
import pandas as pd

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Check whether actual Fmax falls inside the reference window
df["actual_inside_window"] = (
    (df["fmax_true_mhz"] >= df["window_low_mhz"]) &
    (df["fmax_true_mhz"] <= df["window_high_mhz"])
)

# Distance of actual Fmax from the nearest window boundary
df["distance_from_window_mhz"] = df.apply(
    lambda r: min(
        r["fmax_true_mhz"] - r["window_low_mhz"],
        r["window_high_mhz"] - r["fmax_true_mhz"]
    ),
    axis=1
)

print("=" * 60)
print("REFERENCE WINDOW BEHAVIOR ANALYSIS")
print("=" * 60)

print("\nActual Fmax inside reference window:")
print(df["actual_inside_window"].value_counts())

print("\nAverage reference tests by window coverage:")
print(
    df.groupby("actual_inside_window")["tests_windowed"]
    .agg(["count", "mean", "min", "max"])
)

print("\nAverage tests by actual Fmax distance from window:")
df["distance_group"] = pd.cut(
    df["distance_from_window_mhz"],
    bins=[-float("inf"), 0, 10, 20, 30, 40, float("inf")],
    labels=["Outside", "0-10", "10-20", "20-30", "30-40", "Above 40"]
)

print(
    df.groupby("distance_group", observed=False)["tests_windowed"]
    .agg(["count", "mean", "min", "max"])
)

print("\nFirst 20 chips:")
print(
    df[
        [
            "unit_id",
            "fmax_true_mhz",
            "window_low_mhz",
            "window_high_mhz",
            "actual_inside_window",
            "tests_windowed",
            "fmax_measured_windowed_mhz"
        ]
    ].head(20).to_string(index=False)
)