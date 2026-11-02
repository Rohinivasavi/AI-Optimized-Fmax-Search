
import pandas as pd

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Calculate how far actual Fmax is outside the window
df["outside_direction"] = "Inside"

df.loc[
    df["fmax_true_mhz"] < df["window_low_mhz"],
    "outside_direction"
] = "Below window"

df.loc[
    df["fmax_true_mhz"] > df["window_high_mhz"],
    "outside_direction"
] = "Above window"

df["outside_distance_mhz"] = 0.0

df.loc[
    df["outside_direction"] == "Below window",
    "outside_distance_mhz"
] = (
    df["window_low_mhz"] - df["fmax_true_mhz"]
)

df.loc[
    df["outside_direction"] == "Above window",
    "outside_distance_mhz"
] = (
    df["fmax_true_mhz"] - df["window_high_mhz"]
)

print("=" * 60)
print("OUTSIDE WINDOW INVESTIGATION")
print("=" * 60)

print("\nChip counts by window position:")
print(df["outside_direction"].value_counts())

print("\nAverage test counts by window position:")
print(
    df.groupby("outside_direction")["tests_windowed"]
    .agg(["count", "mean", "min", "max"])
)

outside = df[df["outside_direction"] != "Inside"].copy()

print("\nOutside chips grouped by distance:")
outside["distance_group"] = pd.cut(
    outside["outside_distance_mhz"],
    bins=[-0.001, 10, 20, 40, 80, float("inf")],
    labels=["0-10", "10-20", "20-40", "40-80", "Above 80"]
)

print(
    outside.groupby("distance_group", observed=False)[
        "tests_windowed"
    ].agg(["count", "mean", "min", "max"])
)

print("\nU02001 and other example chips:")

examples = ["U02001", "U00014", "U00013", "U00017"]

print(
    df[df["unit_id"].isin(examples)][
        [
            "unit_id",
            "fmax_true_mhz",
            "window_low_mhz",
            "window_high_mhz",
            "outside_direction",
            "outside_distance_mhz",
            "tests_windowed",
            "fmax_measured_windowed_mhz"
        ]
    ].to_string(index=False)
)
