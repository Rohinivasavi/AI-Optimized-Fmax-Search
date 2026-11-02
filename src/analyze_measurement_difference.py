
import pandas as pd

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Difference between reference measured Fmax and actual Fmax
df["measurement_difference_mhz"] = (
    df["fmax_measured_windowed_mhz"] - df["fmax_true_mhz"]
)

df["absolute_measurement_difference_mhz"] = (
    df["measurement_difference_mhz"].abs()
)

print("=" * 60)
print("MEASUREMENT DIFFERENCE ANALYSIS")
print("=" * 60)

print("\nMeasurement difference statistics:")
print(df["measurement_difference_mhz"].describe())

print("\nAbsolute measurement difference statistics:")
print(df["absolute_measurement_difference_mhz"].describe())

print("\nMeasurement difference distribution:")
print(
    df["measurement_difference_mhz"]
    .value_counts()
    .sort_index()
    .to_string()
)

# Group chips by absolute measurement difference
df["difference_group"] = pd.cut(
    df["absolute_measurement_difference_mhz"],
    bins=[-0.001, 2, 5, 10, 20, 40, float("inf")],
    labels=["0-2", "2-5", "5-10", "10-20", "20-40", "Above 40"]
)

print("\nAverage test counts by measurement difference:")
print(
    df.groupby("difference_group", observed=False)["tests_windowed"]
    .agg(["count", "mean", "min", "max"])
)

print("\nExamples with largest measurement differences:")
print(
    df.sort_values(
        "absolute_measurement_difference_mhz",
        ascending=False
    )[
        [
            "unit_id",
            "fmax_true_mhz",
            "fmax_measured_windowed_mhz",
            "measurement_difference_mhz",
            "tests_windowed"
        ]
    ].head(15).to_string(index=False)
)

print("\nSelected example chips:")
examples = ["U02001", "U00013", "U00014", "U00017"]

print(
    df[df["unit_id"].isin(examples)][
        [
            "unit_id",
            "fmax_true_mhz",
            "fmax_measured_windowed_mhz",
            "measurement_difference_mhz",
            "tests_windowed"
        ]
    ].to_string(index=False)
)
