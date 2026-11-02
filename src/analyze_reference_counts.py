
import pandas as pd

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Compare reference test counts
df["test_count_difference"] = (
    df["tests_windowed"] - df["tests_binary"]
)

print("=" * 60)
print("REFERENCE TEST COUNT ANALYSIS")
print("=" * 60)

print("\nTotal chips:", len(df))

print("\nBinary search test-count distribution:")
print(df["tests_binary"].value_counts().sort_index())

print("\nWindowed search test-count distribution:")
print(df["tests_windowed"].value_counts().sort_index())

print("\nDifference distribution (windowed - binary):")
print(df["test_count_difference"].value_counts().sort_index())

print("\nAverage binary tests:", df["tests_binary"].mean())
print("Average windowed tests:", df["tests_windowed"].mean())

print("\nFirst 15 chips:")
print(
    df[
        [
            "unit_id",
            "fmax_true_mhz",
            "fmax_pred_mhz",
            "window_low_mhz",
            "window_high_mhz",
            "tests_binary",
            "tests_windowed",
            "fmax_measured_windowed_mhz",
        ]
    ].head(15).to_string(index=False)
)
