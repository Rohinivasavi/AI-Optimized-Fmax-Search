
import pandas as pd
from pathlib import Path

# 1. Load the dataset
file_path = Path("data/raw/synthetic_fmax_dataset.csv")
df = pd.read_csv(file_path)

print("=" * 50)
print("FEATURE SELECTION")
print("=" * 50)

# 2. Select input features
feature_columns = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm",
    "vmin_norm",
    "die_x",
    "die_y",
    "radius_sq"
]

# 3. Select target variable
target_column = "fmax_true_mhz"

# 4. Separate features and target
X = df[feature_columns]
y = df[target_column]

# 5. Separate the provided dataset splits
train_df = df[df["split"] == "train"].copy()
calibration_df = df[df["split"] == "calibration"].copy()
test_df = df[df["split"] == "test"].copy()

# 6. Display feature information
print("\nSelected Features:")
print(feature_columns)

print("\nNumber of input features:", len(feature_columns))

print("\nTarget Variable:", target_column)

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# 7. Display split sizes
print("\nDataset Split Information:")
print("Training records:", len(train_df))
print("Calibration records:", len(calibration_df))
print("Testing records:", len(test_df))

# 8. Check missing values in selected features
print("\nMissing values in selected features:")
print(X.isnull().sum())

# 9. Check missing target values
print("\nMissing target values:", y.isnull().sum())

print("\nFeature selection completed successfully!")
