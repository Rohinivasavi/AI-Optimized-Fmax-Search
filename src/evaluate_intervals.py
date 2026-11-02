
import pandas as pd
import numpy as np
import joblib

# 1. Load dataset
df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# 2. Define features and target
features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

target = "fmax_true_mhz"

# 3. Separate calibration and test data
cal_df = df[df["split"] == "calibration"]
test_df = df[df["split"] == "test"]

X_cal = cal_df[features]
y_cal = cal_df[target]

X_test = test_df[features]
y_test = test_df[target]

# 4. Load trained model
model = joblib.load("models/fmax_random_forest.pkl")

# 5. Calculate calibration errors
cal_predictions = model.predict(X_cal)

errors = np.abs(y_cal.to_numpy() - cal_predictions)

# 6. Calculate conformal threshold
alpha = 0.10
n = len(errors)

quantile_level = np.ceil((n + 1) * (1 - alpha)) / n

q = np.quantile(
    errors,
    min(quantile_level, 1.0),
    method="higher"
)

# 7. Predict on test data
test_predictions = model.predict(X_test)

# 8. Calculate prediction intervals
lower_bounds = test_predictions - q
upper_bounds = test_predictions + q

# 9. Check actual Fmax coverage
covered = (
    (y_test.to_numpy() >= lower_bounds) &
    (y_test.to_numpy() <= upper_bounds)
)

coverage = np.mean(covered) * 100

print("=" * 50)
print("PREDICTION INTERVAL VALIDATION")
print("=" * 50)

print("\nTest records:", len(test_df))
print("Covered records:", np.sum(covered))
print("Not covered records:", np.sum(~covered))

print(f"\nTarget coverage: {(1-alpha)*100:.2f}%")
print(f"Actual test coverage: {coverage:.2f}%")

print(f"\nAverage interval width: {2*q:.4f} MHz")

print("\nInterval validation completed!")
