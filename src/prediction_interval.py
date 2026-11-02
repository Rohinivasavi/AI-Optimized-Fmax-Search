
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# 1. Load dataset
df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# 2. Select input features
features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

target = "fmax_true_mhz"

# 3. Separate calibration data
calibration_df = df[df["split"] == "calibration"]

X_cal = calibration_df[features]
y_cal = calibration_df[target]

# 4. Load trained model
model = joblib.load("models/fmax_random_forest.pkl")

# 5. Predict Fmax on calibration data
cal_predictions = model.predict(X_cal)

# 6. Calculate absolute prediction errors
errors = np.abs(y_cal.to_numpy() - cal_predictions)

# 7. Calculate 90% conformal error threshold
alpha = 0.10
n = len(errors)

quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
q = np.quantile(errors, min(quantile_level, 1.0), method="higher")

# 8. Display results
print("=" * 50)
print("CONFORMAL PREDICTION INTERVAL")
print("=" * 50)

print("\nCalibration records:", n)
print("Confidence level: 90%")
print(f"Calibration error threshold: {q:.4f} MHz")

# 9. Example prediction interval
sample_prediction = cal_predictions[0]

lower_bound = sample_prediction - q
upper_bound = sample_prediction + q

print("\nExample predicted Fmax:", round(sample_prediction, 2), "MHz")
print("Lower bound:", round(lower_bound, 2), "MHz")
print("Upper bound:", round(upper_bound, 2), "MHz")

print("\nPrediction interval calculation completed!")
