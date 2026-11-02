
import pandas as pd
import numpy as np
import joblib

# 1. Load dataset
df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# 2. Select test records
test_df = df[df["split"] == "test"].copy()

# 3. Define input features
features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

# 4. Load trained model
model = joblib.load("models/fmax_random_forest.pkl")

# 5. Calculate conformal threshold using calibration data
cal_df = df[df["split"] == "calibration"]

cal_predictions = model.predict(cal_df[features])
cal_actual = cal_df["fmax_true_mhz"].to_numpy()

errors = np.abs(cal_actual - cal_predictions)

alpha = 0.10
n = len(errors)

quantile_level = np.ceil((n + 1) * (1 - alpha)) / n

q = np.quantile(
    errors,
    min(quantile_level, 1.0),
    method="higher"
)

print("Conformal threshold:", round(q, 2), "MHz")

# 6. Define search settings
FREQ_MIN = 1500
FREQ_MAX = 2200
STEP = 10


# 7. Simulate frequency testing
def chip_passes(frequency, true_fmax):
    return frequency <= true_fmax


# 8. ML-windowed binary search
def ml_windowed_search(true_fmax, prediction, threshold):

    test_count = 0

    # Calculate predicted interval
    lower = prediction - threshold
    upper = prediction + threshold

    # Align interval to 10 MHz frequency grid
    low = max(FREQ_MIN, int(np.floor(lower / STEP) * STEP))
    high = min(FREQ_MAX, int(np.ceil(upper / STEP) * STEP))

    # Test lower boundary
    test_count += 1

    while not chip_passes(low, true_fmax):

        if low <= FREQ_MIN:
            return low, test_count

        low = max(FREQ_MIN, low - STEP)
        test_count += 1

    # Test upper boundary
    test_count += 1

    while chip_passes(high, true_fmax):

        if high >= FREQ_MAX:
            return high, test_count

        high = min(FREQ_MAX, high + STEP)
        test_count += 1

    # Binary search inside the verified bracket
    while high - low > STEP:

        mid = ((low + high) // 2 // STEP) * STEP

        if mid <= low:
            break

        test_count += 1

        if chip_passes(mid, true_fmax):
            low = mid
        else:
            high = mid

    return low, test_count


# 9. Predict Fmax for test chips
test_predictions = model.predict(test_df[features])

results = []

for (_, row), prediction in zip(
    test_df.iterrows(), test_predictions
):

    true_fmax = row["fmax_true_mhz"]

    measured_fmax, tests = ml_windowed_search(
        true_fmax,
        prediction,
        q
    )

    results.append({
        "unit_id": row["unit_id"],
        "true_fmax_mhz": true_fmax,
        "predicted_fmax_mhz": prediction,
        "measured_fmax_mhz": measured_fmax,
        "tests_required": tests
    })


# 10. Convert results into DataFrame
results_df = pd.DataFrame(results)

# 11. Display performance
print("\n" + "=" * 50)
print("ML-WINDOWED BINARY SEARCH RESULTS")
print("=" * 50)

print("\nTotal chips tested:", len(results_df))

print(
    "Average tests per chip:",
    round(results_df["tests_required"].mean(), 2)
)

print(
    "Maximum tests:",
    results_df["tests_required"].max()
)

print("\nFirst five results:")
print(results_df.head())

# 12. Save results
results_df.to_csv(
    "outputs/ml_windowed_search_results.csv",
    index=False
)

print("\nML-windowed search completed!")

