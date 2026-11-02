
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Select test records
test_df = df[df["split"] == "test"].copy()

# Select calibration records
cal_df = df[df["split"] == "calibration"].copy()


# ============================================================
# 2. DEFINE INPUT FEATURES
# ============================================================

features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]


# ============================================================
# 3. LOAD EXISTING RANDOM FOREST MODEL
# ============================================================

model = joblib.load("models/fmax_random_forest.pkl")


# ============================================================
# 4. CALCULATE CONFORMAL PREDICTION THRESHOLD
# ============================================================

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


# ============================================================
# 5. SEARCH SETTINGS
# ============================================================

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

# Measurement noise standard deviation from the document
NOISE_STD = 4

# Fixed seed for reproducible experiments
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 6. GENERATE NOISY MEASUREMENT THRESHOLDS
# ============================================================

# One noise sample per chip.
# This creates a monotonic pass/fail threshold for each chip.

noise_values = rng.normal(
    loc=0,
    scale=NOISE_STD,
    size=len(test_df)
)

test_df["measurement_noise_mhz"] = noise_values

test_df["observed_fmax_mhz"] = (
    test_df["fmax_true_mhz"].to_numpy()
    + noise_values
)


# ============================================================
# 7. MEASUREMENT RESPONSE
# ============================================================

def chip_passes(frequency, observed_fmax):
    """
    Simulate a measured PASS/FAIL response.

    The search uses the noisy observed threshold,
    not the actual ground-truth Fmax.
    """

    return frequency <= observed_fmax


# ============================================================
# 8. ML-WINDOWED SEARCH WITH NOISY MEASUREMENTS
# ============================================================

def ml_windowed_search_noisy(
    observed_fmax,
    prediction,
    threshold
):
    """
    Search for the highest passing frequency using
    an ML prediction interval and noisy measurements.

    Each frequency is tested at most once.
    """

    test_count = 0
    measurement_cache = {}

    def measure(frequency):

        nonlocal test_count

        if frequency not in measurement_cache:

            result = chip_passes(
                frequency,
                observed_fmax
            )

            measurement_cache[frequency] = result
            test_count += 1

        return measurement_cache[frequency]

    # Calculate prediction interval
    lower = prediction - threshold
    upper = prediction + threshold

    # Align boundaries to the frequency grid
    low = int(np.floor(lower / STEP) * STEP)
    high = int(np.ceil(upper / STEP) * STEP)

    low = max(FREQ_MIN, min(low, FREQ_MAX))
    high = max(FREQ_MIN, min(high, FREQ_MAX))

    # Ensure valid frequency order
    if low > high:
        low, high = high, low

    # --------------------------------------------------------
    # Phase 1: Find a passing lower boundary
    # --------------------------------------------------------

    while not measure(low):

        if low <= FREQ_MIN:
            return FREQ_MIN, test_count

        low = max(FREQ_MIN, low - STEP)

    # --------------------------------------------------------
    # Phase 2: Find a failing upper boundary
    # --------------------------------------------------------

    while measure(high):

        if high >= FREQ_MAX:
            return FREQ_MAX, test_count

        high = min(FREQ_MAX, high + STEP)

    # --------------------------------------------------------
    # Phase 3: Binary search inside the verified bracket
    # --------------------------------------------------------

    while high - low > STEP:

        mid = ((low + high) // 2 // STEP) * STEP

        if mid <= low:
            break

        if measure(mid):
            low = mid
        else:
            high = mid

    return low, test_count


# ============================================================
# 9. PREDICT FMAX FOR TEST CHIPS
# ============================================================

test_predictions = model.predict(test_df[features])

results = []

for (_, row), prediction in zip(
    test_df.iterrows(),
    test_predictions
):

    true_fmax = float(row["fmax_true_mhz"])

    observed_fmax = float(row["observed_fmax_mhz"])

    noise = float(row["measurement_noise_mhz"])

    measured_fmax, tests = ml_windowed_search_noisy(
        observed_fmax,
        prediction,
        q
    )

    results.append({

        "unit_id": row["unit_id"],

        "true_fmax_mhz": true_fmax,

        "predicted_fmax_mhz": prediction,

        "measurement_noise_mhz": noise,

        "observed_fmax_mhz": observed_fmax,

        "measured_fmax_mhz": measured_fmax,

        "tests_required": tests

    })


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

# Calculate accuracy against actual Fmax.
# Actual Fmax is used only for evaluation, not searching.

results_df["absolute_error_mhz"] = (
    results_df["measured_fmax_mhz"]
    - results_df["true_fmax_mhz"]
).abs()


# ============================================================
# 11. DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("ML-WINDOWED SEARCH WITH NOISY MEASUREMENTS")
print("=" * 60)

print("\nTotal chips tested:", len(results_df))

print(
    "Average tests per chip:",
    round(results_df["tests_required"].mean(), 2)
)

print(
    "Maximum tests:",
    results_df["tests_required"].max()
)

print(
    "Minimum tests:",
    results_df["tests_required"].min()
)

print(
    "Mean absolute error:",
    round(results_df["absolute_error_mhz"].mean(), 4),
    "MHz"
)

print(
    "Median absolute error:",
    round(results_df["absolute_error_mhz"].median(), 4),
    "MHz"
)

print(
    "Measurement noise standard deviation:",
    NOISE_STD,
    "MHz"
)

print("\nFirst five results:")
print(results_df.head().to_string(index=False))


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_path = "outputs/ml_windowed_search_noisy_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\nNoisy search results saved to:", output_path)

print("\nML-windowed noisy search completed!")
