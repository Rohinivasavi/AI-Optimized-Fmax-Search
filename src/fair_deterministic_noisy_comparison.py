
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

test_df = df[df["split"] == "test"].copy()
cal_df = df[df["split"] == "calibration"].copy()

# ============================================================
# 2. INPUT FEATURES
# ============================================================

features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

# ============================================================
# 3. LOAD EXISTING MODEL
# ============================================================

model = joblib.load("models/fmax_random_forest.pkl")

# ============================================================
# 4. CALCULATE CONFORMAL THRESHOLD
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
# 5. IDENTICAL SEARCH SETTINGS
# ============================================================

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

NOISE_STD = 4
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# Same noise values for reproducibility
noise_values = rng.normal(
    loc=0,
    scale=NOISE_STD,
    size=len(test_df)
)

# ============================================================
# 6. COMMON SEARCH FUNCTION
# ============================================================

def search_fmax(observed_fmax, prediction, threshold):

    test_count = 0
    cache = {}

    def measure(frequency):

        nonlocal test_count

        if frequency not in cache:
            cache[frequency] = frequency <= observed_fmax
            test_count += 1

        return cache[frequency]

    lower = prediction - threshold
    upper = prediction + threshold

    low = int(np.floor(lower / STEP) * STEP)
    high = int(np.ceil(upper / STEP) * STEP)

    low = max(FREQ_MIN, min(low, FREQ_MAX))
    high = max(FREQ_MIN, min(high, FREQ_MAX))

    if low > high:
        low, high = high, low

    # Find a passing lower boundary
    while not measure(low):

        if low <= FREQ_MIN:
            return FREQ_MIN, test_count

        low = max(FREQ_MIN, low - STEP)

    # Find a failing upper boundary
    while measure(high):

        if high >= FREQ_MAX:
            return FREQ_MAX, test_count

        high = min(FREQ_MAX, high + STEP)

    # Binary search
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
# 7. RUN BOTH EXPERIMENTS
# ============================================================

predictions = model.predict(test_df[features])

results = []

for i, ((_, row), prediction) in enumerate(
    zip(test_df.iterrows(), predictions)
):

    true_fmax = float(row["fmax_true_mhz"])
    noise = float(noise_values[i])

    # Deterministic: zero measurement noise
    deterministic_fmax, deterministic_tests = search_fmax(
        true_fmax,
        prediction,
        q
    )

    # Noisy: same chip with 4 MHz standard deviation noise
    observed_fmax = true_fmax + noise

    noisy_fmax, noisy_tests = search_fmax(
        observed_fmax,
        prediction,
        q
    )

    results.append({
        "unit_id": row["unit_id"],
        "true_fmax_mhz": true_fmax,
        "predicted_fmax_mhz": prediction,
        "noise_mhz": noise,
        "observed_fmax_mhz": observed_fmax,
        "deterministic_fmax_mhz": deterministic_fmax,
        "noisy_fmax_mhz": noisy_fmax,
        "deterministic_tests": deterministic_tests,
        "noisy_tests": noisy_tests,
        "deterministic_error_mhz": abs(
            deterministic_fmax - true_fmax
        ),
        "noisy_error_mhz": abs(
            noisy_fmax - true_fmax
        )
    })

# ============================================================
# 8. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

# ============================================================
# 9. DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("FAIR DETERMINISTIC VS NOISY COMPARISON")
print("=" * 65)

print("\nSEARCH SETTINGS")
print("Frequency range:", FREQ_MIN, "-", FREQ_MAX, "MHz")
print("Frequency step:", STEP, "MHz")
print("Noise standard deviation:", NOISE_STD, "MHz")
print("Total chips:", len(results_df))

print("\n1. TEST EFFICIENCY")
print("-" * 50)

print(
    "Deterministic average tests:",
    round(results_df["deterministic_tests"].mean(), 4)
)

print(
    "Noisy average tests:",
    round(results_df["noisy_tests"].mean(), 4)
)

print(
    "Deterministic maximum tests:",
    results_df["deterministic_tests"].max()
)

print(
    "Noisy maximum tests:",
    results_df["noisy_tests"].max()
)

print("\n2. FMAX ACCURACY")
print("-" * 50)

print(
    "Deterministic MAE:",
    round(results_df["deterministic_error_mhz"].mean(), 4),
    "MHz"
)

print(
    "Noisy MAE:",
    round(results_df["noisy_error_mhz"].mean(), 4),
    "MHz"
)

print(
    "Deterministic median error:",
    round(results_df["deterministic_error_mhz"].median(), 4),
    "MHz"
)

print(
    "Noisy median error:",
    round(results_df["noisy_error_mhz"].median(), 4),
    "MHz"
)

print("\n3. CHIP-BY-CHIP ACCURACY")
print("-" * 50)

print(
    "Deterministic result closer:",
    (
        results_df["deterministic_error_mhz"]
        < results_df["noisy_error_mhz"]
    ).sum()
)

print(
    "Noisy result closer:",
    (
        results_df["noisy_error_mhz"]
        < results_df["deterministic_error_mhz"]
    ).sum()
)

print(
    "Equal accuracy:",
    (
        results_df["noisy_error_mhz"]
        == results_df["deterministic_error_mhz"]
    ).sum()
)

print("\n4. TEST COUNT DIFFERENCE")
print("-" * 50)

difference = (
    results_df["noisy_tests"]
    - results_df["deterministic_tests"]
)

print("Average additional tests:", round(difference.mean(), 4))
print("More tests:", (difference > 0).sum())
print("Fewer tests:", (difference < 0).sum())
print("Equal tests:", (difference == 0).sum())

# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_path = (
    "outputs/fair_deterministic_noisy_comparison.csv"
)

results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("\nFair comparison completed!")
