
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

test_df = df[df["split"] == "test"].copy()
cal_df = df[df["split"] == "calibration"].copy()

# ============================================================
# 2. DEFINE FEATURES
# ============================================================

features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

# ============================================================
# 3. COMMON SEARCH SETTINGS
# ============================================================

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

NOISE_STD = 4
RANDOM_SEED = 42
ALPHA = 0.10

# ============================================================
# 4. LOAD EXISTING MODEL
# ============================================================

model = joblib.load("models/fmax_random_forest.pkl")

# ============================================================
# 5. CALCULATE CONFORMAL THRESHOLD
# ============================================================

cal_predictions = model.predict(cal_df[features])
cal_actual = cal_df["fmax_true_mhz"].to_numpy()

cal_errors = np.abs(cal_actual - cal_predictions)

n = len(cal_errors)
quantile_level = np.ceil((n + 1) * (1 - ALPHA)) / n

q = np.quantile(
    cal_errors,
    min(quantile_level, 1.0),
    method="higher"
)

print("Conformal threshold:", round(q, 2), "MHz")

# ============================================================
# 6. LINEAR SEARCH
# ============================================================

def linear_search(observed_fmax):

    test_count = 0
    last_pass = FREQ_MIN

    frequency = FREQ_MIN

    while frequency <= FREQ_MAX:

        test_count += 1

        if frequency <= observed_fmax:
            last_pass = frequency
            frequency += STEP
        else:
            break

    return last_pass, test_count


# ============================================================
# 7. BINARY SEARCH
# ============================================================

def binary_search(observed_fmax):

    low = FREQ_MIN
    high = FREQ_MAX
    test_count = 0

    while high - low > STEP:

        mid = ((low + high) // 2 // STEP) * STEP

        test_count += 1

        if mid <= observed_fmax:
            low = mid
        else:
            high = mid

    return low, test_count


# ============================================================
# 8. ML-WINDOWED SEARCH
# ============================================================

def ml_windowed_search(
    observed_fmax,
    prediction,
    threshold
):

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

    # Find passing lower boundary
    while not measure(low):

        if low <= FREQ_MIN:
            return FREQ_MIN, test_count

        low = max(FREQ_MIN, low - STEP)

    # Find failing upper boundary
    while measure(high):

        if high >= FREQ_MAX:
            return FREQ_MAX, test_count

        high = min(FREQ_MAX, high + STEP)

    # Binary search within verified boundaries
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
# 9. RUN ALL FOUR ALGORITHMS
# ============================================================

predictions = model.predict(test_df[features])

rng = np.random.default_rng(RANDOM_SEED)

noise_values = rng.normal(
    loc=0,
    scale=NOISE_STD,
    size=len(test_df)
)

results = []

for i, ((_, row), prediction) in enumerate(
    zip(test_df.iterrows(), predictions)
):

    true_fmax = float(row["fmax_true_mhz"])
    noise = float(noise_values[i])

    # 1. Linear search: deterministic
    linear_fmax, linear_tests = linear_search(true_fmax)

    # 2. Binary search: deterministic
    binary_fmax, binary_tests = binary_search(true_fmax)

    # 3. ML-windowed search: deterministic
    ml_fmax, ml_tests = ml_windowed_search(
        true_fmax,
        prediction,
        q
    )

    # 4. ML-windowed search: noisy measurement
    observed_fmax = true_fmax + noise

    noisy_fmax, noisy_tests = ml_windowed_search(
        observed_fmax,
        prediction,
        q
    )

    results.append({

        "unit_id": row["unit_id"],

        "true_fmax_mhz": true_fmax,

        "predicted_fmax_mhz": prediction,

        "noise_mhz": noise,

        "linear_fmax_mhz": linear_fmax,
        "linear_tests": linear_tests,

        "binary_fmax_mhz": binary_fmax,
        "binary_tests": binary_tests,

        "ml_fmax_mhz": ml_fmax,
        "ml_tests": ml_tests,

        "noisy_fmax_mhz": noisy_fmax,
        "noisy_tests": noisy_tests

    })


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

algorithms = {
    "Linear Search": ("linear_fmax_mhz", "linear_tests"),
    "Binary Search": ("binary_fmax_mhz", "binary_tests"),
    "ML-Windowed Search": ("ml_fmax_mhz", "ml_tests"),
    "Noisy ML-Windowed Search": ("noisy_fmax_mhz", "noisy_tests")
}

# ============================================================
# 11. CALCULATE FINAL PERFORMANCE
# ============================================================

comparison_rows = []

for name, (fmax_column, tests_column) in algorithms.items():

    mae = (
        results_df[fmax_column]
        - results_df["true_fmax_mhz"]
    ).abs().mean()

    average_tests = results_df[tests_column].mean()

    maximum_tests = results_df[tests_column].max()

    comparison_rows.append({

        "Algorithm": name,

        "Average Tests": round(average_tests, 4),

        "Maximum Tests": int(maximum_tests),

        "Fmax MAE (MHz)": round(mae, 4)

    })

comparison = pd.DataFrame(comparison_rows)

# ============================================================
# 12. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL FMAX SEARCH ALGORITHM COMPARISON")
print("=" * 75)

print("\nCommon settings:")
print("Frequency range:", FREQ_MIN, "-", FREQ_MAX, "MHz")
print("Frequency step:", STEP, "MHz")
print("Total test chips:", len(results_df))
print("Conformal threshold:", round(q, 2), "MHz")
print("Noisy measurement standard deviation:", NOISE_STD, "MHz")

print("\nFINAL PERFORMANCE TABLE")
print("-" * 75)

print(comparison.to_string(index=False))

# ============================================================
# 13. CALCULATE TEST REDUCTION
# ============================================================

linear_avg = results_df["linear_tests"].mean()
binary_avg = results_df["binary_tests"].mean()
ml_avg = results_df["ml_tests"].mean()
noisy_avg = results_df["noisy_tests"].mean()

print("\nTEST REDUCTION")
print("-" * 75)

for name, average in [
    ("Binary vs Linear", binary_avg),
    ("ML vs Linear", ml_avg),
    ("Noisy ML vs Linear", noisy_avg),
    ("ML vs Binary", ml_avg),
    ("Noisy ML vs Binary", noisy_avg)
]:

    baseline = (
        linear_avg if "Linear" in name
        else binary_avg
    )

    reduction = ((baseline - average) / baseline) * 100

    print(f"{name}: {reduction:.2f}%")

# ============================================================
# 14. SAVE RESULTS
# ============================================================

output_dir = Path("outputs")
graph_dir = output_dir / "graphs"

graph_dir.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output_dir / "final_algorithm_comparison.csv",
    index=False
)

results_df.to_csv(
    output_dir / "final_algorithm_detailed_results.csv",
    index=False
)

# ============================================================
# 15. CREATE COMPARISON GRAPH
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Algorithm"],
    comparison["Average Tests"]
)

plt.title("Average Frequency Tests per Chip")
plt.xlabel("Search Algorithm")
plt.ylabel("Average Number of Tests")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    graph_dir / "final_algorithm_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved files:")
print("outputs/final_algorithm_comparison.csv")
print("outputs/final_algorithm_detailed_results.csv")
print("outputs/graphs/final_algorithm_comparison.png")

print("\nFinal algorithm comparison completed!")
