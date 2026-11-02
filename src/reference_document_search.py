
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

test_df = df[df["split"] == "test"].copy()
calibration_df = df[df["split"] == "calibration"].copy()

FEATURES = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"
]

# Load existing trained Random Forest model
model = joblib.load("models/fmax_random_forest.pkl")


# ============================================================
# 2. CONFORMAL PREDICTION
# ============================================================

ALPHA = 0.10

X_cal = calibration_df[FEATURES]
y_cal = calibration_df["fmax_true_mhz"]

cal_predictions = model.predict(X_cal)

cal_errors = np.abs(y_cal - cal_predictions)

q = np.quantile(
    cal_errors,
    1 - ALPHA,
    method="higher"
)

print("\nCONFORMAL PREDICTION")
print("--------------------")
print(f"Alpha: {ALPHA}")
print(f"Conformal threshold: {q:.4f} MHz")


# ============================================================
# 3. FREQUENCY SEARCH GRID
# ============================================================

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

frequency_grid = np.arange(
    FREQ_MIN,
    FREQ_MAX + STEP,
    STEP
)


# ============================================================
# 4. SIMULATED PASS/FAIL TEST
# ============================================================

def chip_passes(frequency, true_fmax):
    return frequency <= true_fmax


# ============================================================
# 5. BINARY SEARCH WITHIN GRID INDICES
# ============================================================

def binary_search_grid(true_fmax, start_idx, end_idx):

    low = start_idx
    high = end_idx

    best = None
    test_count = 0

    while low <= high:

        mid = (low + high) // 2
        frequency = frequency_grid[mid]

        test_count += 1

        if chip_passes(frequency, true_fmax):

            best = mid
            low = mid + 1

        else:

            high = mid - 1

    return best, test_count


# ============================================================
# 6. GENERATE MODEL PREDICTION WINDOWS
# ============================================================

X_test = test_df[FEATURES]

test_df["model_pred_mhz"] = model.predict(X_test)

test_df["model_window_low_mhz"] = (
    test_df["model_pred_mhz"] - q
)

test_df["model_window_high_mhz"] = (
    test_df["model_pred_mhz"] + q
)

print("\nPREDICTION WINDOWS")
print("------------------")

print(
    test_df[
        [
            "unit_id",
            "model_pred_mhz",
            "model_window_low_mhz",
            "model_window_high_mhz"
        ]
    ].head()
)


# ============================================================
# 7. CONVERT PREDICTION WINDOWS TO GRID INDICES
# ============================================================

def get_prediction_window(lower_frequency, upper_frequency):

    lower_idx = np.searchsorted(
        frequency_grid,
        lower_frequency,
        side="left"
    )

    upper_idx = np.searchsorted(
        frequency_grid,
        upper_frequency,
        side="right"
    ) - 1

    return lower_idx, upper_idx


# ============================================================
# 8. MAIN WINDOWED SEARCH ALGORITHM
# ============================================================

def reference_windowed_search(
    true_fmax,
    lower_frequency,
    upper_frequency
):

    test_count = 0

    # Convert prediction window into grid indices
    lo, hi = get_prediction_window(
        lower_frequency,
        upper_frequency
    )

    # Handle windows completely outside the frequency grid
    if hi < 0 or lo >= len(frequency_grid):
        return None, test_count

    # Restrict indices to valid grid positions
    lo = max(0, lo)
    hi = min(len(frequency_grid) - 1, hi)

    if lo > hi:
        return None, test_count

    # Search inside the prediction window
    best, count = binary_search_grid(
        true_fmax,
        lo,
        hi
    )

    test_count += count

    # --------------------------------------------------------
    # CASE 1: No passing frequency inside the window
    # Search below the lower boundary
    # --------------------------------------------------------

    if best is None:

        if lo > 0:

            best, count = binary_search_grid(
                true_fmax,
                0,
                lo - 1
            )

            test_count += count

        measured_fmax = (
            frequency_grid[best]
            if best is not None
            else None
        )

        return measured_fmax, test_count

    # --------------------------------------------------------
    # CASE 2: Upper boundary passes
    # Expand the upper boundary exponentially
    # --------------------------------------------------------

    if best == hi:

        last_pass_idx = hi
        stride = 1
        first_fail_idx = None

        while last_pass_idx + stride < len(frequency_grid):

            probe_idx = last_pass_idx + stride
            probe_frequency = frequency_grid[probe_idx]

            test_count += 1

            if chip_passes(probe_frequency, true_fmax):

                last_pass_idx = probe_idx
                stride *= 2

            else:

                first_fail_idx = probe_idx
                break

        # Search the remaining interval after expansion
        if first_fail_idx is not None:

            best, count = binary_search_grid(
                true_fmax,
                last_pass_idx,
                first_fail_idx - 1
            )

            test_count += count

        else:

            # Reached the maximum frequency without failure
            best = last_pass_idx

    # Convert best grid position into measured frequency
    measured_fmax = (
        frequency_grid[best]
        if best is not None
        else None
    )

    return measured_fmax, test_count


# ============================================================
# 9. IMPLEMENTATION SUMMARY
# ============================================================

print("\nIMPLEMENTATION SETUP COMPLETED")
print("--------------------------------")

print(f"Test records: {len(test_df)}")
print(f"Calibration records: {len(calibration_df)}")
print(f"Frequency grid points: {len(frequency_grid)}")

print(f"Frequency range: {FREQ_MIN}-{FREQ_MAX} MHz")
print(f"Frequency step: {STEP} MHz")

print("\nMain windowed search function is ready.")

# ============================================================
# 10. RUN SEARCH FOR ALL TEST CHIPS
# ============================================================

results = []

for _, row in test_df.iterrows():

    true_fmax = row["fmax_true_mhz"]

    lower_frequency = row["model_window_low_mhz"]
    upper_frequency = row["model_window_high_mhz"]

    measured_fmax, test_count = reference_windowed_search(
        true_fmax,
        lower_frequency,
        upper_frequency
    )

    results.append({
        "unit_id": row["unit_id"],
        "true_fmax_mhz": true_fmax,
        "model_pred_mhz": row["model_pred_mhz"],
        "window_low_mhz": lower_frequency,
        "window_high_mhz": upper_frequency,
        "measured_fmax_mhz": measured_fmax,
        "tests_required": test_count
    })


# Convert results into a DataFrame
results_df = pd.DataFrame(results)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

output_path = "outputs/reference_document_search_results.csv"

results_df.to_csv(output_path, index=False)


# ============================================================
# 12. DISPLAY PERFORMANCE
# ============================================================

print("\nREFERENCE DOCUMENT SEARCH RESULTS")
print("---------------------------------")

print(f"Total test chips: {len(results_df)}")

print(
    f"Average tests required: "
    f"{results_df['tests_required'].mean():.4f}"
)

print(
    f"Maximum tests required: "
    f"{results_df['tests_required'].max()}"
)

print(
    f"Average absolute Fmax error: "
    f"{np.abs(results_df['true_fmax_mhz'] - results_df['measured_fmax_mhz']).mean():.4f} MHz"
)

print("\nFirst five results:")
print(results_df.head().to_string(index=False))

print(f"\nResults saved to: {output_path}")
