
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# Select original test records
test_df = df[df["split"] == "test"].copy()

print("\nORIGINAL DATASET")
print("----------------")
print(f"Total dataset records: {len(df)}")
print(f"Test records: {len(test_df)}")


# ============================================================
# 2. FREQUENCY SEARCH GRID
# ============================================================

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

frequency_grid = np.arange(
    FREQ_MIN,
    FREQ_MAX + STEP,
    STEP
)

print(f"Frequency grid points: {len(frequency_grid)}")


# ============================================================
# 3. SIMULATED PASS/FAIL TEST
# ============================================================

def chip_passes(frequency, true_fmax):
    return frequency <= true_fmax


# ============================================================
# 4. BINARY SEARCH
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
# 5. CONVERT ORIGINAL WINDOWS TO GRID INDICES
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
# 6. WINDOWED SEARCH USING ORIGINAL DATASET WINDOWS
# ============================================================

def reference_window_search(
    true_fmax,
    lower_frequency,
    upper_frequency
):

    test_count = 0

    lo, hi = get_prediction_window(
        lower_frequency,
        upper_frequency
    )

    # Handle windows outside the frequency grid
    if hi < 0 or lo >= len(frequency_grid):
        return None, test_count

    lo = max(0, lo)
    hi = min(len(frequency_grid) - 1, hi)

    if lo > hi:
        return None, test_count

    # Search inside the original prediction window
    best, count = binary_search_grid(
        true_fmax,
        lo,
        hi
    )

    test_count += count

    # CASE 1: No passing frequency inside the window
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

    # CASE 2: Upper boundary passes
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

        if first_fail_idx is not None:

            best, count = binary_search_grid(
                true_fmax,
                last_pass_idx,
                first_fail_idx - 1
            )

            test_count += count

        else:

            best = last_pass_idx

    measured_fmax = (
        frequency_grid[best]
        if best is not None
        else None
    )

    return measured_fmax, test_count


# ============================================================
# 7. RUN DIAGNOSTIC FOR ALL TEST CHIPS
# ============================================================

results = []

for _, row in test_df.iterrows():

    true_fmax = row["fmax_true_mhz"]

    # Use ORIGINAL dataset prediction windows
    lower_frequency = row["window_low_mhz"]
    upper_frequency = row["window_high_mhz"]

    measured_fmax, test_count = reference_window_search(
        true_fmax,
        lower_frequency,
        upper_frequency
    )

    results.append({

        "unit_id": row["unit_id"],

        "true_fmax_mhz": true_fmax,

        "original_pred_mhz": row["fmax_pred_mhz"],

        "original_window_low_mhz": lower_frequency,

        "original_window_high_mhz": upper_frequency,

        "original_tests_windowed": row["tests_windowed"],

        "diagnostic_measured_fmax_mhz": measured_fmax,

        "diagnostic_tests_required": test_count,

        "original_measured_fmax_mhz":
            row["fmax_measured_windowed_mhz"]

    })


# ============================================================
# 8. CREATE COMPARISON DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df["test_count_match"] = (
    results_df["original_tests_windowed"]
    == results_df["diagnostic_tests_required"]
)

results_df["measured_fmax_match"] = (
    results_df["original_measured_fmax_mhz"]
    == results_df["diagnostic_measured_fmax_mhz"]
)

results_df["test_count_difference"] = (
    results_df["diagnostic_tests_required"]
    - results_df["original_tests_windowed"]
)


# ============================================================
# 9. SAVE DIAGNOSTIC RESULTS
# ============================================================

output_path = "outputs/diagnose_reference_mismatch.csv"

results_df.to_csv(output_path, index=False)


# ============================================================
# 10. PERFORMANCE SUMMARY
# ============================================================

print("\nREFERENCE WINDOW DIAGNOSTIC")
print("---------------------------")

print(f"Total test chips: {len(results_df)}")

print(
    "Original average tests:",
    f"{results_df['original_tests_windowed'].mean():.4f}"
)

print(
    "Diagnostic average tests:",
    f"{results_df['diagnostic_tests_required'].mean():.4f}"
)

print(
    "Original maximum tests:",
    results_df["original_tests_windowed"].max()
)

print(
    "Diagnostic maximum tests:",
    results_df["diagnostic_tests_required"].max()
)

print(
    "Matching test counts:",
    results_df["test_count_match"].sum()
)

print(
    "Mismatched test counts:",
    (~results_df["test_count_match"]).sum()
)

print(
    "Matching measured Fmax:",
    results_df["measured_fmax_match"].sum()
)

print(
    "Average absolute diagnostic Fmax error:",
    f"{np.abs(results_df['true_fmax_mhz'] - results_df['diagnostic_measured_fmax_mhz']).mean():.4f} MHz"
)


# ============================================================
# 11. DISPLAY MISMATCHED CHIPS
# ============================================================

mismatches = results_df[
    ~results_df["test_count_match"]
]

print("\nFIRST 10 MISMATCHED CHIPS")
print("-------------------------")

print(
    mismatches[
        [
            "unit_id",
            "original_tests_windowed",
            "diagnostic_tests_required",
            "original_window_low_mhz",
            "original_window_high_mhz",
            "test_count_difference"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\nDIAGNOSTIC COMPLETED")
print("--------------------")
print(f"Results saved to: {output_path}")

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# Select original test records
test_df = df[df["split"] == "test"].copy()

print("\nORIGINAL DATASET")
print("----------------")
print(f"Total dataset records: {len(df)}")
print(f"Test records: {len(test_df)}")


# ============================================================
# 2. FREQUENCY SEARCH GRID
# ============================================================

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

frequency_grid = np.arange(
    FREQ_MIN,
    FREQ_MAX + STEP,
    STEP
)

print(f"Frequency grid points: {len(frequency_grid)}")


# ============================================================
# 3. SIMULATED PASS/FAIL TEST
# ============================================================

def chip_passes(frequency, true_fmax):
    return frequency <= true_fmax


# ============================================================
# 4. BINARY SEARCH
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
# 5. CONVERT ORIGINAL WINDOWS TO GRID INDICES
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
# 6. WINDOWED SEARCH USING ORIGINAL DATASET WINDOWS
# ============================================================

def reference_window_search(
    true_fmax,
    lower_frequency,
    upper_frequency
):

    test_count = 0

    lo, hi = get_prediction_window(
        lower_frequency,
        upper_frequency
    )

    # Handle windows outside the frequency grid
    if hi < 0 or lo >= len(frequency_grid):
        return None, test_count

    lo = max(0, lo)
    hi = min(len(frequency_grid) - 1, hi)

    if lo > hi:
        return None, test_count

    # Search inside the original prediction window
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

            # Count this new boundary test
            test_count += 1

            if chip_passes(probe_frequency, true_fmax):

                last_pass_idx = probe_idx
                stride *= 2

            else:

                first_fail_idx = probe_idx
                break

        # ----------------------------------------------------
        # Search only untested positions.
        # Do not count last_pass_idx again.
        # ----------------------------------------------------

        if first_fail_idx is not None:

            best, count = binary_search_grid(
                true_fmax,
                last_pass_idx + 1,
                first_fail_idx - 1
            )

            test_count += count

            # Retain the previously confirmed passing point
            if best is None:
                best = last_pass_idx

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
# 7. RUN DIAGNOSTIC FOR ALL TEST CHIPS
# ============================================================

results = []

for _, row in test_df.iterrows():

    true_fmax = row["fmax_true_mhz"]

    # Use ORIGINAL dataset prediction windows
    lower_frequency = row["window_low_mhz"]
    upper_frequency = row["window_high_mhz"]

    measured_fmax, test_count = reference_window_search(
        true_fmax,
        lower_frequency,
        upper_frequency
    )

    results.append({

        "unit_id": row["unit_id"],

        "true_fmax_mhz": true_fmax,

        "original_pred_mhz": row["fmax_pred_mhz"],

        "original_window_low_mhz": lower_frequency,

        "original_window_high_mhz": upper_frequency,

        "original_tests_windowed": row["tests_windowed"],

        "diagnostic_measured_fmax_mhz": measured_fmax,

        "diagnostic_tests_required": test_count,

        "original_measured_fmax_mhz":
            row["fmax_measured_windowed_mhz"]

    })


# ============================================================
# 8. CREATE COMPARISON DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df["test_count_match"] = (
    results_df["original_tests_windowed"]
    == results_df["diagnostic_tests_required"]
)

results_df["measured_fmax_match"] = (
    results_df["original_measured_fmax_mhz"]
    == results_df["diagnostic_measured_fmax_mhz"]
)

results_df["test_count_difference"] = (
    results_df["diagnostic_tests_required"]
    - results_df["original_tests_windowed"]
)


# ============================================================
# 9. SAVE DIAGNOSTIC RESULTS
# ============================================================

output_path = "outputs/diagnose_reference_mismatch.csv"

results_df.to_csv(output_path, index=False)


# ============================================================
# 10. PERFORMANCE SUMMARY
# ============================================================

print("\nREFERENCE WINDOW DIAGNOSTIC")
print("---------------------------")

print(f"Total test chips: {len(results_df)}")

print(
    "Original average tests:",
    f"{results_df['original_tests_windowed'].mean():.4f}"
)

print(
    "Diagnostic average tests:",
    f"{results_df['diagnostic_tests_required'].mean():.4f}"
)

print(
    "Original maximum tests:",
    results_df["original_tests_windowed"].max()
)

print(
    "Diagnostic maximum tests:",
    results_df["diagnostic_tests_required"].max()
)

print(
    "Matching test counts:",
    results_df["test_count_match"].sum()
)

print(
    "Mismatched test counts:",
    (~results_df["test_count_match"]).sum()
)

print(
    "Matching measured Fmax:",
    results_df["measured_fmax_match"].sum()
)

print(
    "Average absolute diagnostic Fmax error:",
    f"{np.abs(results_df['true_fmax_mhz'] - results_df['diagnostic_measured_fmax_mhz']).mean():.4f} MHz"
)


# ============================================================
# 11. DISPLAY MISMATCHED CHIPS
# ============================================================

mismatches = results_df[
    ~results_df["test_count_match"]
]

print("\nFIRST 10 MISMATCHED CHIPS")
print("-------------------------")

print(
    mismatches[
        [
            "unit_id",
            "original_tests_windowed",
            "diagnostic_tests_required",
            "original_window_low_mhz",
            "original_window_high_mhz",
            "test_count_difference"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\nDIAGNOSTIC COMPLETED")
print("--------------------")
print(f"Results saved to: {output_path}")
