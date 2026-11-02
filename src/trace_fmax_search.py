
import pandas as pd

# Original dataset path
DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

# Search configuration used in our diagnostic
FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10

CHIP_ID = "U02001"


def trace_binary_search(grid, low, high, true_fmax, phase, log):
    """
    Binary search that records every frequency tested.
    Returns the highest passing frequency and test count.
    """
    best = None

    while low <= high:
        mid = (low + high) // 2
        frequency = grid[mid]
        passed = frequency <= true_fmax

        log.append({
            "Phase": phase,
            "Frequency_MHz": frequency,
            "Result": "PASS" if passed else "FAIL"
        })

        if passed:
            best = frequency
            low = mid + 1
        else:
            high = mid - 1

    return best


def main():
    df = pd.read_csv(DATA_PATH)

    chip = df[df["unit_id"] == CHIP_ID]

    if chip.empty:
        print(f"Chip {CHIP_ID} not found in dataset.")
        return

    row = chip.iloc[0]

    true_fmax = float(row["fmax_true_mhz"])
    window_low = float(row["window_low_mhz"])
    window_high = float(row["window_high_mhz"])
    reference_tests = int(row["tests_windowed"])

    grid = list(range(FREQ_MIN, FREQ_MAX + STEP, STEP))

    # Find grid points inside the reference prediction window
    window_indices = [
        i for i, frequency in enumerate(grid)
        if window_low <= frequency <= window_high
    ]

    if not window_indices:
        print("No grid points found inside the prediction window.")
        return

    log = []

    print("=" * 55)
    print(f"SEARCH TRACE FOR CHIP: {CHIP_ID}")
    print("=" * 55)
    print(f"True Fmax       : {true_fmax} MHz")
    print(f"Reference window: {window_low} - {window_high} MHz")
    print(f"Reference tests : {reference_tests}")
    print()

    # Phase 1: Search inside the prediction window
    print("PHASE 1: SEARCH INSIDE PREDICTION WINDOW")

    best = trace_binary_search(
        grid,
        window_indices[0],
        window_indices[-1],
        true_fmax,
        "Window search",
        log
    )

    # Phase 2: If no passing frequency exists in the window,
    # search below the lower window boundary.
    if best is None:
        print("\nPHASE 2: SEARCH BELOW LOWER WINDOW BOUNDARY")

        best = trace_binary_search(
            grid,
            0,
            window_indices[0] - 1,
            true_fmax,
            "Lower fallback",
            log
        )

    # Phase 3: If the highest passing point is at the upper
    # window edge, expand the search above the window.
    elif best == grid[window_indices[-1]]:
        print("\nPHASE 3: EXPAND ABOVE UPPER WINDOW BOUNDARY")

        last_pass_idx = window_indices[-1]
        stride = 1

        while last_pass_idx + stride < len(grid):
            next_idx = last_pass_idx + stride
            frequency = grid[next_idx]
            passed = frequency <= true_fmax

            log.append({
                "Phase": "Upper expansion",
                "Frequency_MHz": frequency,
                "Result": "PASS" if passed else "FAIL"
            })

            if passed:
                last_pass_idx = next_idx
                stride *= 2
            else:
                break

        # Search only the untested points after the last
        # passing point, avoiding duplicate test counts.
        search_start = last_pass_idx + 1
        search_end = min(
            last_pass_idx + stride - 1,
            len(grid) - 1
        )

        if search_start <= search_end:
            best = trace_binary_search(
                grid,
                search_start,
                search_end,
                true_fmax,
                "Upper binary search",
                log
            )
        else:
            best = grid[last_pass_idx]

    print("\n" + "=" * 55)
    print("FREQUENCY-BY-FREQUENCY TRACE")
    print("=" * 55)

    for number, item in enumerate(log, start=1):
        print(
            f"Test {number:02d} | "
            f"{item['Phase']:<20} | "
            f"{item['Frequency_MHz']} MHz | "
            f"{item['Result']}"
        )

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"Total diagnostic tests : {len(log)}")
    print(f"Reference tests         : {reference_tests}")
    print(f"Diagnostic best Fmax    : {best} MHz")
    print(f"Actual Fmax             : {true_fmax} MHz")
    print(f"Difference in test count: {len(log) - reference_tests}")


if __name__ == "__main__":
    main()
