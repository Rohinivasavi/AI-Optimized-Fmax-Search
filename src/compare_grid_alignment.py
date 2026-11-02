
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/synthetic_fmax_dataset.csv"

FREQ_MIN = 1600
FREQ_MAX = 2400
STEP = 10


def binary_search(grid, low, high, true_fmax):
    """Return highest passing frequency and number of tests."""
    best = None
    tests = 0

    while low <= high:
        mid = (low + high) // 2
        frequency = grid[mid]
        tests += 1

        if frequency <= true_fmax:
            best = frequency
            low = mid + 1
        else:
            high = mid - 1

    return best, tests


def run_search(grid, true_fmax, low_idx, high_idx):
    """Run window search, fallback, and upper expansion."""

    tests = 0

    # Phase 1: Search inside the aligned window
    best, count = binary_search(
        grid, low_idx, high_idx, true_fmax
    )
    tests += count

    # Phase 2: If no passing point exists, search below window
    if best is None:
        best, count = binary_search(
            grid, 0, low_idx - 1, true_fmax
        )
        tests += count

    # Phase 3: If upper boundary passes, expand upwards
    elif best == grid[high_idx]:

        last_pass_idx = high_idx
        stride = 1

        while last_pass_idx + stride < len(grid):
            next_idx = last_pass_idx + stride
            frequency = grid[next_idx]
            tests += 1

            if frequency <= true_fmax:
                last_pass_idx = next_idx
                stride *= 2
            else:
                break

        search_start = last_pass_idx + 1
        search_end = min(
            last_pass_idx + stride - 1,
            len(grid) - 1
        )

        if search_start <= search_end:
            best, count = binary_search(
                grid,
                search_start,
                search_end,
                true_fmax
            )
            tests += count
        else:
            best = grid[last_pass_idx]

    return best, tests


def get_aligned_indices(low, high, method, grid):
    """Map continuous window boundaries to grid indices."""

    if method == "Ceiling":
        low_idx = np.searchsorted(grid, low, side="left")
        high_idx = np.searchsorted(grid, high, side="right") - 1

    elif method == "Floor":
        low_idx = np.searchsorted(grid, low, side="right") - 1
        high_idx = np.searchsorted(grid, high, side="right") - 1

    elif method == "Nearest":
        low_idx = int(np.argmin(np.abs(grid - low)))
        high_idx = int(np.argmin(np.abs(grid - high)))

    else:
        raise ValueError("Unknown alignment method")

    low_idx = max(0, min(low_idx, len(grid) - 1))
    high_idx = max(0, min(high_idx, len(grid) - 1))

    # Ensure the selected window has a valid index range
    if low_idx > high_idx:
        return None

    return low_idx, high_idx


def main():

    df = pd.read_csv(DATA_PATH)

    grid = np.arange(
        FREQ_MIN,
        FREQ_MAX + STEP,
        STEP
    )

    methods = ["Ceiling", "Floor", "Nearest"]

    results = []

    print("=" * 65)
    print("GRID ALIGNMENT COMPARISON")
    print("=" * 65)

    for method in methods:

        test_counts = []
        measured_results = []

        for _, row in df.iterrows():

            true_fmax = float(row["fmax_true_mhz"])
            window_low = float(row["window_low_mhz"])
            window_high = float(row["window_high_mhz"])

            aligned = get_aligned_indices(
                window_low,
                window_high,
                method,
                grid
            )

            if aligned is None:
                best = None
                tests = 0
            else:
                low_idx, high_idx = aligned

                best, tests = run_search(
                    grid,
                    true_fmax,
                    low_idx,
                    high_idx
                )

            test_counts.append(tests)
            measured_results.append(best)

        df[f"{method.lower()}_tests"] = test_counts

        matches = (
            df[f"{method.lower()}_tests"]
            == df["tests_windowed"]
        ).sum()

        results.append({
            "Method": method,
            "Average Tests": np.mean(test_counts),
            "Maximum Tests": np.max(test_counts),
            "Matching Reference Counts": matches,
            "Mismatching Counts": len(df) - matches
        })

    result_df = pd.DataFrame(results)

    print("\nRESULTS FOR ALL 6000 CHIPS")
    print(result_df.to_string(index=False))

    print("\nREFERENCE AVERAGE TESTS:")
    print(df["tests_windowed"].mean())

    print("\nU02001 COMPARISON:")

    chip = df[df["unit_id"] == "U02001"].iloc[0]

    print("Reference tests:", chip["tests_windowed"])

    for method in methods:
        print(
            f"{method} alignment tests:",
            chip[f"{method.lower()}_tests"]
        )

    output_path = "outputs/grid_alignment_comparison.csv"
    result_df.to_csv(output_path, index=False)

    print("\nComparison saved to:", output_path)


if __name__ == "__main__":
    main()
