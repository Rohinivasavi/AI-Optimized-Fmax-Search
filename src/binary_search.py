
import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# 2. Select test records
test_df = df[df["split"] == "test"]

# 3. Define frequency range
frequency_min = 1500
frequency_max = 2350
frequency_step = 10
# 4. Binary search function
def binary_search(true_fmax):

    low = frequency_min
    high = frequency_max
    test_count = 0

    while high - low > frequency_step:

        # Find midpoint
        mid = ((low + high) // 2 // frequency_step) * frequency_step

        test_count += 1

        # Simulate chip pass/fail response
        if mid <= true_fmax:
            low = mid
        else:
            high = mid

    return low, test_count

# 5. Run binary search for every chip
results = []

for _, row in test_df.iterrows():

    true_fmax = row["fmax_true_mhz"]

    measured_fmax, tests = binary_search(true_fmax)

    results.append({
        "unit_id": row["unit_id"],
        "true_fmax_mhz": true_fmax,
        "measured_fmax_mhz": measured_fmax,
        "tests_required": tests
    })

# 6. Convert results into DataFrame
results_df = pd.DataFrame(results)

# 7. Calculate performance
average_tests = results_df["tests_required"].mean()

print("=" * 50)
print("BINARY SEARCH RESULTS")
print("=" * 50)

print("\nTotal chips tested:", len(results_df))
print("Average tests per chip:", round(average_tests, 2))
print("Maximum tests:", results_df["tests_required"].max())

print("\nFirst five results:")
print(results_df.head())

# 8. Save results
results_df.to_csv(
    "outputs/binary_search_results.csv",
    index=False
)

print("\nBinary search completed!")
