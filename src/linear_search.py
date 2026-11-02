
import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# 2. Select test records
test_df = df[df["split"] == "test"]

# 3. Define frequency search range
frequency_min = 1500
frequency_max = 2350
frequency_step = 10

# 4. Function to simulate chip testing
def linear_search(true_fmax):

    frequency = frequency_min
    test_count = 0
    last_passing_frequency = frequency_min

    while frequency <= frequency_max:

        test_count += 1

        # Chip passes if tested frequency is <= actual Fmax
        if frequency <= true_fmax:
            last_passing_frequency = frequency
            frequency += frequency_step
        else:
            break

    return last_passing_frequency, test_count

# 5. Run linear search for every test chip
results = []

for _, row in test_df.iterrows():

    true_fmax = row["fmax_true_mhz"]

    measured_fmax, tests = linear_search(true_fmax)

    results.append({
        "unit_id": row["unit_id"],
        "true_fmax_mhz": true_fmax,
        "measured_fmax_mhz": measured_fmax,
        "tests_required": tests
    })

# 6. Convert results into DataFrame
results_df = pd.DataFrame(results)

# 7. Calculate average test count
average_tests = results_df["tests_required"].mean()

print("=" * 50)
print("LINEAR SEARCH RESULTS")
print("=" * 50)

print("\nTotal chips tested:", len(results_df))

print("Average tests per chip:", round(average_tests, 2))

print("Maximum tests:", results_df["tests_required"].max())

print("\nFirst five results:")
print(results_df.head())

# 8. Save results
results_df.to_csv(
    "outputs/linear_search_results.csv",
    index=False
)

print("\nLinear search completed!")
