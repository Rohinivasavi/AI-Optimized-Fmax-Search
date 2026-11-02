
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load results from all three algorithms
linear = pd.read_csv("outputs/linear_search_results.csv")
binary = pd.read_csv("outputs/binary_search_results.csv")
ml_windowed = pd.read_csv(
    "outputs/ml_windowed_search_results.csv"
)

# 2. Calculate average tests
linear_avg = linear["tests_required"].mean()
binary_avg = binary["tests_required"].mean()
ml_avg = ml_windowed["tests_required"].mean()

# 3. Calculate test reduction percentages
binary_reduction = (
    (linear_avg - binary_avg) / linear_avg
) * 100

ml_vs_binary = (
    (binary_avg - ml_avg) / binary_avg
) * 100

ml_vs_linear = (
    (linear_avg - ml_avg) / linear_avg
) * 100

# 4. Display comparison
print("=" * 55)
print("FMAX SEARCH ALGORITHM COMPARISON")
print("=" * 55)

print("\nAverage tests per chip:")
print(f"Linear Search:             {linear_avg:.2f}")
print(f"Binary Search:             {binary_avg:.2f}")
print(f"ML-Windowed Binary Search:  {ml_avg:.2f}")

print("\nTEST REDUCTION")
print("-" * 55)

print(f"Binary vs Linear: {binary_reduction:.2f}%")
print(f"ML vs Binary:     {ml_vs_binary:.2f}%")
print(f"ML vs Linear:     {ml_vs_linear:.2f}%")

# 5. Create comparison table
comparison = pd.DataFrame({
    "Algorithm": [
        "Linear Search",
        "Binary Search",
        "ML-Windowed Binary Search"
    ],
    "Average Tests": [
        linear_avg,
        binary_avg,
        ml_avg
    ]
})

# 6. Create output folder
output_dir = Path("outputs/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# 7. Plot comparison graph
plt.figure(figsize=(9, 5))

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
    output_dir / "algorithm_comparison.png"
)

plt.show()

# 8. Save comparison table
comparison.to_csv(
    "outputs/algorithm_comparison.csv",
    index=False
)

print("\nComparison graph saved!")
print("Comparison table saved!")
print("\nAlgorithm comparison completed!")
