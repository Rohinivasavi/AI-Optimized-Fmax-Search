
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the dataset
file_path = Path("data/raw/synthetic_fmax_dataset.csv")
df = pd.read_csv(file_path)

# 2. Create folder for saving graphs
output_dir = Path("outputs/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 3. Basic dataset information
print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# 4. Fmax distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["fmax_true_mhz"], kde=True)
plt.title("Fmax Distribution")
plt.xlabel("Actual Fmax (MHz)")
plt.ylabel("Number of Chips")
plt.tight_layout()
plt.savefig(output_dir / "fmax_distribution.png")
plt.show()

# 5. Ring Oscillator distribution
ro_columns = [f"ro_{i:02d}" for i in range(1, 11)]

plt.figure(figsize=(9, 5))
for column in ro_columns:
    sns.kdeplot(df[column], label=column)

plt.title("Ring Oscillator Measurements Distribution")
plt.xlabel("RO Measurement")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "ro_distribution.png")
plt.show()

# 6. Compare Fmax across dataset splits
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="split", y="fmax_true_mhz")
plt.title("Fmax Comparison Across Dataset Splits")
plt.xlabel("Dataset Split")
plt.ylabel("Actual Fmax (MHz)")
plt.tight_layout()
plt.savefig(output_dir / "fmax_split_comparison.png")
plt.show()

# 7. Correlation between RO measurements and actual Fmax
correlation_columns = ro_columns + ["fmax_true_mhz"]
correlation = df[correlation_columns].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("RO Measurements and Fmax Correlation")
plt.tight_layout()
plt.savefig(output_dir / "fmax_correlation.png")
plt.show()

print("\nEDA completed successfully!")
print("Graphs saved inside:", output_dir)
