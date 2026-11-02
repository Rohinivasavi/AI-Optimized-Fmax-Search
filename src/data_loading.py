
import pandas as pd
from pathlib import Path

# 1. Define the dataset location
file_path = Path("data/raw/synthetic_fmax_dataset.csv")

# 2. Check whether the dataset exists
if not file_path.exists():
    print("Dataset not found!")
    print("Please check the CSV file location.")
else:
    # 3. Load the CSV dataset
    df = pd.read_csv(file_path)

    # 4. Display basic information
    print("=" * 50)
    print("FMAX DATASET INFORMATION")
    print("=" * 50)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nDataset loaded successfully!")
