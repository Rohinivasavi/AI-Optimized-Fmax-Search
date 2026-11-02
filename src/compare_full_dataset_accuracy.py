
import pandas as pd

DATA_PATH = "outputs/grid_alignment_detailed_fixed.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    actual = df["fmax_true_mhz"]
    reference = df["fmax_measured_windowed_mhz"]
    floor = df["floor_measured_fmax"]

    reference_error = (reference - actual).abs()
    floor_error = (floor - actual).abs()

    print("=" * 70)
    print("FULL DATASET ACCURACY COMPARISON")
    print("=" * 70)

    print("\nTOTAL CHIPS:", len(df))

    print("\n1. MEAN ABSOLUTE ERROR")
    print("-" * 50)

    print("Original reference:",
          round(reference_error.mean(), 4), "MHz")

    print("Floor algorithm:",
          round(floor_error.mean(), 4), "MHz")

    print("\n2. MEDIAN ABSOLUTE ERROR")
    print("-" * 50)

    print("Original reference:",
          round(reference_error.median(), 4), "MHz")

    print("Floor algorithm:",
          round(floor_error.median(), 4), "MHz")

    print("\n3. WHICH RESULT IS CLOSER?")
    print("-" * 50)

    print("Original reference closer:",
          (reference_error < floor_error).sum())

    print("Floor algorithm closer:",
          (floor_error < reference_error).sum())

    print("Equal accuracy:",
          (floor_error == reference_error).sum())

    print("\n4. REFERENCE TEST COUNT MATCH")
    print("-" * 50)

    matches = (df["floor_tests"] == df["tests_windowed"]).sum()

    print("Matching counts:", matches)
    print("Mismatching counts:", len(df) - matches)

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
