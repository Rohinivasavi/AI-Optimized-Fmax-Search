
import pandas as pd
import numpy as np

DATA_PATH = "outputs/floor_measurement_patterns.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    # Error of original reference measurement
    df["reference_error"] = (
        df["fmax_measured_windowed_mhz"]
        - df["fmax_true_mhz"]
    )

    # Error of our Floor algorithm
    df["floor_error"] = (
        df["floor_measured_fmax"]
        - df["fmax_true_mhz"]
    )

    df["reference_absolute_error"] = df["reference_error"].abs()
    df["floor_absolute_error"] = df["floor_error"].abs()

    reference_better = (
        df["reference_absolute_error"]
        < df["floor_absolute_error"]
    ).sum()

    floor_better = (
        df["floor_absolute_error"]
        < df["reference_absolute_error"]
    ).sum()

    equal_accuracy = (
        df["reference_absolute_error"]
        == df["floor_absolute_error"]
    ).sum()

    print("=" * 70)
    print("REFERENCE VS FLOOR - ACTUAL FMAX ERROR")
    print("=" * 70)

    print("\nTOTAL CHIPS:", len(df))

    print("\n1. MEAN ABSOLUTE ERROR")
    print("-" * 50)

    print(
        "Original reference:",
        round(df["reference_absolute_error"].mean(), 4),
        "MHz"
    )

    print(
        "Floor algorithm:",
        round(df["floor_absolute_error"].mean(), 4),
        "MHz"
    )

    print("\n2. WHICH RESULT IS CLOSER TO TRUE FMAX?")
    print("-" * 50)

    print("Original reference is closer:", reference_better)
    print("Floor result is closer:", floor_better)
    print("Equal accuracy:", equal_accuracy)

    print("\n3. ERROR STATISTICS")
    print("-" * 50)

    print(
        "Original reference median error:",
        round(df["reference_absolute_error"].median(), 4),
        "MHz"
    )

    print(
        "Floor median error:",
        round(df["floor_absolute_error"].median(), 4),
        "MHz"
    )

    print("\n4. FIRST 20 DIFFERING RESULTS")
    print("-" * 50)

    columns = [
        "unit_id",
        "fmax_true_mhz",
        "fmax_measured_windowed_mhz",
        "floor_measured_fmax",
        "reference_absolute_error",
        "floor_absolute_error"
    ]

    print(
        df[columns].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
