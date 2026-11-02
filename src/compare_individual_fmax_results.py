
import pandas as pd

DATA_PATH = "outputs/grid_alignment_detailed_fixed.csv"

def main():

    df = pd.read_csv(DATA_PATH)

    ceiling = df["ceiling_measured_fmax"]
    floor = df["floor_measured_fmax"]
    nearest = df["nearest_measured_fmax"]

    print("=" * 65)
    print("INDIVIDUAL FMAX RESULT COMPARISON")
    print("=" * 65)

    print("\nCeiling vs Floor:")
    print("Identical results:", (ceiling == floor).sum())
    print("Different results:", (ceiling != floor).sum())

    print("\nCeiling vs Nearest:")
    print("Identical results:", (ceiling == nearest).sum())
    print("Different results:", (ceiling != nearest).sum())

    print("\nFloor vs Nearest:")
    print("Identical results:", (floor == nearest).sum())
    print("Different results:", (floor != nearest).sum())

    different = df[
        (ceiling != floor) |
        (ceiling != nearest) |
        (floor != nearest)
    ]

    print("\nTotal chips with different Fmax results:", len(different))

    if not different.empty:
        columns = [
            "unit_id",
            "fmax_true_mhz",
            "fmax_measured_windowed_mhz",
            "ceiling_measured_fmax",
            "floor_measured_fmax",
            "nearest_measured_fmax"
        ]

        print("\nFirst 20 differing records:")
        print(different[columns].head(20).to_string(index=False))

    else:
        print("\nAll three methods produce identical individual Fmax results.")

if __name__ == "__main__":
    main()
