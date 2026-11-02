
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

# 1. Load trained Random Forest model
model = joblib.load("models/fmax_random_forest.pkl")

# 2. Define feature names
features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

# 3. Extract feature importance
importance = model.feature_importances_

# 4. Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# 5. Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 50)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

# 6. Create output folder
output_dir = Path("outputs/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# 7. Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()

# 8. Save graph
plt.savefig(
    output_dir / "feature_importance.png"
)

plt.show()

# 9. Save values
importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nFeature importance analysis completed!")
