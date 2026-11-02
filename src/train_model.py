
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/raw/synthetic_fmax_dataset.csv")

# 2. Select input features
features = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm",
    "die_x", "die_y", "radius_sq"
]

target = "fmax_true_mhz"

# 3. Separate training and testing data
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# 4. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# 5. Train the model
print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# 6. Predict Fmax
y_pred = model.predict(X_test)

# 7. Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 40)
print(f"MAE:  {mae:.4f} MHz")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f} MHz")
print(f"R2 Score: {r2:.4f}")

# 8. Save trained model
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "fmax_random_forest.pkl")

print("\nModel saved successfully!")
print("Location: models/fmax_random_forest.pkl")

