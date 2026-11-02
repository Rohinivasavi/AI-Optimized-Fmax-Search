
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Project paths
ROOT = Path(__file__).resolve().parent.parent

# Page configuration
st.set_page_config(
    page_title="AI-Optimized Fmax Search",
    page_icon="⚡",
    layout="wide"
)

# Dashboard title
st.title("⚡ AI-Optimized Fmax Search")
st.subheader("AI-Based Semiconductor Testing Optimization")

st.write(
    "A machine-learning-based approach to predict "
    "chip Fmax and reduce frequency testing effort."
)

# Input features used by the trained model
FEATURES = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"
]

# Load dataset and model
df = pd.read_csv(ROOT / "data/raw/synthetic_fmax_dataset.csv")
model = joblib.load(ROOT / "models/fmax_random_forest.pkl")

# Load saved results
comparison = pd.read_csv(ROOT / "outputs/algorithm_comparison.csv")
importance = pd.read_csv(ROOT / "outputs/feature_importance.csv")

# Dataset overview
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Total Features", len(FEATURES))
col3.metric("Unique Chip Units", df["unit_id"].nunique())

st.write("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Dataset distribution
st.header("2. Dataset Distribution")

split_counts = (
    df["split"]
    .value_counts()
    .rename_axis("Split")
    .reset_index(name="Number of Chips")
)

st.bar_chart(
    split_counts,
    x="Split",
    y="Number of Chips"
)

st.dataframe(split_counts, use_container_width=True)

# Machine learning model
st.header("3. Machine Learning Model")

st.info("Model: Random Forest Regression")

st.write(
    "The model predicts actual Fmax using "
    "15 semiconductor input features."
)

st.write("Model file: fmax_random_forest.pkl")

# Calculate test-set performance
test_data = df[df["split"] == "test"]

X_test = test_data[FEATURES]
y_test = test_data["fmax_true_mhz"]

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f} MHz")
col2.metric("RMSE", f"{rmse:.2f} MHz")
col3.metric("R² Score", f"{r2:.4f}")

# Algorithm comparison
st.header("4. Search Algorithm Comparison")

comparison = comparison.sort_values(
    "Average Tests",
    ascending=True
)

st.bar_chart(
    comparison,
    x="Algorithm",
    y="Average Tests"
)

st.dataframe(comparison, use_container_width=True)

# Feature importance
st.header("5. Feature Importance Analysis")

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance,
    x="Feature",
    y="Importance"
)

st.dataframe(importance, use_container_width=True)

# Search results
st.header("6. Search Results")

linear = pd.read_csv(
    ROOT / "outputs/linear_search_results.csv"
)

binary = pd.read_csv(
    ROOT / "outputs/binary_search_results.csv"
)

ml = pd.read_csv(
    ROOT / "outputs/ml_windowed_search_results.csv"
)

search_type = st.selectbox(
    "Select Search Algorithm",
    [
        "Linear Search",
        "Binary Search",
        "ML-Windowed Binary Search"
    ]
)

if search_type == "Linear Search":
    selected = linear

elif search_type == "Binary Search":
    selected = binary

else:
    selected = ml

st.metric(
    "Average Tests Per Chip",
    f"{selected['tests_required'].mean():.2f}"
)

st.dataframe(
    selected.head(20),
    use_container_width=True
)

st.success("Dashboard loaded successfully!")
