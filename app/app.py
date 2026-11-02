
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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
    "A machine-learning-based approach to predict chip Fmax "
    "and reduce semiconductor frequency testing effort."
)

# Input features
FEATURES = [
    "ro_01", "ro_02", "ro_03", "ro_04", "ro_05",
    "ro_06", "ro_07", "ro_08", "ro_09", "ro_10",
    "iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"
]

# Load project files
df = pd.read_csv(
    ROOT / "data/raw/synthetic_fmax_dataset.csv"
)

model = joblib.load(
    ROOT / "models/fmax_random_forest.pkl"
)

comparison = pd.read_csv(
    ROOT / "outputs/final_algorithm_comparison.csv"
)

detailed = pd.read_csv(
    ROOT / "outputs/final_algorithm_detailed_results.csv"
)

importance = pd.read_csv(
    ROOT / "outputs/feature_importance.csv"
)

# --------------------------------------------------
# 1. DATASET OVERVIEW
# --------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Input Features", len(FEATURES))
col3.metric("Unique Chip Units", df["unit_id"].nunique())

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# 2. DATASET DISTRIBUTION
# --------------------------------------------------

st.header("2. Dataset Distribution")

split_counts = (
    df["split"]
    .value_counts()
    .rename_axis("Split")
    .reset_index(name="Number of Chips")
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Chip Distribution")
    st.bar_chart(
        split_counts,
        x="Split",
        y="Number of Chips"
    )

with col2:
    st.subheader("Split Summary")
    st.dataframe(
        split_counts,
        use_container_width=True
    )

# --------------------------------------------------
# 3. MACHINE LEARNING MODEL
# --------------------------------------------------

st.header("3. Machine Learning Model")

st.info("Model: Random Forest Regression")

st.write(
    "The Random Forest model predicts chip Fmax "
    "using 15 semiconductor input features."
)

st.write("Model file: fmax_random_forest.pkl")

# Evaluate model on test dataset
test_data = df[df["split"] == "test"]

X_test = test_data[FEATURES]
y_test = test_data["fmax_true_mhz"]

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f} MHz")
col2.metric("RMSE", f"{rmse:.2f} MHz")
col3.metric("R² Score", f"{r2:.4f}")

# --------------------------------------------------
# 4. FINAL ALGORITHM COMPARISON
# --------------------------------------------------

st.header("4. Search Algorithm Comparison")

st.write(
    "Comparison of four search algorithms using "
    "the same frequency grid."
)

comparison = comparison.sort_values(
    "Average Tests",
    ascending=True
)

st.bar_chart(
    comparison,
    x="Algorithm",
    y="Average Tests"
)

st.subheader("Algorithm Performance")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# Display comparison graph if available
graph_path = ROOT / "outputs/graphs/final_algorithm_comparison.png"

if graph_path.exists():
    st.subheader("Final Algorithm Comparison Graph")
    st.image(
        str(graph_path),
        use_container_width=True
    )

# --------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------

st.header("5. Feature Importance Analysis")

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.write(
    "Feature importance shows the relative contribution "
    "of input features to the Random Forest predictions."
)

st.bar_chart(
    importance,
    x="Feature",
    y="Importance"
)

st.dataframe(
    importance,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 6. DETAILED SEARCH RESULTS
# --------------------------------------------------

st.header("6. Detailed Search Results")

algorithm = st.selectbox(
    "Select Search Algorithm",
    [
        "Linear Search",
        "Binary Search",
        "ML-Windowed Search",
        "Noisy ML-Windowed Search"
    ]
)

# Map the selected algorithm to its result columns
result_columns = {
    "Linear Search": ("linear_fmax_mhz", "linear_tests"),
    "Binary Search": ("binary_fmax_mhz", "binary_tests"),
    "ML-Windowed Search": ("ml_fmax_mhz", "ml_tests"),
    "Noisy ML-Windowed Search": ("noisy_fmax_mhz", "noisy_tests")
}

fmax_column, tests_column = result_columns[algorithm]

# Calculate selected algorithm metrics
average_tests = detailed[tests_column].mean()
maximum_tests = detailed[tests_column].max()

selected_mae = mean_absolute_error(
    detailed["true_fmax_mhz"],
    detailed[fmax_column]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Tests Per Chip",
    f"{average_tests:.2f}"
)

col2.metric(
    "Maximum Tests",
    int(maximum_tests)
)

col3.metric(
    "Fmax MAE",
    f"{selected_mae:.2f} MHz"
)

# Prepare selected algorithm results
selected_results = detailed[
    [
        "unit_id",
        "true_fmax_mhz",
        "predicted_fmax_mhz",
        fmax_column,
        tests_column
    ]
].copy()

selected_results = selected_results.rename(
    columns={
        fmax_column: "Measured Fmax (MHz)",
        tests_column: "Tests Required",
        "true_fmax_mhz": "True Fmax (MHz)",
        "predicted_fmax_mhz": "Predicted Fmax (MHz)"
    }
)

st.subheader(f"{algorithm} - Chip Results")

st.dataframe(
    selected_results.head(20),
    use_container_width=True,
    hide_index=True
)

# Download selected algorithm results
csv_data = selected_results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Search Results CSV",
    data=csv_data,
    file_name="selected_search_results.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 7. PROJECT SUMMARY
# --------------------------------------------------

st.header("7. Project Summary")

st.write(
    """
    This project uses machine learning to predict semiconductor
    chip maximum operating frequency (Fmax).

    The predicted Fmax is used to narrow the frequency search range,
    helping reduce the number of required testing operations.

    Four algorithms are evaluated:

    - Linear Search
    - Binary Search
    - ML-Windowed Search
    - Noisy ML-Windowed Search

    The noisy implementation uses a simplified Gaussian
    measurement-noise simulation with a standard deviation of 4 MHz.
    """
)

st.success("Dashboard loaded successfully!")
