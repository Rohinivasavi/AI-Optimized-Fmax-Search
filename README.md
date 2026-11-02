# AI-Optimized Fmax Search for Test Time Reduction

## Project Overview

This project uses Machine Learning to predict the maximum operating frequency (Fmax) of semiconductor chips and reduce the number of frequency tests required during chip testing.

A Random Forest Regression model predicts Fmax. The prediction is combined with a conformal prediction interval to guide an ML-windowed search algorithm.

## Objectives

- Predict the Fmax of semiconductor chips.
- Reduce the number of frequency tests.
- Compare Linear Search, Binary Search, and ML-Windowed Search.
- Evaluate prediction accuracy and search efficiency.
- Display results through an interactive Streamlit dashboard.

## Dataset

**Dataset:** `synthetic_fmax_dataset.csv`

- Total records: 6,000
- Input features: 15
- Target column: `fmax_true_mhz`
- Data splits: Training, calibration, and testing.

The dataset is synthetic and is used for project simulation.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regression
- Joblib
- Matplotlib and Seaborn
- Streamlit

## Machine Learning Model

The project uses a Random Forest Regressor.

**Test Performance:**

- MAE: 21.6690 MHz
- MSE: 739.0319
- RMSE: 27.1851 MHz
- R-squared Score: 0.9318

## Search Algorithms

1. Linear Search
2. Binary Search
3. ML-Windowed Search
4. Noisy ML-Windowed Search

### Search Comparison

| Algorithm | Average Tests | Maximum Tests | Fmax MAE |
|---|---:|---:|---:|
| Linear Search | 40.4935 | 74 | 4.9592 MHz |
| Binary Search | 6.4095 | 7 | 4.9592 MHz |
| ML-Windowed Search | 5.3238 | 11 | 4.9592 MHz |
| Noisy ML-Windowed Search | 5.3325 | 11 | 5.8156 MHz |

ML-Windowed Search reduced the average number of tests by approximately 16.94% compared with Binary Search in the project's simulation.

## Prediction Interval

A conformal prediction interval is used to estimate a range around the predicted Fmax.

- Calibration records: 500
- Alpha: 0.10
- Nominal coverage: 90%
- Observed test coverage: 88.70%

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for viewing:

- Dataset overview
- Exploratory data analysis
- Model performance
- Feature importance
- Search algorithm comparison
- Detailed search results

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd AI_Optimized_Fmax_Search
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app/app.py
```

## Project Structure

```text
AI_Optimized_Fmax_Search/
├── app/
├── data/
│   └── raw/
├── models/
├── outputs/
├── src/
├── final_algorithm_comparison.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Execution

The trained model is saved locally using Joblib and loaded by the project scripts. The current implementation uses an offline model and a locally running Streamlit dashboard.

## Limitations

- The dataset is synthetic.
- Frequency testing is simulated rather than performed on physical semiconductor hardware.
- Results may differ in real-world chip testing environments.
- The dashboard is currently designed for local execution.

## Future Enhancements

- Validate the approach using real semiconductor test data.
- Improve prediction accuracy and search efficiency.
- Explore deployment and integration with real testing environments.

## Author

Rohini Vasavi