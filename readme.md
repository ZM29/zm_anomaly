# Wine Quality Anomaly Detection

This package implements an anomaly detection system for wine quality assessment using Isolation Forest. The model is trained to identify "bad" wines (anomalies) from "good" wines based on their properties.

## Project Overview

The project uses the UCI Wine Quality dataset to demonstrate anomaly detection on tabular data. It includes:

- Data preprocessing and feature engineering
- Model training with hyperparameter tuning
- Multiple threshold selection strategies
- Model evaluation and performance metrics
- Feature importance analysis using SHAP values

## Installation

### Prerequisites

- Python 3.9 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ZM29/zm_anomaly.git
cd zm_anomaly
```

2. Set up a virtual environment (Optional):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Requirements

The following packages are required:
```
joblib==1.5.0
matplotlib==3.10.3
numpy==2.2.6
optuna==4.3.0
pandas==2.2.3
scikit_learn==1.6.1
scipy==1.15.3
shap==0.47.2
statsmodels==0.14.4
typer==0.15.4
```

These are summed up in `requirements.txt`

## Project Structure

```
wine_quality/
│
├── __init__.py
├── cli.py                #Command-line interface
├── config.py             #Configuration parameters
├── data.py               #Data loading and preparation
├── inference.py          #Model evaluation
├── model.py              #Model training and prediction
├── shap.py               #Feature importance analysis
├── thresholds.py         #Threshold selection methods
└── thresholds_check.py   #Threshold diagnostics
```

## Usage

The package provides a command-line interface for common operations:

### 1. Data Preparation

Split the original dataset into train/test sets:

```bash
python -m wine_quality.cli split --raw-dir="data/raw" --out-dir="data"
```

This will create the following files:
- `data/white_train.csv`: 90% of good white wine samples
- `data/red_train.csv`: 90% of good red wine samples
- `data/white_test.csv`: Remaining good and all bad white wine samples
- `data/red_test.csv`: Remaining good and all bad red wine samples

### 2. Model Training

Train Isolation Forest models for both white and red wines:

```bash
python -m wine_quality.cli fit --train-dir="data" --model-dir="artifacts" --n-trials=25
```

This will:
- Apply RobustScaler to normalize the data
- Use Optuna to tune hyperparameters
- Train Isolation Forest models
- Save models and scalers to the artifacts directory

### 3. Evaluation and Analysis

Evaluate the models on test data and generate feature importance analysis:

```bash
python -m wine_quality.cli evaluate --test-dir="data" --model-dir="artifacts" --results-dir="results" --threshold-method="evt"
```

Available threshold methods:
- `quantile`: Simple percentile cutoff
- `iqr`: Interquartile range-based
- `evt`: Extreme Value Theory
- `gamma`: Gamma distribution fitting

This will:
- Generate performance metrics
- Create SHAP plots for feature importance
- Test model performance after dropping least important features
- Save all results to the specified directory

## Outputs

The package produces several outputs:

### Results Directory

- `results/{color}/metrics.json`: Performance metrics (precision, recall, F1 score)
- `results/{color}/predictions.csv`: Individual predictions

### Figures Directory

- `figures/{color}_shap_bar.png`: Feature importance bar chart
- `figures/{color}_shap_beeswarm.png`: SHAP value distribution
- `figures/{color}_threshold_plots.png`: Anomaly score distribution diagnostics

## Advanced Usage

### Selecting a Threshold Method

The package implements four methods for determining the anomaly threshold:

1. **Quantile**: Simple percentile-based threshold
```bash
python -m wine_quality.cli evaluate --threshold-method="quantile"
```

2. **IQR**: Outlier detection using the interquartile range
```bash
python -m wine_quality.cli evaluate --threshold-method="iqr" --k=1.5
```

3. **EVT**: Extreme Value Theory
```bash
python -m wine_quality.cli evaluate --threshold-method="evt" --alpha=0.05
```

4. **Gamma**: Fits a Gamma distribution to the scores
```bash
python -m wine_quality.cli evaluate --threshold-method="gamma" --alpha=0.05
```

## Extension to Other Datasets

While developed for wine quality data, this package can be used for other tabular datasets:

1. Prepare your data in CSV format with similar structure
2. Update the feature engineering in `data.py` if needed
3. Follow the same workflow: split → fit → evaluate
