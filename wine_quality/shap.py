"""
This script focuses on the implementation of the Shapley values, the plotting of these values and the retraining 
"""

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from .model import predict
from .thresholds import select_threshold
from .config import threshold_method, SEED

def shap_plots(
    model: IsolationForest,
    X_train: pd.DataFrame,
    output_dir: Path,
    colour: str,
) -> tuple[shap.TreeExplainer, np.ndarray]: 
    """
    Computes and saves SHAP summary plots for feature importance. It saves a:
    - Bar chart: average absolute importance per feature
    - Beeswarm: distribution of SHAP values for each feature

    Args:
      model: Trained IsolationForest
      X_train: Training data as DataFrame
      output_dir: Directory to save figures.
      colour: Identifier for filenames
    """
    output_dir.mkdir(parents = True, exist_ok = True)
    
    #explainer and computation of shap values
    explainer = shap.TreeExplainer(model, X_train)
    values = explainer.shap_values(X_train)
    rng = np.random.default_rng(SEED)

    shap.summary_plot(values, X_train, show = False, 
                      plot_type = 'bar', rng = rng)
    plt.savefig(output_dir / f"{colour}_shap_bar.png", dpi = 150)
    plt.close()
    
    shap.summary_plot(values, X_train, show = False,
                      rng = rng)
    plt.savefig(output_dir / f"{colour}_shap_beeswarm.png", dpi = 150)
    plt.close()

    return explainer, values


def shap_retrain(
    model: IsolationForest,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_true,
    colour: str,
    output_dir: Path,
    n_drop: int = 3,
    threshold_method: str = threshold_method,
    alpha: float = 0.05,
    k: float = 1.5,
    pct: float = 0.90,
):
    """
    Generates and identifies the least important featres by using the mean absolute shap values.
    Then drops these features and retrains the forest on the new subset to check if dropping will
    lead to performance increase. In this assessment I stop after doing it once but in an actual
    business case, I would've continued

    Args:
      model: Original trained model
      X_train, X_test: Scaled feature DataFrames
      y_true: True labels
      colour: Identifier for naming outputs
      output_dir: Directory path for saving plots
      n_drop: Number of features to remove
      threshold_method: Strategy for thresholding anomalies
      alpha, k, pct: Parameters for threshold selection
    """
    _, values = shap_plots(model, X_train, output_dir, colour)

    #computing the mean abs shap values
    mean_abs = np.abs(values).mean(axis = 0)

    #determining the least important features and dropping from set
    drop_idx = np.argsort(mean_abs)[:n_drop]
    drop_cols = X_train.columns[drop_idx].tolist()
    X_tr = X_train.drop(columns = drop_cols)
    X_te = X_test.drop(columns = drop_cols)

    new_model = IsolationForest(**model.get_params()).fit(X_tr)
    scores_small = -new_model.decision_function(X_tr)

    #new threshold and evaluating performance
    threshold_small = select_threshold(threshold_method, scores_small, pct = pct,
        alpha = alpha, k = k)
    label, _, _ = predict(new_model, X_te, threshold_small, np.sort(scores_small))
    f1 = f1_score(~y_true, ~label)
    prec = precision_score(~y_true, ~label, zero_division = 0)
    rec = recall_score(~y_true, ~label, zero_division = 0)
    cm = confusion_matrix(~y_true, ~label, labels = [False, True])

    print(f"Threshold method: {threshold_method}")
    print(f"Dropped features: {drop_cols}")
    print(f"After dropping {n_drop}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    print("Confusion matrix (true vs. pred):")
    print(cm)