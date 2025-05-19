"""
This script runs the model prediction, compute metrics and save results
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .model import predict

def inference(
    model,
    X_test,
    y_true,
    threshold: float,
    colour: str,
    save_dir: Path,
    train_scores_sorted: np.ndarray,
):
    """
    Predicts on test set, compute bad class metrics and save JSON + CSV outputs.

    Args:
      model: Trained IsolationForest
      X_test: Feature DataFrame
      y_true: True quality flags
      threshold: Threshold for anomaly detection
      colour: Identifier for naming outputs
      save_dir: Directory to save output
      train_scores_sorted: Sorted training scores
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    #obtaining labels/scores and evaluating the performance
    label_good, prob_bad, score = predict(model, X_test, threshold, train_scores_sorted)
    prec = precision_score(~y_true, ~label_good, zero_division=0)
    rec  = recall_score(~y_true, ~label_good, zero_division=0)
    f1   = f1_score(~y_true, ~label_good, zero_division=0)
    cm   = confusion_matrix(~y_true, ~label_good, labels=[False, True]).tolist()
    
    metrics = {
        'precision_bad': prec,
        'recall_bad': rec,
        'f1_bad': f1,
        'confusion_matrix': cm,
        'threshold': threshold,
    }

    (save_dir / colour).mkdir(parents=True, exist_ok=True)
    with open(save_dir / colour / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'y_true_good': y_true,
        'pred_good': label_good,
        'prob_bad': prob_bad,
        'score': score,
    }).to_csv(save_dir / colour / 'predictions.csv', index=False)
    
    return metrics

