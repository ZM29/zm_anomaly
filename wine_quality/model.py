"""
This script handles training, tuning, saving and prediction for IsolationForest models
"""

import numpy as np
from pathlib import Path
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import optuna
import pandas as pd

from .config import SEED, TARGET_TAIL


def tune_iforest(
    X: np.ndarray,
    n_trials: int = 25,
    seed: int = SEED,
) -> dict:
    """
    Uses Optuna to find hyperparameters that maximize separation between tail and bulk

    Args:
      X: 2D array of scaled features
      n_trials: Number of Optuna trials
      seed: Random seed for reproducibility
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step = 100),
            'max_samples':  trial.suggest_float('max_samples', 0.6, 1.0, step = 0.1),
            'max_features': trial.suggest_float('max_features', 0.6, 1.0, step = 0.1),
        }
        model = IsolationForest(**params, random_state=seed).fit(X)
        s = -model.decision_function(X)

        #separate bulk and tail by the target_tail
        p = np.quantile(s, 1 - TARGET_TAIL)
        bulk_mean = s[s <= p].mean()
        tail_mean = s[s > p].mean()
        return tail_mean - bulk_mean

    study = optuna.create_study(
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed = seed)
    )
    study.optimize(objective, n_trials = n_trials, show_progress_bar = False, n_jobs = 1)
    return study.best_params


def train_iforest(
    X: pd.DataFrame,
    colour: str,
    save_path: Path,
    use_scaler: bool = True,
    n_trials: int = 25,
    seed: int = SEED,
) -> tuple[IsolationForest, RobustScaler]:
    """
    Scales, tunes, trains and saves an IsolationForest model

    Args:
      X: Training DataFrame
      colour: Identifier for filenames
      save_path: Directory to save model artifacts
      use_scaler: Whether to apply RobustScaler
      n_trials: Optuna trials for tuning
      seed: Random seed for reproducibility
    """
    save_path.mkdir(parents = True, exist_ok = True)

    #fits scaler and if no scaler is found, stop the whole process
    if use_scaler:
        scaler = RobustScaler().fit(X)
        X_input = pd.DataFrame(scaler.transform(X), columns = X.columns)
        dump(scaler, save_path / f"scaler_{colour}.joblib")
    else:
        raise FileNotFoundError("Scaler missing while the model was trained with scaling")

    #tuning, training and saving model
    best_params = tune_iforest(X_input.values, n_trials = n_trials, seed = seed)
    model = IsolationForest(**best_params, random_state = seed).fit(X_input)
    dump(model, save_path / f"iso_{colour}.joblib")

    #saves sorted training scores
    train_scores = -model.decision_function(X_input)
    np.save(save_path / f"{colour}_scores.npy", np.sort(train_scores))

    return model, scaler


def load_model(
    model_dir: Path,
    colour: str
) -> tuple[IsolationForest, RobustScaler]:
    """
    Loads a saved IsolationForest model and its scaler (if any).

    Args:
      model_dir: Path to artifacts directory
      colour: Identifier used when saving
    """
    model = load(model_dir / f"iso_{colour}.joblib")
    try:
        scaler = load(model_dir / f"scaler_{colour}.joblib")
    except FileNotFoundError:
        scaler = None
        print("----------Scaler missing----------")
    return model, scaler


def predict(
    model: IsolationForest,
    X: pd.DataFrame,
    thresh: float,
    train_scores_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scores data, computes anomaly probability and assigns a label

    Args:
      model: Trained IsolationForest
      X: Feature DataFrame
      thresh: Numeric threshold for anomaly flag
      train_scores_sorted: Sorted training scores for probability mapping
    """
    score = -model.decision_function(X)

    idx = np.clip(
        np.searchsorted(train_scores_sorted, score, side='right'),
        0, len(train_scores_sorted)
    )
    prob =  idx / len(train_scores_sorted)
    label = score <= thresh
    return label, prob, score