"""
This script contains a CLI for data processing, training, evaluation and SHAP analysis
"""

import typer
from pathlib import Path
import pandas as pd
import numpy as np

from .config import SEED, DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR, FIGURES_DIR, TARGET_TAIL, colours, threshold_method
from .data import parse_csv
from .model import train_iforest, load_model
from .thresholds import select_threshold
from .inference import inference
from .thresholds_check import score_distribution
from .shap import shap_retrain
from .data import feature_engineer

app = typer.Typer()

@app.command()
def split(
    raw_dir: Path = DATA_DIR / "raw",
    out_dir: Path = DATA_DIR,
    seed: int = SEED,
):
    """
    Reads raw CSVs, engineer features, split into train/test and save them

    Args:
      raw_dir: Path to raw files
      out_dir: Path for the output files
      seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    def prep_data(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data by flagging good quality and dropping
        original column
        """
        data = raw.copy()
        data["quality_bool"] = data["quality"] >= 5
        data = data.drop(columns = "quality")
        return feature_engineer(data)

    for colour, fname in [("white", "winequality-white.csv"),
                          ("red",   "winequality-red.csv")]:
        raw = pd.read_csv(raw_dir / fname, sep = ";")
        data = prep_data(raw)

        #sampling
        good = data[data["quality_bool"]]
        train_good = good.sample(frac = 0.9, random_state = seed)
        train = train_good.drop(columns = "quality_bool")
        test = data.drop(train_good.index)

        #saving
        train.to_csv(out_dir / f"{colour}_train.csv", sep = ";", index = False)
        test.to_csv(out_dir / f"{colour}_test.csv",  sep = ";", index = False)

        typer.echo(
            f"{colour:<5}  train={train.shape[0]:>5} rows | "
            f"test={test.shape[0]:>5} rows"
        )

@app.command()
def fit(
    train_dir: Path = DATA_DIR,
    model_dir: Path = ARTIFACTS_DIR,
    use_scaler: bool = True,
    n_trials: int = 25,
):
    """
    Trains IsolationForest for each wine colour and saves artifacts

    Args:
      train_dir: Directory containing training data
      model_dir: Directory containing model artifacts
      use_scaler: Boolean value to use a scaler
      n_trials: Optuna trials for tuning
    """
    np.random.seed(SEED)
    for colour in colours:
        X, _ = parse_csv(train_dir / f"{colour}_train.csv", train_data = True)
        model, scaler = train_iforest(X, colour, model_dir, use_scaler, n_trials)
        typer.echo(f"Trained {colour}")

@app.command()
def evaluate(
    test_dir: Path = DATA_DIR,
    model_dir: Path = ARTIFACTS_DIR,
    results_dir: Path = RESULTS_DIR,
    threshold_method: str = threshold_method,
    alpha: float = 0.05,
    k: float = 1.5,
    do_shap: bool = True,
):
    """
    Evaluates on test sets: select threshold, save metrics and produce SHAP retrain analysis

    Args:
      test_dir: Directory containing test data
      model_dir: Directory containing model artifacts
      results_dir: Directory to store results
      threshold_method: Strategy for thresholding anomalies
      alpha, k, pct: Parameters for threshold selection
    """
    np.random.seed(SEED)
    for colour in colours:
        X_test, y = parse_csv(test_dir / f"{colour}_test.csv", train_data = False)
        model, scaler = load_model(model_dir, colour)
        if scaler:
            X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        #determining scores, thresholds and saving diagnostics
        scores_train = np.load(model_dir / f"{colour}_scores.npy")
        thresh = select_threshold(threshold_method, scores_train, pct = 1 - TARGET_TAIL, alpha = alpha, k = k)
        score_distribution(scores_train, colour, FIGURES_DIR)

        metrics = inference(model, X_test, y, thresh, colour, results_dir, scores_train)
        typer.echo(f"{colour} -> {metrics}")
        
        #shapley values analysis
        X_train, _ = parse_csv(DATA_DIR / f"{colour}_train.csv", train_data = True)
        if scaler:
            X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

        if do_shap:

            shap_retrain(
                model = model,
                X_train = X_train,
                X_test = X_test,
                y_true = y,
                colour = colour,
                output_dir = FIGURES_DIR,
                n_drop = 3
            )


if __name__ == '__main__':
    app()