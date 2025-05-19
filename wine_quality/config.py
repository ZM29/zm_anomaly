"""
This script focuses on the configurations of the models and paths
"""
from pathlib import Path

#constants
SEED = 29
TARGET_TAIL = 0.10

#directories
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")

#wine types
colours = ["white", "red"]

#options: quantile, iqr, evt, gamma
threshold_method = "evt"

