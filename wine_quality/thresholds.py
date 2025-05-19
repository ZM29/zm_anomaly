"""
This script focuses on the implementation of different threshold that will determine whether
a certain data point is considered bad or good
"""

import numpy as np
from scipy.stats import genpareto, gamma

def quantile_threshold(
    scores: np.ndarray,
    pct: float
) -> float:
    """
    Simply just cuts off at a certain quantile of scores
    
    Args:
        scores: Anomaly scores
        pct: Quantile level
    """
    return np.quantile(scores, pct)


def iqr_threshold(
    scores: np.ndarray,
    k: float = 1.5
) -> float:
    """
    Interquartile method: Q3 + k * (Q3 - Q1)

    Args:
      scores: Anomaly scores
      k: Multiplier for the interquartile range
    
    """
    q1, q3 = np.percentile(scores, [25, 75])
    return q3 + k * (q3 - q1)


def evt_threshold(
    scores: np.ndarray,
    alpha: float = 0.05,
    u_pct: float = 0.90,
) -> float:
    """
    Fits a Generalized Pareto Distribution to the upper tail
    then choose a threshold so that P(score > threshold) = alpha
    If the data doesnt have enough samples, the (1-alpha) quantile is used

    Args:
      scores: Anomaly scores
      alpha: Tail probability
      u_pct: Quantile to define the tail cutoff
    """

    #filtering the values higher than the percentile given
    u = np.quantile(scores, u_pct)
    z = scores[scores > u] - u

    #in the case of not enough data, uses quantile
    if len(z) < 20:
        return np.quantile(scores, 1 - alpha)
    
    #fitting GPD to data
    c, loc, scale = genpareto.fit(z, floc = 0)

    c_safe = c if abs(c) > 1e-6 else np.sign(c) * 1e-6
    n, nu = len(scores), len(z)

    return u + scale / c_safe * ((nu / n / alpha) ** (-c_safe) - 1) #GPD quantile formula


def gamma_threshold(
    scores: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Fits a Gamma distribution to positive-shifted scores
    then use the (1 - alpha) percentile of as threshold

    Args:
      scores: Anomaly scores
      alpha: Tail probability
    """
    min_score = scores.min()
    #shifts only if scores include non-positive values
    shift = -min_score + 1e-9 if min_score <= 0 else 0.0 
    scores_pos = scores + shift

    shape, loc, scale = gamma.fit(scores_pos, floc=0) 
    thr = gamma.ppf(1 - alpha, shape, loc=loc, scale=scale)
    return thr - shift

def select_threshold(
    method: str,
    scores: np.ndarray,
    pct: float = 0.90,
    alpha: float = 0.05,
    k: float = 1.5,
) -> float:
    """
    Selects the chosen threshold function by name

    Args:
      method: One of 'quantile', 'iqr', 'evt', 'gamma'
      scores: Anomaly scores
      pct: Quantile level for 'quantile' and EVT's u_pct
      alpha: Tail probability for 'evt' and 'gamma'
      k: IQR multiplier
    """
    methods = {
        'quantile': lambda: quantile_threshold(scores, pct),
        'iqr': lambda: iqr_threshold(scores, k),
        'evt': lambda: evt_threshold(scores, alpha),
        'gamma': lambda: gamma_threshold(scores, alpha),
    }
    if method not in methods:
        raise ValueError(f"Unknown threshold method: {method}")
    return methods[method]()
