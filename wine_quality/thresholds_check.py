"""
This script analyzes and visualizes the anomaly score distribution for helping in determining
the appropriate threshold selection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import genpareto
from pathlib import Path

from .thresholds import evt_threshold, gamma_threshold
from .config import SEED

def score_distribution(
    scores: np.ndarray,
    colour: str,
    output_dir: Path,
) -> dict:
    """
    Analyzes, plots and saves score distribution diagnostics
    Returns a dict of statistics and a threshold recommendation

    Args:
      scores: Anomaly scores
      colour: Identifier for filenames
      output_dir: Directory for saving diagnostic plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #compute basic statistics and distribution characteristics
    mean = np.mean(scores)
    median = np.median(scores)
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1

    min_val = np.min(scores)
    max_val = np.max(scores) 
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores, fisher = False)

    heavy_tail = kurtosis > 3.5 
    
    #mean-to-median ratio (indicator of right skew)
    mean_median_ratio = mean / median if median != 0 else float('inf')
    
    #EVT-specific diagnostics
    u_pct = 0.90 
    u = np.quantile(scores, u_pct)
    tail_samples = scores[scores > u]
    sufficient_tail_samples = len(tail_samples) >= 20
    
    #recommendation logic
    recommendation = {}
    if heavy_tail and kurtosis > 6 and sufficient_tail_samples:
        recommendation["primary"] = "evt_threshold"
        recommendation["reason"] = f"Heavy-tailed distribution detected (kurtosis = {kurtosis:.3f})"
    elif skewness > 1.0:
        recommendation["primary"] = "gamma_threshold"
        recommendation["reason"] = f"Right-skewed distribution detected (skewness = {skewness:.3f})"
    elif heavy_tail and sufficient_tail_samples:
        recommendation["primary"] = "evt_threshold"
        recommendation["reason"] = "Moderately heavy-tailed distribution detected"
    else:
        recommendation["primary"] = "iqr_threshold"
        recommendation["reason"] = "Distribution appears approximately symmetric"

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    #histogram with KDE and thresholds
    axs[0, 0].hist(scores, bins=30, density=True, alpha=0.6, label='Histogram')
    
    try:
        x_kde = np.linspace(min_val, max_val, 1000)
        kde = stats.gaussian_kde(scores)
        axs[0, 0].plot(x_kde, kde(x_kde), 'r-', label='KDE')
    except:
        pass

    #mark the thresholds
    iqr_thresh = q3 + 1.5 * iqr
    axs[0, 0].axvline(iqr_thresh, color='g', linestyle='--', label=f'IQR (k=1.5): {iqr_thresh:.3f}')
    try:
        evt_thresh = evt_threshold(scores, alpha=0.05)
        axs[0, 0].axvline(evt_thresh, color='r', linestyle='--', label=f'EVT: {evt_thresh:.3f}')
    except:
        pass
    try:
        gamma_thresh = gamma_threshold(scores, alpha=0.05)
        axs[0, 0].axvline(gamma_thresh, color='m', linestyle='--', label=f'Gamma: {gamma_thresh:.3f}')
    except:
        pass
    
    axs[0, 0].set_title('Anomaly Score Distribution with Thresholds')
    axs[0, 0].set_xlabel('Anomaly Score')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()
    
    #QQ plot against normal distribution
    stats.probplot(scores, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Q-Q Plot (vs Normal)')

    # Log-log plot of the tail (only strictly positive values)
    if len(tail_samples) > 0:
        #sorting and pick top 10%
        sorted_scores = np.sort(scores)[::-1]
        n = len(sorted_scores)
        cutoff = max(2, int(0.1 * n))
        x_vals = sorted_scores[:cutoff]
        ranks  = np.arange(1, cutoff + 1) / n

        #masking out any non-positive values before logging
        mask = (x_vals > 0) & (ranks > 0)
        if mask.sum() >= 2:
            x_pos = x_vals[mask]
            y_pos = ranks[mask]

            axs[1, 0].loglog(x_pos, y_pos, 'o-')
            axs[1, 0].set_title('Log-Log Plot of Score Tail')
            axs[1, 0].set_xlabel('Score (log scale)')
            axs[1, 0].set_ylabel('P(X > x) (log scale)')

            #annotating if approximately linear in log–log
            mid = mask.sum() // 2
            x_text = x_pos[mid]
            y_text = y_pos[mid]
            corr = np.corrcoef(np.log(x_pos), np.log(y_pos))[0, 1]
            if corr > 0.9:
                axs[1, 0].annotate(
                    'Approximately linear → Heavy tail',
                    xy=(x_text, y_text),
                    xytext=(x_text * 1.5, y_text * 2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5)
                )

        
    #Box-plot to visualize outliers
    axs[1, 1].boxplot(scores, vert=False)
    axs[1, 1].set_title('Box Plot (outliers as individual points)')
    axs[1, 1].set_xlabel('Anomaly Score')
    axs[1, 1].set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{colour}_threshold_plots.png")
    plt.close()
    
    #additional diagnostics for EVT suitability
    evt_diagnostics = {
        "tail_samples": len(tail_samples),
        "sufficient_for_evt": sufficient_tail_samples,
        "kurtosis": kurtosis,
        "heavy_enough_for_evt": kurtosis > 3.5
    }
    
    #examining parameters for EVT
    if sufficient_tail_samples:
        try:
            u = np.quantile(scores, u_pct)
            z = scores[scores > u] - u
            c, loc, scale = genpareto.fit(z, floc=0)
            evt_diagnostics["gpd_shape_parameter"] = c
            evt_diagnostics["heavy_tail_gpd"] = c > 0
        except:
            pass
    
    return {
        "mean": mean,
        "median": median,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "mean_median_ratio": mean_median_ratio,
        "heavy_tail": heavy_tail,
        "recommendation": recommendation,
        "evt_diagnostics": evt_diagnostics
    }
