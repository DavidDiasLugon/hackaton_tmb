import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.config import FIGURES, REPORTS, BUREAU_SCORE_COLS, SEED

logger = logging.getLogger(__name__)


def run_robustness_tests(model, X: np.ndarray, y: np.ndarray,
                         feature_names: list[str], handles_nan: bool) -> dict:
    logger.info("Running robustness tests...")
    results = {}

    results["bureau_missing"] = _test_bureau_missing(model, X, y, feature_names, handles_nan)
    results["extreme_missing"] = _test_extreme_missing(model, X, y, handles_nan)
    results["threshold_stability"] = _test_threshold_stability(model, X, y, handles_nan)

    _save_robustness_report(results)
    return results


def _test_bureau_missing(model, X, y, feature_names, handles_nan) -> dict:
    bureau_indices = [i for i, f in enumerate(feature_names)
                     if any(b in f for b in ["SCORE_H", "bureau_", "HPG5", "HCR5", "H5OR"])]

    if not bureau_indices:
        return {"status": "no bureau features found"}

    X_clean = X if handles_nan else np.nan_to_num(X, nan=0.0)
    baseline_probs = model.predict_proba(X_clean)[:, 1]
    baseline_auc = roc_auc_score(y, baseline_probs)
    baseline_prauc = average_precision_score(y, baseline_probs)

    results = {"baseline_auc": baseline_auc, "baseline_prauc": baseline_prauc}

    for pct, label in [(0.5, "50%"), (1.0, "100%")]:
        X_mod = X.copy()
        rng = np.random.RandomState(SEED)
        n_mask = int(len(X) * pct)
        mask_rows = rng.choice(len(X), n_mask, replace=False)
        X_mod[np.ix_(mask_rows, bureau_indices)] = 0.0 if not handles_nan else np.nan
        if not handles_nan:
            X_mod = np.nan_to_num(X_mod, nan=0.0)
        probs = model.predict_proba(X_mod)[:, 1]
        auc = roc_auc_score(y, probs)
        prauc = average_precision_score(y, probs)
        results[f"bureau_missing_{label}"] = {"auc": auc, "prauc": prauc}
        logger.info("Bureau %s missing — AUC: %.4f (Δ=%.4f), PR-AUC: %.4f (Δ=%.4f)",
                    label, auc, auc - baseline_auc, prauc, prauc - baseline_prauc)

    return results


def _test_extreme_missing(model, X, y, handles_nan) -> dict:
    X_clean = X if handles_nan else np.nan_to_num(X, nan=0.0)
    baseline_probs = model.predict_proba(X_clean)[:, 1]
    baseline_auc = roc_auc_score(y, baseline_probs)

    results = {"baseline_auc": baseline_auc}
    for pct in [0.1, 0.3, 0.5]:
        X_mod = X.copy()
        rng = np.random.RandomState(SEED)
        mask = rng.random(X_mod.shape) < pct
        X_mod[mask] = 0.0 if not handles_nan else np.nan
        if not handles_nan:
            X_mod = np.nan_to_num(X_mod, nan=0.0)
        probs = model.predict_proba(X_mod)[:, 1]
        auc = roc_auc_score(y, probs)
        results[f"random_{int(pct*100)}pct_missing"] = auc
        logger.info("Random %d%% missing — AUC: %.4f (Δ=%.4f)",
                    int(pct * 100), auc, auc - baseline_auc)

    return results


def _test_threshold_stability(model, X, y, handles_nan) -> dict:
    X_clean = X if handles_nan else np.nan_to_num(X, nan=0.0)
    probs = model.predict_proba(X_clean)[:, 1]

    from sklearn.metrics import precision_recall_curve, f1_score
    prec, rec, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]

    # Test stability: ±10% threshold variation
    results = {}
    for delta in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        t = max(0.01, min(0.99, optimal_thresh + delta))
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y, y_pred)
        results[f"thresh_{t:.3f}"] = f1
        logger.info("Threshold %.3f (Δ=%.2f) — F1: %.4f", t, delta, f1)

    # Plot
    thresholds_plot = np.linspace(0.05, 0.95, 100)
    f1_list = []
    for t in thresholds_plot:
        y_pred = (probs >= t).astype(int)
        f1_list.append(f1_score(y, y_pred))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds_plot, f1_list, color="#3498db", linewidth=2)
    ax.axvline(optimal_thresh, color="red", linestyle="--", label=f"Optimal: {optimal_thresh:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("Threshold Stability Analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "28_threshold_stability.png", dpi=150)
    plt.close(fig)

    return results


def _save_robustness_report(results):
    lines = ["=" * 60, "ROBUSTNESS TEST REPORT", "=" * 60]

    for test_name, test_results in results.items():
        lines.append(f"\n## {test_name.upper()}")
        if isinstance(test_results, dict):
            for k, v in test_results.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for kk, vv in v.items():
                        lines.append(f"    {kk}: {vv:.4f}" if isinstance(vv, float) else f"    {kk}: {vv}")
                else:
                    lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    report = "\n".join(lines)
    (REPORTS / "robustness_report.txt").write_text(report)
    logger.info("Robustness report saved")
