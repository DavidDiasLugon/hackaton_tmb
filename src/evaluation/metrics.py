import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, precision_score,
    recall_score, confusion_matrix,
)
from src.utils.config import FIGURES, METRICS, TARGET

logger = logging.getLogger(__name__)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str) -> dict:
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    ks = _ks_statistic(y_true, y_pred)
    optimal_thresh = _optimal_threshold(y_true, y_pred)
    y_class = (y_pred >= optimal_thresh).astype(int)
    prec = precision_score(y_true, y_class, zero_division=0)
    rec = recall_score(y_true, y_class)
    f1 = f1_score(y_true, y_class)

    decile_df = _decile_analysis(y_true, y_pred)
    top5 = _top_capture(y_true, y_pred, 0.05)
    top10 = _top_capture(y_true, y_pred, 0.10)
    top20 = _top_capture(y_true, y_pred, 0.20)

    metrics = {
        "model": model_name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "KS": ks,
        "Optimal_Threshold": optimal_thresh,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Top5_Capture": top5,
        "Top10_Capture": top10,
        "Top20_Capture": top20,
    }

    logger.info("[%s] ROC-AUC=%.4f PR-AUC=%.4f KS=%.4f F1=%.4f Top10=%.1f%%",
                model_name, roc_auc, pr_auc, ks, f1, top10 * 100)

    return metrics


def _ks_statistic(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(tpr - fpr))


def _optimal_threshold(y_true, y_pred):
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thresholds[np.argmax(f1_scores)])


def _top_capture(y_true, y_pred, pct):
    n_top = max(1, int(len(y_true) * pct))
    order = np.argsort(-y_pred)
    return float(y_true[order[:n_top]].sum() / max(y_true.sum(), 1))


def _decile_analysis(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], 10, labels=False, duplicates="drop") + 1
    grouped = df.groupby("decile").agg(
        count=("y_true", "count"),
        n_bad=("y_true", "sum"),
        mean_score=("y_pred", "mean"),
        fpd_rate=("y_true", "mean"),
    ).reset_index()
    grouped["cumulative_bad"] = grouped["n_bad"].cumsum()
    grouped["cumulative_bad_pct"] = grouped["cumulative_bad"] / max(grouped["n_bad"].sum(), 1)
    return grouped


def plot_all_curves(results: dict, y_true: np.ndarray) -> None:
    _plot_roc(results, y_true)
    _plot_pr(results, y_true)
    _plot_ks(results, y_true)
    _plot_gain(results, y_true)
    _plot_lift(results, y_true)
    _save_decile_table(results, y_true)


def _plot_roc(results, y_true):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, res["val_preds"])
        auc = roc_auc_score(y_true, res["val_preds"])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "15_roc_curves.png", dpi=150)
    plt.close(fig)


def _plot_pr(results, y_true):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, res["val_preds"])
        prauc = average_precision_score(y_true, res["val_preds"])
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{name} (PR-AUC={prauc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "16_pr_curves.png", dpi=150)
    plt.close(fig)


def _plot_ks(results, y_true):
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    y_pred = results[best_name]["val_preds"]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks_val = float(np.max(tpr - fpr))

    # Align all arrays to same length (sklearn versions differ)
    n = min(len(fpr), len(tpr), len(thresholds))
    t = thresholds[:n]
    fpr_t = fpr[:n]
    tpr_t = tpr[:n]
    ks_idx = int(np.argmax(tpr_t - fpr_t))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, tpr_t, label="TPR", color="#2ecc71", linewidth=2)
    ax.plot(t, fpr_t, label="FPR", color="#e74c3c", linewidth=2)
    ax.fill_between(t, tpr_t, fpr_t, alpha=0.1, color="#3498db")
    ax.axvline(t[ks_idx], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Threshold")
    ax.set_title(f"KS Curve — {best_name} (KS={ks_val:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "17_ks_curve.png", dpi=150)
    plt.close(fig)


def _plot_gain(results, y_true):
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    y_pred = results[best_name]["val_preds"]
    order = np.argsort(-y_pred)
    sorted_true = y_true[order]
    cumulative = np.cumsum(sorted_true) / max(sorted_true.sum(), 1)
    pct_population = np.arange(1, len(cumulative) + 1) / len(cumulative)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pct_population, cumulative, color="#3498db", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("% Population")
    ax.set_ylabel("% FPD Captured")
    ax.set_title(f"Gain Chart — {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "18_gain_chart.png", dpi=150)
    plt.close(fig)


def _plot_lift(results, y_true):
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    y_pred = results[best_name]["val_preds"]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], 10, labels=False, duplicates="drop") + 1
    decile_rate = df.groupby("decile")["y_true"].mean()
    global_rate = y_true.mean()
    lift = decile_rate / global_rate

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(lift.index.astype(str), lift.values, color="#3498db")
    ax.axhline(1.0, color="red", linestyle="--", label="Baseline (lift=1)")
    ax.set_xlabel("Decile (10=highest risk)")
    ax.set_ylabel("Lift")
    ax.set_title(f"Lift Chart — {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "19_lift_chart.png", dpi=150)
    plt.close(fig)


def _save_decile_table(results, y_true):
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    y_pred = results[best_name]["val_preds"]
    decile_df = _decile_analysis(y_true, y_pred)
    decile_df.to_csv(METRICS / "decile_analysis.csv", index=False)
    logger.info("Decile table saved to %s", METRICS / "decile_analysis.csv")


def save_metrics_summary(all_metrics: list[dict]) -> None:
    df = pd.DataFrame(all_metrics)
    df.to_csv(METRICS / "model_comparison.csv", index=False)
    logger.info("Metrics summary saved to %s", METRICS / "model_comparison.csv")
