import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from src.utils.config import FIGURES, EXPLAINABILITY, SEED

logger = logging.getLogger(__name__)


def run_explainability(model, X: np.ndarray, y: np.ndarray,
                       feature_names: list[str], model_name: str,
                       handles_nan: bool) -> None:
    logger.info("Running explainability for %s...", model_name)

    X_clean = X if handles_nan else np.nan_to_num(X, nan=0.0)

    _native_importance(model, feature_names, model_name)
    _shap_analysis(model, X_clean, feature_names, model_name)
    _permutation_importance(model, X_clean, y, feature_names, model_name)
    _local_explanations(model, X_clean, feature_names, model_name)


def _native_importance(model, feature_names, model_name):
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        lr = model.named_steps.get("lr")
        if lr is not None and hasattr(lr, "coef_"):
            imp = np.abs(lr.coef_[0])

    if imp is None:
        return

    importance_df = pd.DataFrame({
        "feature": feature_names, "importance": imp
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(EXPLAINABILITY / "feature_importance.csv", index=False)

    top20 = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20)), top20["importance"].values, color="#3498db")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values)
    ax.invert_yaxis()
    ax.set_title(f"Top 20 Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "21_feature_importance.png", dpi=150)
    plt.close(fig)


def _shap_analysis(model, X, feature_names, model_name):
    try:
        sample_size = min(2000, len(X))
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]

        if hasattr(model, "named_steps"):
            explainer = shap.Explainer(model.predict_proba, X_sample,
                                       feature_names=feature_names)
            shap_values = explainer(X_sample)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # Beeswarm
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         show=False, max_display=20)
        plt.title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        plt.savefig(FIGURES / "22_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Mean |Value| — {model_name}")
        plt.tight_layout()
        plt.savefig(FIGURES / "23_shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Dependence plots for top 5
        if hasattr(shap_values, "values"):
            sv = shap_values.values
        else:
            sv = shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        top5_idx = np.argsort(-mean_abs)[:5]
        for rank, fi in enumerate(top5_idx):
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(fi, sv, X_sample, feature_names=feature_names,
                                show=False, ax=ax)
            plt.tight_layout()
            plt.savefig(FIGURES / f"24_shap_dep_{rank}_{feature_names[fi]}.png",
                       dpi=150, bbox_inches="tight")
            plt.close("all")

        logger.info("SHAP analysis complete")

    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)


def _permutation_importance(model, X, y, feature_names, model_name):
    try:
        result = permutation_importance(model, X, y, n_repeats=10,
                                        random_state=SEED, scoring="average_precision",
                                        n_jobs=-1)
        perm_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(EXPLAINABILITY / "permutation_importance.csv", index=False)

        top20 = perm_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top20)), top20["importance_mean"].values,
                xerr=top20["importance_std"].values, color="#e74c3c", alpha=0.7)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["feature"].values)
        ax.invert_yaxis()
        ax.set_title(f"Permutation Importance — {model_name}")
        ax.set_xlabel("Mean PR-AUC Decrease")
        fig.tight_layout()
        fig.savefig(FIGURES / "25_permutation_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning("Permutation importance failed: %s", e)


def _local_explanations(model, X, feature_names, model_name):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            return

        low_idx = np.argmin(probs)
        high_idx = np.argmax(probs)
        mid_val = np.median(probs)
        mid_idx = np.argmin(np.abs(probs - mid_val))

        records = []
        for label, idx in [("Low Risk", low_idx), ("Medium Risk", mid_idx), ("High Risk", high_idx)]:
            record = {"label": label, "prob_fpd": float(probs[idx])}
            for i, fn in enumerate(feature_names):
                record[fn] = float(X[idx, i])
            records.append(record)

        pd.DataFrame(records).to_csv(EXPLAINABILITY / "local_examples.csv", index=False)
        logger.info("Local explanations saved")
    except Exception as e:
        logger.warning("Local explanations failed: %s", e)
