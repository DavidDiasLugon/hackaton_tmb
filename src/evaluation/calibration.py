import logging
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR_Platt
from sklearn.metrics import average_precision_score, brier_score_loss
from src.utils.config import FIGURES, MODELS_DIR

logger = logging.getLogger(__name__)


def calibrate_model(model, X_val, y_val, X_holdout, y_holdout,
                    model_name: str, handles_nan: bool) -> tuple:
    if not handles_nan:
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_holdout = np.nan_to_num(X_holdout, nan=0.0)

    raw_probs_val = model.predict_proba(X_val)[:, 1]
    raw_probs_holdout = model.predict_proba(X_holdout)[:, 1]

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs_val, y_val)
    iso_probs = iso.predict(raw_probs_holdout)

    # Platt scaling
    platt = LR_Platt()
    platt.fit(raw_probs_val.reshape(-1, 1), y_val)
    platt_probs = platt.predict_proba(raw_probs_holdout.reshape(-1, 1))[:, 1]

    # Compare
    raw_brier = brier_score_loss(y_holdout, raw_probs_holdout)
    iso_brier = brier_score_loss(y_holdout, iso_probs)
    platt_brier = brier_score_loss(y_holdout, platt_probs)

    raw_prauc = average_precision_score(y_holdout, raw_probs_holdout)
    iso_prauc = average_precision_score(y_holdout, iso_probs)
    platt_prauc = average_precision_score(y_holdout, platt_probs)

    logger.info("Calibration results on holdout:")
    logger.info("  Raw   — Brier=%.4f PR-AUC=%.4f", raw_brier, raw_prauc)
    logger.info("  Iso   — Brier=%.4f PR-AUC=%.4f", iso_brier, iso_prauc)
    logger.info("  Platt — Brier=%.4f PR-AUC=%.4f", platt_brier, platt_prauc)

    # Pick best calibration by Brier
    if iso_brier <= platt_brier:
        best_calibrator = iso
        best_name = "isotonic"
        best_probs = iso_probs
    else:
        best_calibrator = platt
        best_name = "platt"
        best_probs = platt_probs

    logger.info("Best calibration: %s", best_name)

    # Save calibrator
    joblib.dump(best_calibrator, MODELS_DIR / "calibrator.joblib")

    # Plot
    _plot_calibration(y_holdout, raw_probs_holdout, iso_probs, platt_probs, model_name)

    calibration_info = {
        "raw_brier": raw_brier, "iso_brier": iso_brier, "platt_brier": platt_brier,
        "raw_prauc": raw_prauc, "iso_prauc": iso_prauc, "platt_prauc": platt_prauc,
        "best_method": best_name,
    }

    return best_calibrator, best_probs, calibration_info


def _plot_calibration(y_true, raw_probs, iso_probs, platt_probs, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for probs, label, color in [
        (raw_probs, "Raw", "#3498db"),
        (iso_probs, "Isotonic", "#2ecc71"),
        (platt_probs, "Platt", "#e74c3c"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        axes[0].plot(prob_pred, prob_true, marker="o", color=color, label=label, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title(f"Calibration Curve — {model_name}")
    axes[0].legend()

    for probs, label, color in [
        (raw_probs, "Raw", "#3498db"),
        (iso_probs, "Isotonic", "#2ecc71"),
        (platt_probs, "Platt", "#e74c3c"),
    ]:
        axes[1].hist(probs, bins=50, alpha=0.5, color=color, label=label, density=True)
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_title("Score Distribution")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "20_calibration.png", dpi=150)
    plt.close(fig)
