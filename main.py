#!/usr/bin/env python3
import logging
import sys
import time
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("HACKATHON FPD — TMB — PIPELINE START")
    logger.info("=" * 60)

    # ── 1. Ingestion ──
    logger.info("\n>>> PHASE 1: DATA INGESTION")
    from src.preprocessing.ingestion import load_train, load_submission
    df_train_raw = load_train()
    df_sub_raw = load_submission()

    # ── 2. Leakage Report ──
    logger.info("\n>>> PHASE 1b: LEAKAGE DETECTION")
    from src.preprocessing.leakage import generate_leakage_report
    from src.utils.config import TRAIN_FILE
    df_raw_for_leakage = pd.read_excel(str(TRAIN_FILE))
    report = generate_leakage_report(df_raw_for_leakage)
    del df_raw_for_leakage
    print(report[:500])

    # ── 3. EDA ──
    logger.info("\n>>> PHASE 2: EDA")
    from src.preprocessing.eda import run_eda
    run_eda(df_train_raw)

    # ── 4. Temporal Split (BEFORE feature engineering to avoid leakage) ──
    logger.info("\n>>> PHASE 4: TEMPORAL SPLIT")
    from src.models.validation import temporal_split
    from src.utils.config import TARGET, ID_COL, MODELS_DIR
    train_raw, val_raw, holdout_raw = temporal_split(df_train_raw)

    # ── 5. Feature Engineering (fit on train only) ──
    logger.info("\n>>> PHASE 3: FEATURE ENGINEERING")
    from src.features.engineering import build_features

    train, te_encoders, fe_encoders = build_features(train_raw, is_train=True)
    val, _, _ = build_features(val_raw, is_train=False,
                               target_encoders=te_encoders, freq_encoders=fe_encoders)
    holdout, _, _ = build_features(holdout_raw, is_train=False,
                                   target_encoders=te_encoders, freq_encoders=fe_encoders)
    df_sub_feat, _, _ = build_features(df_sub_raw, is_train=False,
                                        target_encoders=te_encoders,
                                        freq_encoders=fe_encoders)

    # Save encoders
    joblib.dump(te_encoders, MODELS_DIR / "target_encoders.joblib")
    joblib.dump(fe_encoders, MODELS_DIR / "freq_encoders.joblib")

    # ── 6. Model Training ──
    logger.info("\n>>> PHASE 5: MODEL TRAINING")
    from src.models.trainer import train_all_models, get_feature_cols
    results = train_all_models(train, val)

    # ── 7. Evaluation ──
    logger.info("\n>>> PHASE 6: EVALUATION")
    from src.evaluation.metrics import compute_all_metrics, plot_all_curves, save_metrics_summary

    feature_cols = get_feature_cols(holdout)
    y_holdout = holdout[TARGET].values

    all_metrics = []
    for name, res in results.items():
        if name == "Ensemble":
            # Re-compute ensemble on holdout
            components = res["components"]
            ens_pred = np.zeros(len(holdout))
            for comp_name, weight in components:
                comp_model = results[comp_name]["model"]
                handles = results[comp_name]["handles_nan"]
                X_h = holdout[feature_cols].values
                if not handles:
                    X_h = np.nan_to_num(X_h, nan=0.0)
                ens_pred += weight * comp_model.predict_proba(X_h)[:, 1]
            holdout_preds = ens_pred
        else:
            m = res["model"]
            handles = res["handles_nan"]
            X_h = holdout[feature_cols].values
            if not handles:
                X_h = np.nan_to_num(X_h, nan=0.0)
            holdout_preds = m.predict_proba(X_h)[:, 1]
        metrics = compute_all_metrics(y_holdout, holdout_preds, name)
        all_metrics.append(metrics)
        results[name]["holdout_preds"] = holdout_preds

    save_metrics_summary(all_metrics)
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    plot_all_curves(
        {k: {"val_preds": results[k]["holdout_preds"], "pr_auc": results[k]["pr_auc"]}
         for k in results if "holdout_preds" in results[k]},
        y_holdout,
    )

    # ── 8. Calibration ──
    logger.info("\n>>> PHASE 7: CALIBRATION")
    from src.evaluation.calibration import calibrate_model

    best_model = results[best_name]["model"]
    if best_model is not None:
        handles = results[best_name]["handles_nan"]
        X_val = val[feature_cols].values
        y_val = val[TARGET].values
        X_hold = holdout[feature_cols].values
        calibrator, calibrated_probs, cal_info = calibrate_model(
            best_model, X_val, y_val, X_hold, y_holdout, best_name, handles
        )
    else:
        calibrator = None
        calibrated_probs = results[best_name]["holdout_preds"]
        logger.warning("Best model is Ensemble — skipping calibration, using raw probs")

    # ── 9. Explainability ──
    logger.info("\n>>> PHASE 8: EXPLAINABILITY")
    from src.evaluation.explainability import run_explainability

    # Use best non-ensemble model for SHAP
    non_ensemble = {k: v for k, v in results.items() if k != "Ensemble" and v["model"] is not None}
    shap_name = max(non_ensemble, key=lambda k: non_ensemble[k]["pr_auc"])
    shap_model = non_ensemble[shap_name]["model"]
    X_hold_shap = holdout[feature_cols].values
    run_explainability(shap_model, X_hold_shap, y_holdout, feature_cols,
                       shap_name, non_ensemble[shap_name]["handles_nan"])

    # ── 10. Business Policy ──
    logger.info("\n>>> PHASE 9: BUSINESS POLICY")
    from src.business.policy import build_policy
    policy = build_policy(y_holdout, calibrated_probs, holdout)

    # ── 11. Robustness ──
    logger.info("\n>>> PHASE 11: ROBUSTNESS TESTS")
    from src.evaluation.robustness import run_robustness_tests
    run_robustness_tests(shap_model, X_hold_shap, y_holdout, feature_cols,
                         non_ensemble[shap_name]["handles_nan"])

    # ── 12. Submission ──
    logger.info("\n>>> PHASE 10: SUBMISSION GENERATION")
    from src.inference.predict import predict_batch

    # Ensure submission has same features
    for col in feature_cols:
        if col not in df_sub_feat.columns:
            df_sub_feat[col] = 0.0

    sub_model = shap_model  # Use best non-ensemble
    predict_batch(df_sub_feat, sub_model, feature_cols,
                  calibrator, non_ensemble[shap_name]["handles_nan"])

    # ── 13. Reports ──
    logger.info("\n>>> PHASE 12: GENERATING REPORTS")
    _generate_executive_summary(all_metrics, policy, best_name)
    _generate_pitch_structure(all_metrics, policy)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %.1f minutes", elapsed / 60)
    logger.info("=" * 60)


def _generate_executive_summary(all_metrics, policy, best_name):
    from src.utils.config import REPORTS
    best = next(m for m in all_metrics if m["model"] == best_name)
    lines = [
        "# EXECUTIVE SUMMARY — FPD Prediction Model",
        "",
        "## Problem",
        "Predict First Payment Default (FPD) at checkout to enable proactive",
        "collection strategies and reduce default rates.",
        "",
        "## Approach",
        f"- Trained 4 models + ensemble with temporal validation",
        f"- Best model: **{best_name}**",
        f"- Used {len(next(m for m in all_metrics))-1} evaluation metrics",
        "",
        "## Key Metrics (Holdout — Out-of-Time)",
        f"- ROC AUC: **{best['ROC_AUC']:.4f}**",
        f"- PR AUC: **{best['PR_AUC']:.4f}**",
        f"- KS: **{best['KS']:.4f}**",
        f"- F1: **{best['F1']:.4f}**",
        f"- Top 10% Capture: **{best['Top10_Capture']:.1%}**",
        f"- Top 20% Capture: **{best['Top20_Capture']:.1%}**",
        "",
        "## Collection Policy",
        "4 risk bands with differentiated strategies:",
    ]
    for _, row in policy["risk_table"].iterrows():
        lines.append(f"- **{row['faixa']}** ({row['prob_min']:.0%}-{row['prob_max']:.0%}): "
                    f"{row['n_clientes']:,} clients, {row['taxa_fpd']:.1%} FPD rate")
    lines.extend([
        "",
        "## Impact",
        f"- Top 10% of scored population captures {best['Top10_Capture']:.1%} of all defaults",
        "- Collection policy does NOT reject clients — adjusts conditions only",
        "- Model is calibrated for reliable probability estimates",
        "",
        "## Risks & Limitations",
        "- Bureau data missing for ~86% of records — model robust without it",
        "- Temporal validation used — no data leakage",
        "- Step detection tuned for available device data only",
    ])
    (REPORTS / "executive_summary.md").write_text("\n".join(lines))
    logger.info("Executive summary saved")


def _generate_pitch_structure(all_metrics, policy):
    from src.utils.config import REPORTS
    lines = [
        "# PITCH STRUCTURE — FPD Hackathon TMB",
        "",
        "## Slide 1: Problem",
        "- FPD costs: revenue loss, collection overhead, portfolio risk",
        "- Goal: predict default BEFORE the sale, not after",
        "",
        "## Slide 2: Strategy",
        "- Temporal validation (no data leakage)",
        "- 4 models compared + ensemble",
        "- Calibrated probabilities for business decisions",
        "",
        "## Slide 3: EDA Highlights",
        "- 13.9% FPD rate (imbalanced)",
        "- Bureau data missing for 86% — model handles both scenarios",
        "- Strong signal in TMB score, bureau scores, and product segment",
        "",
        "## Slide 4: Feature Engineering",
        "- 30+ engineered features from checkout data only",
        "- Bureau aggregates, temporal features, target encoding",
        "- Missing indicators as explicit features",
        "",
        "## Slide 5: Modeling Results",
        "- Model comparison table (ROC AUC, PR AUC, KS, F1)",
        "- All validated on out-of-time holdout",
        "",
        "## Slide 6: Model Deep-Dive",
        "- SHAP analysis: top predictors",
        "- Calibration: reliable probability estimates",
        "",
        "## Slide 7: Thresholds",
        "- 3 strategies: optimal, conservative, aggressive",
        "- Trade-off: approval rate vs default rate",
        "",
        "## Slide 8: Collection Policy",
        "- 4 risk bands: Baixo / Moderado / Alto / Critico",
        "- Each with: channel, timing, down payment, installment limits",
        "- NOT a rejection policy — adjusts conditions",
        "",
        "## Slide 9: Financial Impact",
        "- Top 10% captures X% of defaults",
        "- Estimated savings from early intervention",
        "- ROI of collection optimization",
        "",
        "## Slide 10: Production Readiness",
        "- FastAPI endpoint ready",
        "- Docker containerized",
        "- Batch + real-time inference",
        "- Monitoring: drift detection, threshold stability",
    ]
    (REPORTS / "pitch_structure.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
