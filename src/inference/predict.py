import logging
import numpy as np
import pandas as pd
import joblib
from src.utils.config import (
    MODELS_DIR, DATA_SUBMISSIONS, RISK_BANDS, TARGET, ID_COL,
)

logger = logging.getLogger(__name__)


def predict_single(record: dict, model=None, feature_cols=None,
                   calibrator=None) -> dict:
    if model is None:
        model = joblib.load(MODELS_DIR / "best_model.joblib")
    if feature_cols is None:
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    if calibrator is None:
        try:
            calibrator = joblib.load(MODELS_DIR / "calibrator.joblib")
        except FileNotFoundError:
            calibrator = None

    X = np.array([[record.get(f, 0.0) for f in feature_cols]])
    X = np.nan_to_num(X, nan=0.0)

    raw_prob = float(model.predict_proba(X)[:, 1][0])
    if calibrator is not None:
        prob = float(calibrator.predict(np.array([raw_prob]))[0])
    else:
        prob = raw_prob

    classe = int(prob >= 0.5)
    faixa = _get_risk_band(prob)
    strategy = _get_strategy(faixa)

    return {
        "prob_fpd": round(prob, 4),
        "classe_fpd": classe,
        "faixa_risco": faixa,
        **strategy,
    }


def predict_batch(df: pd.DataFrame, model, feature_cols: list[str],
                  calibrator=None, handles_nan: bool = True) -> pd.DataFrame:
    logger.info("Batch prediction on %d records", len(df))

    X = df[feature_cols].values
    if not handles_nan:
        X = np.nan_to_num(X, nan=0.0)

    raw_probs = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict(raw_probs)
    else:
        probs = raw_probs

    result = pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "prob_fpd": np.round(probs, 4),
        "classe_fpd": (probs >= 0.5).astype(int),
        "faixa_risco": [_get_risk_band(p) for p in probs],
    })

    output_path = DATA_SUBMISSIONS / "submission.csv"
    result.to_csv(output_path, index=False)
    logger.info("Submission saved to %s — shape: %s", output_path, result.shape)
    logger.info("Risk band distribution:\n%s", result["faixa_risco"].value_counts().to_string())

    return result


def _get_risk_band(prob: float) -> str:
    for band, (lo, hi) in RISK_BANDS.items():
        if lo <= prob < hi:
            return band
    return "Critico"


def _get_strategy(faixa: str) -> dict:
    strategies = {
        "Baixo": {
            "acao_cobranca": "Padrão — sem ação especial",
            "parcelamento_maximo": "Sem restrição",
            "entrada_minima": "0%",
            "juros": "Taxa padrão",
        },
        "Moderado": {
            "acao_cobranca": "SMS D-3 preventivo + e-mail lembrete",
            "parcelamento_maximo": "12x",
            "entrada_minima": "10%",
            "juros": "Taxa padrão",
        },
        "Alto": {
            "acao_cobranca": "Ligação D-1 + cobrança ativa D+1",
            "parcelamento_maximo": "6x",
            "entrada_minima": "20%",
            "juros": "+2% a.m.",
        },
        "Critico": {
            "acao_cobranca": "Ligação imediata + régua intensiva",
            "parcelamento_maximo": "3x",
            "entrada_minima": "30%",
            "juros": "+4% a.m.",
        },
    }
    return strategies.get(faixa, strategies["Critico"])
