import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils.config import MODELS_DIR, RISK_BANDS
from src.inference.predict import _get_risk_band, _get_strategy

app = FastAPI(title="FPD Prediction API", version="1.0.0")

model = None
feature_cols = None
calibrator = None


@app.on_event("startup")
def load_artifacts():
    global model, feature_cols, calibrator
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    try:
        calibrator = joblib.load(MODELS_DIR / "calibrator.joblib")
    except FileNotFoundError:
        calibrator = None


class PredictionRequest(BaseModel):
    features: dict


class PredictionResponse(BaseModel):
    prob_fpd: float
    classe_fpd: int
    faixa_risco: str
    acao_cobranca: str
    parcelamento_maximo: str
    entrada_minima: str
    juros: str


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array([[request.features.get(f, 0.0) for f in feature_cols]])
    X = np.nan_to_num(X, nan=0.0)
    raw_prob = float(model.predict_proba(X)[:, 1][0])
    if calibrator is not None:
        prob = float(calibrator.predict(np.array([raw_prob]))[0])
    else:
        prob = raw_prob
    faixa = _get_risk_band(prob)
    strategy = _get_strategy(faixa)
    return PredictionResponse(
        prob_fpd=round(prob, 4),
        classe_fpd=int(prob >= 0.5),
        faixa_risco=faixa,
        **strategy,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
