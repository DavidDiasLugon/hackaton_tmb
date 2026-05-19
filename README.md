# Hackathon FPD — TMB

End-to-end ML solution for predicting **First Payment Default (FPD)** at checkout time.

## Architecture

```
hackathon-fpd/
├── data/
│   ├── raw/                  # Original Excel files
│   ├── processed/            # Intermediate datasets
│   └── submissions/          # Final submission.csv
├── models/                   # Serialized models + encoders
├── notebooks/                # Exploratory notebooks
├── src/
│   ├── preprocessing/
│   │   ├── ingestion.py      # Data loading + cleaning
│   │   ├── leakage.py        # Automated leakage detection
│   │   └── eda.py            # 13+ auto-generated EDA plots
│   ├── features/
│   │   └── engineering.py    # 30+ engineered features
│   ├── models/
│   │   ├── validation.py     # Temporal split + TimeSeriesCV
│   │   └── trainer.py        # 4 models + Optuna tuning + ensemble
│   ├── evaluation/
│   │   ├── metrics.py        # ROC/PR/KS/Gain/Lift/Decile
│   │   ├── calibration.py    # Isotonic + Platt scaling
│   │   ├── explainability.py # SHAP + permutation importance
│   │   └── robustness.py     # Bureau-missing, drift, threshold tests
│   ├── inference/
│   │   ├── predict.py        # Batch + single prediction
│   │   └── api.py            # FastAPI endpoint
│   ├── business/
│   │   └── policy.py         # Risk bands + collection strategy
│   └── utils/
│       └── config.py         # Centralized configuration
├── outputs/
│   ├── figures/              # All auto-generated plots (28+)
│   ├── reports/              # Executive summary, policy, leakage
│   ├── metrics/              # Model comparison CSV, decile table
│   └── explainability/       # Feature importance, SHAP, local examples
├── main.py                   # Single entry point
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM requires OpenMP:
```bash
brew install libomp
```

## Quick Start

### Run full pipeline
```bash
python main.py
```

This executes all 12 phases sequentially:
1. Data ingestion + leakage detection
2. EDA (13+ auto-generated plots)
3. Temporal split (70/15/15 chronological)
4. Feature engineering (30+ features, target encoding fitted on train only)
5. Model training: Logistic Regression, LightGBM, XGBoost, CatBoost + Ensemble
6. Hyperparameter tuning: Optuna 50 trials/model, optimizing PR-AUC
7. Evaluation: ROC/PR/KS curves, gain/lift charts, decile analysis
8. Calibration: Isotonic + Platt scaling comparison
9. SHAP explainability + permutation importance
10. Business policy: 4 risk bands with collection strategies
11. Robustness tests: bureau-missing, extreme-missing, threshold stability
12. Submission generation + reports

### Start API server
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t fpd-model .
docker run -p 8000:8000 fpd-model
```

## Leakage Prevention

The most critical constraint: **no post-event features**.

15 columns are automatically detected and excluded:
- `status_cobranca`, `status_financeiro`, `status_pedido` (post-sale outcomes)
- `saldo_vencido`, `dias_em_atraso`, `recebido`, etc. (post-billing data)
- `data_quitacao` (payoff date)

Only checkout-time features are used: product info, demographics, bureau scores, TMB score, temporal features.

## Validation Strategy

- **No random split** — temporal split only
- Train: oldest 70% by `data_efetivacao`
- Validation: next 15% (Optuna tuning + calibration)
- Holdout: most recent 15% (out-of-time evaluation)
- Target encoding fitted on train only to prevent CV leakage

## Models

| Model | Tuning | Imbalance Handling |
|-------|--------|--------------------|
| Logistic Regression | C=0.1 | class_weight='balanced' |
| LightGBM | Optuna 50 trials | scale_pos_weight |
| XGBoost | Optuna 50 trials | scale_pos_weight |
| CatBoost | Optuna 50 trials | auto_class_weights |
| Ensemble | Weighted average | From component models |

## Collection Policy

4 risk bands (does NOT reject clients — adjusts conditions):

| Band | Prob Range | Action | Down Payment | Max Installments |
|------|-----------|--------|-------------|-----------------|
| Baixo | 0-15% | Standard | 0% | No limit |
| Moderado | 15-35% | SMS D-3 | 10% | 12x |
| Alto | 35-60% | Phone D-1 | 20% | 6x |
| Critico | 60-100% | Immediate call | 30% | 3x |

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"score": 450, "total_financiado": 1500, "quantidade_parcelas": 12}}'
```

Response:
```json
{
  "prob_fpd": 0.23,
  "classe_fpd": 0,
  "faixa_risco": "Moderado",
  "acao_cobranca": "SMS D-3 preventivo + e-mail lembrete",
  "parcelamento_maximo": "12x",
  "entrada_minima": "10%",
  "juros": "Taxa padrão"
}
```

## Key Technical Decisions

1. **Temporal split before feature engineering** — target encoding fitted only on train split
2. **Bureau aggregates** — mean/std/min/max/count across 19 bureau scores as meta-features
3. **Missing indicators** — explicit flags for bureau/score/HI absence (high predictive value when ~86% of records lack bureau data)
4. **HI01 vs HI02** — kept as separate features with availability flags (training has HI01, submission has HI02)
5. **Calibration** — Isotonic vs Platt compared on holdout via Brier score
6. **PR-AUC optimization** — better than ROC-AUC for imbalanced problems (13.9% positive rate)

## Risks & Limitations

- Bureau data missing for ~86% of records — model designed to work without it
- Temporal validation guards against leakage but training date range may include corrupted entries
- FALLBACK PDR tuning validated on limited device set
- Collection policy thresholds should be re-validated with business stakeholders
- Model should be retrained quarterly to account for population drift

## Reproduction

```bash
# 1. Place data files in data/raw/
cp base-treinamento.xlsx data/raw/
cp submissao.xlsx data/raw/

# 2. Run pipeline
python main.py

# 3. Check outputs
ls outputs/figures/      # 28+ plots
ls outputs/reports/      # Executive summary, policy, leakage report
ls data/submissions/     # submission.csv
ls models/               # Serialized model + calibrator
```

Seeds are fixed (42) for full reproducibility.
