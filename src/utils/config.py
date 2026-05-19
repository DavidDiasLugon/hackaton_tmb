import os
from pathlib import Path

SEED = 42
N_JOBS = -1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SUBMISSIONS = PROJECT_ROOT / "data" / "submissions"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
METRICS = OUTPUTS / "metrics"
EXPLAINABILITY = OUTPUTS / "explainability"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_RAW / "base-treinamento.xlsx"
SUBMISSION_FILE = DATA_RAW / "submissao.xlsx"
DICT_FILE = DATA_RAW / "data_dictionary.csv"

TARGET = "FPD"
ID_COL = "pedido_id"
DATE_COL = "data_efetivacao"

LEAKAGE_COLS = [
    "status_cobranca", "status_financeiro", "status_pedido",
    "saldo_vencido", "quantidade_parcelas_vencidas", "recebido",
    "primeiro_vencimento_em_atraso", "dias_em_atraso", "pdd",
    "saldo_vencido_com_juros", "total_pago_com_juros",
    "aguardando_pagamento_sem_juros", "vencidos_sem_juros_tmb",
    "recebido_sem_juros_tmb", "data_quitacao",
]

ID_PII_COLS = [
    "CPF", "documento", "documento2", "nome", "email",
    "telefone_ativo", "endereco_cidade", "endereco_cep",
]

USELESS_COLS = [
    "MENSAGEM_TIPO_REGISTRO", "pedido_pai_ob", "order_bump",
]

DROP_COLS = LEAKAGE_COLS + ID_PII_COLS + USELESS_COLS

BUREAU_SCORE_COLS = [
    "SCORE_HCP4", "SCORE_HCP5", "SCORE_HCC4", "SCORE_HCMV",
    "SCORE_HEST", "SCORE_HFI4", "SCORE_HFI5", "SCORE_HIPA",
    "SCORE_HIPN", "SCORE_HIRF", "SCORE_HRCP", "SCORE_HRM5",
    "SCORE_HSV4", "SCORE_HSV5", "SCORE_HVA4", "SCORE_HVA5",
    "SCORE_HPG5", "SCORE_HCR4", "SCORE_H5OR",
]

CATEGORICAL_COLS = [
    "segmento", "modalidade", "categoria_risco_score",
    "produtor", "lancamento", "endereco_estado",
]

OPTUNA_TRIALS = 50
CV_FOLDS = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RISK_BANDS = {
    "Baixo":    (0.00, 0.15),
    "Moderado": (0.15, 0.35),
    "Alto":     (0.35, 0.60),
    "Critico":  (0.60, 1.00),
}

for d in [DATA_PROCESSED, DATA_SUBMISSIONS, FIGURES, REPORTS, METRICS, EXPLAINABILITY, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
