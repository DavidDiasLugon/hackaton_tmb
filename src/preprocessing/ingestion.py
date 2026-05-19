import logging
import pandas as pd
import numpy as np
from src.utils.config import TRAIN_FILE, SUBMISSION_FILE, TARGET, DROP_COLS, ID_COL, DATE_COL

logger = logging.getLogger(__name__)


def load_train() -> pd.DataFrame:
    logger.info("Loading training data from %s", TRAIN_FILE)
    df = pd.read_excel(TRAIN_FILE)
    logger.info("Training shape: %s", df.shape)

    df[TARGET] = df[TARGET].map({"Sim": 1, "Não": 0, "NÃ£o": 0, "NÃO": 0})
    if df[TARGET].isna().any():
        n = df[TARGET].isna().sum()
        logger.warning("Dropping %d rows with unmapped FPD values", n)
        df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    df = _parse_dates(df)
    df = _clean_columns(df)
    return df


def load_submission() -> pd.DataFrame:
    logger.info("Loading submission data from %s", SUBMISSION_FILE)
    df = pd.read_excel(SUBMISSION_FILE)
    logger.info("Submission shape: %s", df.shape)
    df = _parse_dates(df)
    df = _clean_columns(df)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in [DATE_COL, "Vencimento", "nascimento"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if DATE_COL in df.columns:
        cutoff = pd.Timestamp("2020-01-01")
        bad = df[DATE_COL] < cutoff
        if bad.any():
            logger.warning("Replacing %d rows with data_efetivacao < 2020 (corrupt dates)", bad.sum())
            df.loc[bad, DATE_COL] = pd.NaT
    return df


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    logger.info("Dropping %d columns: %s", len(existing_drops), existing_drops)
    df = df.drop(columns=existing_drops)

    for col in ["score", "total_financiado", "idade"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "quantidade_parcelas" in df.columns:
        df["quantidade_parcelas"] = pd.to_numeric(df["quantidade_parcelas"], errors="coerce")

    return df
