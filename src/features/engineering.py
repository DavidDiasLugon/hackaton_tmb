import logging
import pandas as pd
import numpy as np
from src.utils.config import (
    TARGET, ID_COL, DATE_COL, BUREAU_SCORE_COLS,
    CATEGORICAL_COLS, SEED, DATA_PROCESSED,
)

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, is_train: bool = True,
                   target_encoders: dict | None = None,
                   freq_encoders: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    logger.info("Building features — is_train=%s, shape=%s", is_train, df.shape)

    df = df.copy()
    df = _temporal_features(df)
    df = _age_features(df)
    df = _estado_cleaning(df)
    df = _bureau_aggregates(df)
    df = _financial_ratios(df)
    df = _hi_flags(df)
    df = _missing_indicators(df)

    if target_encoders is None:
        target_encoders = {}
    if freq_encoders is None:
        freq_encoders = {}

    df, target_encoders, freq_encoders = _encode_categoricals(
        df, is_train, target_encoders, freq_encoders
    )

    df = _score_tmb(df)
    df = _drop_raw_categoricals(df)
    df = _final_cleanup(df)

    logger.info("Features built — final shape: %s", df.shape)
    return df, target_encoders, freq_encoders


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL in df.columns:
        dt = df[DATE_COL]
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["hour"] = dt.dt.hour
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    if DATE_COL in df.columns and "Vencimento" in df.columns:
        venc = pd.to_datetime(df["Vencimento"], errors="coerce")
        df["days_until_payment"] = (venc - df[DATE_COL]).dt.days

    for col in [DATE_COL, "Vencimento"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def _age_features(df: pd.DataFrame) -> pd.DataFrame:
    if "nascimento" in df.columns:
        df = df.drop(columns=["nascimento"])
    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
        df["idade"] = df["idade"].clip(16, 90)
    return df


def _estado_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if "endereco_estado" not in df.columns:
        return df
    df["endereco_estado"] = df["endereco_estado"].astype(str).str.strip().str.upper()
    br_states = {
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
        "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
        "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
    }
    df.loc[~df["endereco_estado"].isin(br_states), "endereco_estado"] = "OUTROS"
    return df


def _bureau_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in BUREAU_SCORE_COLS if c in df.columns]
    if not available:
        return df
    bureau_data = df[available]
    df["bureau_available"] = bureau_data.notna().any(axis=1).astype(int)
    df["bureau_score_count"] = bureau_data.notna().sum(axis=1)
    df["bureau_mean_score"] = bureau_data.mean(axis=1)
    df["bureau_std_score"] = bureau_data.std(axis=1)
    df["bureau_min_score"] = bureau_data.min(axis=1)
    df["bureau_max_score"] = bureau_data.max(axis=1)
    return df


def _financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if "total_financiado" in df.columns and "quantidade_parcelas" in df.columns:
        df["valor_por_parcela"] = df["total_financiado"] / df["quantidade_parcelas"].replace(0, np.nan)
    return df


def _hi_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["flag_hi01_available"] = 0
    df["flag_hi02_available"] = 0
    if "HI01_PROB" in df.columns:
        df["flag_hi01_available"] = df["HI01_PROB"].notna().astype(int)
    if "HI02_PROB" in df.columns:
        df["flag_hi02_available"] = df["HI02_PROB"].notna().astype(int)

    for col in ["HI01_PROB", "HI02_PROB"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["HI01_CONCEITO", "HI02_CONCEITO"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "score" in df.columns:
        df["flag_score_missing"] = df["score"].isna().astype(int)

    available_bureau = [c for c in BUREAU_SCORE_COLS if c in df.columns]
    if available_bureau:
        df["flag_bureau_missing"] = df[available_bureau].isna().all(axis=1).astype(int)

    return df


def _encode_categoricals(df: pd.DataFrame, is_train: bool,
                         target_encoders: dict,
                         freq_encoders: dict) -> tuple[pd.DataFrame, dict, dict]:
    te_cols = ["segmento", "modalidade", "categoria_risco_score"]
    fe_cols = ["produtor", "lancamento"]

    for col in te_cols:
        if col not in df.columns:
            continue
        enc_name = f"{col}_te"
        if is_train:
            global_mean = df[TARGET].mean()
            mapping = df.groupby(col)[TARGET].mean().to_dict()
            target_encoders[col] = {"mapping": mapping, "default": global_mean}
        encoder = target_encoders.get(col, {"mapping": {}, "default": 0.5})
        df[enc_name] = df[col].map(encoder["mapping"]).fillna(encoder["default"])

    for col in fe_cols:
        if col not in df.columns:
            continue
        enc_name = f"{col}_freq"
        if is_train:
            mapping = df[col].value_counts(normalize=True).to_dict()
            freq_encoders[col] = mapping
        mapping = freq_encoders.get(col, {})
        df[enc_name] = df[col].map(mapping).fillna(0.0)

    if "endereco_estado" in df.columns:
        enc_name = "estado_te"
        if is_train:
            global_mean = df[TARGET].mean()
            mapping = df.groupby("endereco_estado")[TARGET].mean().to_dict()
            target_encoders["endereco_estado"] = {"mapping": mapping, "default": global_mean}
        encoder = target_encoders.get("endereco_estado", {"mapping": {}, "default": 0.5})
        df[enc_name] = df["endereco_estado"].map(encoder["mapping"]).fillna(encoder["default"])

    return df, target_encoders, freq_encoders


def _score_tmb(df: pd.DataFrame) -> pd.DataFrame:
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def _drop_raw_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in CATEGORICAL_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "datetime64"]).columns.tolist()
    if ID_COL in obj_cols:
        obj_cols.remove(ID_COL)
    if TARGET in obj_cols:
        obj_cols.remove(TARGET)
    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    obj_remaining = df.select_dtypes(include=["object", "datetime64"]).columns.tolist()
    drop_remaining = [c for c in obj_remaining if c not in [ID_COL, TARGET]]
    if drop_remaining:
        logger.warning("Dropping remaining non-numeric columns: %s", drop_remaining)
        df = df.drop(columns=drop_remaining)
    return df
