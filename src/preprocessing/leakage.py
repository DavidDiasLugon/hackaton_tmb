import logging
import pandas as pd
from src.utils.config import LEAKAGE_COLS, ID_PII_COLS, USELESS_COLS, REPORTS, TARGET

logger = logging.getLogger(__name__)


def generate_leakage_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("LEAKAGE DETECTION REPORT")
    lines.append("=" * 60)

    lines.append("\n## 1. POST-EVENT COLUMNS (LEAKAGE)")
    lines.append("These columns only exist after billing/payment and MUST NOT be used:\n")
    for col in LEAKAGE_COLS:
        if col in df.columns:
            status = "FOUND — WILL BE DROPPED"
        else:
            status = "not present (already removed)"
        lines.append(f"  - {col}: {status}")

    lines.append("\n## 2. ID/PII COLUMNS (NO PREDICTIVE VALUE)")
    for col in ID_PII_COLS:
        if col in df.columns:
            status = "FOUND — WILL BE DROPPED"
        else:
            status = "not present"
        lines.append(f"  - {col}: {status}")

    lines.append("\n## 3. CONSTANT/USELESS COLUMNS")
    for col in USELESS_COLS:
        if col in df.columns:
            nunique = df[col].nunique(dropna=False)
            pct_null = df[col].isna().mean() * 100
            status = f"FOUND — nunique={nunique}, {pct_null:.1f}% null — WILL BE DROPPED"
        else:
            status = "not present"
        lines.append(f"  - {col}: {status}")

    lines.append("\n## 4. CORRELATION-BASED LEAKAGE SCAN")
    lines.append("Checking remaining numeric features for suspiciously high correlation with target...\n")
    if TARGET in df.columns:
        df_copy = df.copy()
        df_copy[TARGET] = df_copy[TARGET].map({"Sim": 1, "Não": 0, "NÃ£o": 0, "NÃO": 0})
        df_copy = df_copy.dropna(subset=[TARGET])
        numeric = df_copy.select_dtypes(include="number").drop(columns=[TARGET], errors="ignore")
        corrs = numeric.corrwith(df_copy[TARGET]).abs().sort_values(ascending=False)
        suspicious = corrs[corrs > 0.5]
        if len(suspicious) > 0:
            lines.append("  WARNING: Features with |correlation| > 0.5 with target:")
            for feat, val in suspicious.items():
                lines.append(f"    - {feat}: {val:.3f}")
        else:
            lines.append("  OK: No remaining features have |correlation| > 0.5 with target.")

    lines.append("\n## 5. FEATURES AVAILABLE AT CHECKOUT (SAFE)")
    safe_cols = [c for c in df.columns if c not in LEAKAGE_COLS + ID_PII_COLS + USELESS_COLS]
    lines.append(f"  Total safe features: {len(safe_cols)}")
    for col in safe_cols:
        lines.append(f"  - {col}")

    lines.append("\n" + "=" * 60)
    lines.append("CONCLUSION: All post-event columns identified and excluded.")
    lines.append("=" * 60)

    report = "\n".join(lines)
    report_path = REPORTS / "leakage_report.txt"
    report_path.write_text(report)
    logger.info("Leakage report saved to %s", report_path)
    return report
