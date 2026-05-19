import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.config import FIGURES, TARGET, BUREAU_SCORE_COLS, SEED

logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


def run_eda(df: pd.DataFrame) -> None:
    logger.info("Starting EDA — generating plots to %s", FIGURES)
    _target_distribution(df)
    _missing_values(df)
    _correlation_matrix(df)
    _fpd_by_category(df, "segmento")
    _fpd_by_category(df, "modalidade")
    _fpd_by_estado(df)
    _fpd_by_risk_score(df)
    _bureau_distributions(df)
    _financiado_distribution(df)
    _parcelas_distribution(df)
    _temporal_analysis(df)
    _age_distribution(df)
    _bureau_availability(df)
    logger.info("EDA complete — %d plots generated", len(list(FIGURES.glob("*.png"))))


def _target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET].value_counts()
    bars = ax.bar(["Adimplente (0)", "Inadimplente (1)"], [counts.get(0, 0), counts.get(1, 0)],
                  color=["#2ecc71", "#e74c3c"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", fontsize=12)
    total = len(df)
    pct_pos = counts.get(1, 0) / total * 100
    ax.set_title(f"Target Distribution — FPD Rate: {pct_pos:.1f}%", fontsize=14)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_target_distribution.png", dpi=150)
    plt.close(fig)


def _missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, len(missing) * 0.3)))
    missing.plot.barh(ax=ax, color="#3498db")
    ax.set_xlabel("Missing Fraction")
    ax.set_title("Missing Values by Feature")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "02_missing_values.png", dpi=150)
    plt.close(fig)


def _correlation_matrix(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_correlation_matrix.png", dpi=150)
    plt.close(fig)


def _fpd_by_category(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    grouped = df.groupby(col)[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
    grouped = grouped[grouped["count"] >= 30]
    if len(grouped) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, len(grouped) * 0.4)))
    colors = plt.cm.RdYlGn_r(grouped["mean"] / grouped["mean"].max())
    grouped["mean"].plot.barh(ax=ax, color=colors)
    ax.set_xlabel("FPD Rate")
    ax.set_title(f"FPD Rate by {col}")
    ax.invert_yaxis()
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(row["mean"] + 0.005, i, f'{row["mean"]:.1%} (n={int(row["count"])})', va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"04_fpd_by_{col}.png", dpi=150)
    plt.close(fig)


def _fpd_by_estado(df: pd.DataFrame) -> None:
    if "endereco_estado" not in df.columns:
        return
    estado = df.copy()
    estado["endereco_estado"] = estado["endereco_estado"].astype(str).str.strip().str.upper()
    estado = estado[estado["endereco_estado"].str.len() == 2]
    _fpd_by_category(estado, "endereco_estado")
    fig_path = FIGURES / "04_fpd_by_endereco_estado.png"
    if fig_path.exists():
        fig_path.rename(FIGURES / "05_fpd_by_estado.png")


def _fpd_by_risk_score(df: pd.DataFrame) -> None:
    if "categoria_risco_score" not in df.columns:
        return
    _fpd_by_category(df, "categoria_risco_score")
    src = FIGURES / "04_fpd_by_categoria_risco_score.png"
    if src.exists():
        src.rename(FIGURES / "06_fpd_by_risk_category.png")


def _bureau_distributions(df: pd.DataFrame) -> None:
    available = [c for c in BUREAU_SCORE_COLS if c in df.columns]
    if not available:
        return
    n = len(available)
    cols_grid = 4
    rows_grid = (n + cols_grid - 1) // cols_grid
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(20, 4 * rows_grid))
    axes = axes.flatten()
    for i, col in enumerate(available):
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = df[df[TARGET] == label][col].dropna()
            if len(subset) > 0:
                axes[i].hist(subset, bins=30, alpha=0.5, color=color, label=f"FPD={label}")
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Bureau Score Distributions by FPD", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_bureau_distributions.png", dpi=150)
    plt.close(fig)


def _financiado_distribution(df: pd.DataFrame) -> None:
    if "total_financiado" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        subset = df[df[TARGET] == label]["total_financiado"].dropna()
        ax.hist(subset, bins=50, alpha=0.5, color=color, label=f"FPD={label}", density=True)
    ax.set_xlabel("Total Financiado (R$)")
    ax.set_title("Total Financiado Distribution by FPD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "08_total_financiado.png", dpi=150)
    plt.close(fig)


def _parcelas_distribution(df: pd.DataFrame) -> None:
    if "quantidade_parcelas" not in df.columns:
        return
    grouped = df.groupby("quantidade_parcelas")[TARGET].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 20].sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(grouped.index.astype(str), grouped["mean"], color="#3498db")
    ax.set_xlabel("Quantidade Parcelas")
    ax.set_ylabel("FPD Rate")
    ax.set_title("FPD Rate by Number of Installments")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_parcelas_fpd.png", dpi=150)
    plt.close(fig)


def _temporal_analysis(df: pd.DataFrame) -> None:
    if "data_efetivacao" not in df.columns:
        return
    temp = df.dropna(subset=["data_efetivacao"]).copy()
    temp["year_month"] = temp["data_efetivacao"].dt.to_period("M")
    monthly = temp.groupby("year_month")[TARGET].agg(["mean", "count"])
    monthly = monthly[monthly["count"] >= 30]
    if len(monthly) == 0:
        return
    fig, ax1 = plt.subplots(figsize=(14, 5))
    x = range(len(monthly))
    ax1.bar(x, monthly["count"], color="#bdc3c7", alpha=0.7, label="Volume")
    ax1.set_ylabel("Volume", color="#7f8c8d")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["mean"], color="#e74c3c", marker="o", linewidth=2, label="FPD Rate")
    ax2.set_ylabel("FPD Rate", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in monthly.index], rotation=45, fontsize=8)
    ax1.set_title("Temporal Analysis: Volume & FPD Rate by Month")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_temporal_analysis.png", dpi=150)
    plt.close(fig)


def _age_distribution(df: pd.DataFrame) -> None:
    if "idade" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        subset = df[df[TARGET] == label]["idade"].dropna()
        subset = subset[(subset >= 16) & (subset <= 90)]
        if len(subset) > 0:
            ax.hist(subset, bins=40, alpha=0.5, color=color, label=f"FPD={label}", density=True)
    ax.set_xlabel("Age")
    ax.set_title("Age Distribution by FPD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "11_age_distribution.png", dpi=150)
    plt.close(fig)


def _bureau_availability(df: pd.DataFrame) -> None:
    available = [c for c in BUREAU_SCORE_COLS if c in df.columns]
    if not available:
        return
    df_copy = df.copy()
    df_copy["bureau_available"] = df_copy[available].notna().any(axis=1).astype(int)
    grouped = df_copy.groupby("bureau_available")[TARGET].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["No Bureau", "Has Bureau"], grouped["mean"].values, color=["#e67e22", "#2ecc71"])
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{row["mean"]:.1%}\n(n={int(row["count"])})', ha="center", fontsize=11)
    ax.set_ylabel("FPD Rate")
    ax.set_title("FPD Rate: Bureau Available vs Missing")
    fig.tight_layout()
    fig.savefig(FIGURES / "12_bureau_availability.png", dpi=150)
    plt.close(fig)
