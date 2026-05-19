import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from src.utils.config import FIGURES, REPORTS, RISK_BANDS, TARGET

logger = logging.getLogger(__name__)


def build_policy(y_true: np.ndarray, y_pred: np.ndarray,
                 df_holdout: pd.DataFrame = None) -> dict:
    thresholds = _compute_thresholds(y_true, y_pred)
    risk_table = _build_risk_table(y_true, y_pred)
    capture_analysis = _capture_analysis(y_true, y_pred)
    _plot_tradeoff(y_true, y_pred)

    if df_holdout is not None:
        _segment_performance(df_holdout, y_pred)

    policy = {
        "thresholds": thresholds,
        "risk_table": risk_table,
        "capture": capture_analysis,
    }

    _save_policy_report(policy)
    return policy


def _compute_thresholds(y_true, y_pred) -> dict:
    prec, rec, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = float(thresholds[optimal_idx])

    # Conservative: maximize recall where precision >= 30%
    mask = prec >= 0.30
    if mask.any():
        conservative_idx = np.argmax(rec[mask])
        conservative_thresh = float(thresholds[np.where(mask)[0][conservative_idx]])
    else:
        conservative_thresh = optimal_thresh * 0.7

    # Aggressive: maximize precision where recall >= 50%
    mask = rec >= 0.50
    if mask.any():
        aggressive_idx = np.argmax(prec[mask])
        aggressive_thresh = float(thresholds[np.where(mask)[0][aggressive_idx]])
    else:
        aggressive_thresh = optimal_thresh * 1.3

    thresholds_dict = {
        "optimal": {"value": optimal_thresh, "rationale": "Maximizes F1 score"},
        "conservative": {"value": conservative_thresh, "rationale": "Maximizes recall at precision >= 30%"},
        "aggressive": {"value": aggressive_thresh, "rationale": "Maximizes precision at recall >= 50%"},
    }

    logger.info("Thresholds — Optimal: %.3f, Conservative: %.3f, Aggressive: %.3f",
                optimal_thresh, conservative_thresh, aggressive_thresh)
    return thresholds_dict


def _build_risk_table(y_true, y_pred) -> pd.DataFrame:
    records = []
    for band_name, (lo, hi) in RISK_BANDS.items():
        mask = (y_pred >= lo) & (y_pred < hi)
        n = mask.sum()
        if n == 0:
            continue
        n_fpd = y_true[mask].sum()
        fpd_rate = y_true[mask].mean()

        strategies = {
            "Baixo": {
                "cobranca": "Padrão — sem ação especial",
                "canal": "E-mail automático",
                "timing": "D+1 após vencimento",
                "entrada_minima": "0%",
                "juros": "Taxa padrão",
                "parcelas_max": "Sem restrição",
                "prioridade": "Baixa",
            },
            "Moderado": {
                "cobranca": "SMS D-3 + e-mail D-1",
                "canal": "SMS + E-mail",
                "timing": "D-3 preventivo",
                "entrada_minima": "10%",
                "juros": "Taxa padrão",
                "parcelas_max": "12x",
                "prioridade": "Média",
            },
            "Alto": {
                "cobranca": "Ligação D-1 + SMS D-3 + cobrança ativa D+1",
                "canal": "Telefone + SMS + E-mail",
                "timing": "D-3 a D+1",
                "entrada_minima": "20%",
                "juros": "+2% a.m.",
                "parcelas_max": "6x",
                "prioridade": "Alta",
            },
            "Critico": {
                "cobranca": "Ligação imediata + régua intensiva D+0 a D+7",
                "canal": "Telefone (prioridade) + SMS + WhatsApp + E-mail",
                "timing": "Imediato (D+0)",
                "entrada_minima": "30%",
                "juros": "+4% a.m.",
                "parcelas_max": "3x",
                "prioridade": "Urgente",
            },
        }

        record = {
            "faixa": band_name,
            "prob_min": lo,
            "prob_max": hi,
            "n_clientes": int(n),
            "n_fpd": int(n_fpd),
            "taxa_fpd": fpd_rate,
            "pct_total": n / len(y_true),
        }
        record.update(strategies.get(band_name, {}))
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(REPORTS / "risk_bands.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
    axes[0].bar(df["faixa"], df["n_clientes"], color=colors[:len(df)])
    axes[0].set_title("Clients per Risk Band")
    axes[0].set_ylabel("Count")
    for i, row in df.iterrows():
        axes[0].text(i, row["n_clientes"] + 50, f'{row["pct_total"]:.1%}', ha="center")

    axes[1].bar(df["faixa"], df["taxa_fpd"], color=colors[:len(df)])
    axes[1].set_title("FPD Rate per Risk Band")
    axes[1].set_ylabel("FPD Rate")
    for i, row in df.iterrows():
        axes[1].text(i, row["taxa_fpd"] + 0.005, f'{row["taxa_fpd"]:.1%}', ha="center")

    fig.tight_layout()
    fig.savefig(FIGURES / "26_risk_bands.png", dpi=150)
    plt.close(fig)

    return df


def _capture_analysis(y_true, y_pred) -> dict:
    captures = {}
    for pct in [0.05, 0.10, 0.20, 0.30]:
        n_top = max(1, int(len(y_true) * pct))
        order = np.argsort(-y_pred)
        captured = y_true[order[:n_top]].sum()
        total_bad = max(y_true.sum(), 1)
        good_blocked = (1 - y_true[order[:n_top]]).sum()
        captures[f"top_{int(pct*100)}pct"] = {
            "fpd_captured": float(captured / total_bad),
            "good_blocked": int(good_blocked),
            "good_blocked_pct": float(good_blocked / max((y_true == 0).sum(), 1)),
        }
        logger.info("Top %d%%: captures %.1f%% FPD, blocks %d good clients (%.1f%%)",
                    int(pct * 100), captured / total_bad * 100,
                    good_blocked, good_blocked / max((y_true == 0).sum(), 1) * 100)

    return captures


def _plot_tradeoff(y_true, y_pred):
    thresholds = np.linspace(0.01, 0.99, 200)
    approvals = []
    fpd_rates = []
    for t in thresholds:
        approved = y_pred < t
        if approved.sum() == 0:
            continue
        approvals.append(approved.mean())
        fpd_rates.append(y_true[approved].mean())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(approvals, fpd_rates, color="#3498db", linewidth=2)
    ax.set_xlabel("Approval Rate")
    ax.set_ylabel("FPD Rate Among Approved")
    ax.set_title("Approval vs Default Trade-off")
    ax.axhline(y_true.mean(), color="red", linestyle="--", alpha=0.5,
               label=f"Baseline FPD: {y_true.mean():.1%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "27_approval_tradeoff.png", dpi=150)
    plt.close(fig)


def _segment_performance(df_holdout, y_pred):
    if "segmento_te" in df_holdout.columns and TARGET in df_holdout.columns:
        pass
    # Segment analysis is done in EDA; here we focus on risk-band analysis
    pass


def _save_policy_report(policy):
    lines = []
    lines.append("=" * 60)
    lines.append("COLLECTION POLICY REPORT")
    lines.append("=" * 60)

    lines.append("\n## THRESHOLDS")
    for name, info in policy["thresholds"].items():
        lines.append(f"  {name.upper()}: {info['value']:.4f} — {info['rationale']}")

    lines.append("\n## RISK BANDS")
    for _, row in policy["risk_table"].iterrows():
        lines.append(f"\n  [{row['faixa']}] ({row['prob_min']:.0%}-{row['prob_max']:.0%})")
        lines.append(f"    Clients: {row['n_clientes']:,} ({row['pct_total']:.1%})")
        lines.append(f"    FPD Rate: {row['taxa_fpd']:.1%}")
        if "cobranca" in row:
            lines.append(f"    Strategy: {row['cobranca']}")
            lines.append(f"    Channel: {row['canal']}")
            lines.append(f"    Min Down Payment: {row['entrada_minima']}")
            lines.append(f"    Max Installments: {row['parcelas_max']}")
            lines.append(f"    Interest: {row['juros']}")

    lines.append("\n## CAPTURE ANALYSIS")
    for key, info in policy["capture"].items():
        lines.append(f"  {key}: captures {info['fpd_captured']:.1%} FPD, "
                    f"blocks {info['good_blocked']} good ({info['good_blocked_pct']:.1%})")

    report = "\n".join(lines)
    (REPORTS / "collection_policy.txt").write_text(report)
    logger.info("Policy report saved")
