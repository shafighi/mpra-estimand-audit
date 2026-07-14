"""Days 2-3: model<->assay agreement, reliability weighting, and the failure map.

Written fresh for the hackathon. Takes a merged table with one row per variant:
  variant_id, mpra_effect, mpra_fdr (or mpra_se), is_dav (bool),
  model_effect  (AlphaGenome summary scalar, e.g. mean allelic raw_score over
                 brain-relevant tracks), baseline_activity (ref enhancer activity).

Provides:
  * summarize_agreement  -> Spearman/Pearson (+ reliability-weighted), sign concordance
  * discrimination       -> AUROC/AUPRC for DAV vs negatives using |model_effect|
  * reliability_weights  -> weights from FDR or SE (lower noise -> higher weight)
  * failure_frame        -> per-variant disagreement + stratification bins
  * plot_agreement / plot_failure_map -> the two hero figures
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ---------- reliability weighting ----------
def reliability_weights(df: pd.DataFrame, fdr_col: str = "mpra_fdr",
                        se_col: str | None = None, floor: float = 1e-3) -> np.ndarray:
    """Weight each measurement by its confidence. If an SE column is given, use
    1/SE^2 (inverse-variance). Otherwise derive a weight from FDR via -log10(FDR)."""
    if se_col and se_col in df:
        se = df[se_col].to_numpy(float)
        w = 1.0 / np.clip(se, floor, None) ** 2
    else:
        fdr = df[fdr_col].to_numpy(float)
        w = -np.log10(np.clip(fdr, floor, 1.0))
    w = np.nan_to_num(w, nan=0.0)
    return w / w.mean() if w.mean() > 0 else w


def _weighted_pearson(x, y, w):
    xm = np.average(x, weights=w); ym = np.average(y, weights=w)
    cov = np.average((x - xm) * (y - ym), weights=w)
    vx = np.average((x - xm) ** 2, weights=w); vy = np.average((y - ym) ** 2, weights=w)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


# ---------- agreement metrics ----------
def summarize_agreement(df: pd.DataFrame, model_col: str = "model_effect",
                        mpra_col: str = "mpra_effect",
                        weights: np.ndarray | None = None) -> dict:
    d = df[[model_col, mpra_col]].dropna()
    x = d[model_col].to_numpy(float); y = d[mpra_col].to_numpy(float)
    out = {
        "n": int(len(d)),
        "pearson_r": float(stats.pearsonr(x, y)[0]) if len(d) > 2 else np.nan,
        "pearson_p": float(stats.pearsonr(x, y)[1]) if len(d) > 2 else np.nan,
        "spearman_r": float(stats.spearmanr(x, y)[0]) if len(d) > 2 else np.nan,
        "sign_concordance": float(np.mean(np.sign(x) == np.sign(y))) if len(d) else np.nan,
    }
    if weights is not None:
        w = np.asarray(weights, float)[d.index.to_numpy() if len(weights) == len(df) else slice(None)]
        out["weighted_pearson_r"] = float(_weighted_pearson(x, y, w))
    return out


def discrimination(df: pd.DataFrame, model_col: str = "model_effect",
                   label_col: str = "is_dav") -> dict:
    """How well |model effect| separates DAVs from negatives (no sklearn dependency)."""
    d = df[[model_col, label_col]].dropna()
    score = np.abs(d[model_col].to_numpy(float)); y = d[label_col].to_numpy(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return {"auroc": np.nan, "auprc": np.nan, "n_pos": int(y.sum())}
    # AUROC via Mann-Whitney U
    order = np.argsort(score); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(score) + 1)
    auroc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    # AUPRC via trapezoid over sorted thresholds
    idx = np.argsort(-score); ys = y[idx]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    prec = tp / (tp + fp); rec = tp / y.sum()
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x renamed trapz
    auprc = float(_trap(prec, rec))
    return {"auroc": float(auroc), "auprc": auprc, "n_pos": int(y.sum())}


# ---------- failure / attribution frame ----------
def failure_frame(df: pd.DataFrame, model_col: str = "model_effect",
                  mpra_col: str = "mpra_effect",
                  activity_col: str = "baseline_activity", n_bins: int = 4) -> pd.DataFrame:
    """Per-variant disagreement, z-scored so model and assay are comparable,
    plus stratification bins for the failure map."""
    out = df.copy()
    zx = stats.zscore(out[model_col].astype(float), nan_policy="omit")
    zy = stats.zscore(out[mpra_col].astype(float), nan_policy="omit")
    out["disagreement"] = np.abs(zx - zy)
    out["effect_bin"] = pd.qcut(out[mpra_col].abs(), q=min(n_bins, out[mpra_col].nunique()),
                                labels=False, duplicates="drop")
    if activity_col in out:
        out["activity_bin"] = pd.qcut(out[activity_col], q=min(n_bins, out[activity_col].nunique()),
                                      labels=False, duplicates="drop")
    return out


# ---------- figures (the two Day 2-3 heroes) ----------
def plot_agreement(df: pd.DataFrame, model_col: str = "model_effect",
                   mpra_col: str = "mpra_effect", label_col: str = "is_dav",
                   weights: np.ndarray | None = None, ax=None, title: str | None = None):
    """Fig 1: predicted (model) vs measured (MPRA) effect scatter, DAVs highlighted,
    with the correlation stats annotated. Returns (fig, ax, stats_dict)."""
    import matplotlib.pyplot as plt
    d = df.dropna(subset=[model_col, mpra_col]).copy()
    stats_d = summarize_agreement(d, model_col, mpra_col, weights=weights)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
    else:
        fig = ax.figure
    neg = d[~d[label_col].astype(bool)] if label_col in d else d
    ax.scatter(neg[model_col], neg[mpra_col], s=10, c="#b7c9c9", alpha=0.5,
               edgecolors="none", label=f"tested (n={len(neg)})", rasterized=True)
    if label_col in d:
        dav = d[d[label_col].astype(bool)]
        ax.scatter(dav[model_col], dav[mpra_col], s=26, c="#0b6b6b",
                   edgecolors="white", linewidths=0.4, label=f"DAV (n={len(dav)})", zorder=3)
    lim = np.nanpercentile(np.abs(np.r_[d[model_col], d[mpra_col]]), 99)
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0); ax.axvline(0, color="#ccc", lw=0.8, zorder=0)
    ax.set_xlabel("AlphaGenome predicted effect (endogenous)")
    ax.set_ylabel("MPRA measured allelic effect (episomal, log2FC)")
    r = stats_d.get("spearman_r", np.nan); wr = stats_d.get("weighted_pearson_r")
    txt = f"Spearman ρ = {r:.3f}\nsign concord. = {stats_d['sign_concordance']:.2f}"
    if wr is not None:
        txt += f"\nweighted r = {wr:.3f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title(title or "Model vs. assay: two imperfect estimators")
    fig.tight_layout()
    return fig, ax, stats_d


def plot_failure_map(ff: pd.DataFrame, effect_bin="effect_bin",
                     activity_bin="activity_bin", value="disagreement", ax=None):
    """Fig 2 (hero): mean model-assay disagreement stratified by measured effect size
    (x) and baseline enhancer activity (y). Bright cells = where the estimators diverge."""
    import matplotlib.pyplot as plt
    if activity_bin not in ff:
        raise ValueError("failure_frame needs a baseline activity column for the 2D map")
    piv = ff.pivot_table(index=activity_bin, columns=effect_bin, values=value, aggfunc="mean")
    piv = piv.sort_index(ascending=False)  # high activity on top
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
    else:
        fig = ax.figure
    im = ax.imshow(piv.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"Q{int(c)+1}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"Q{int(i)+1}" for i in piv.index])
    ax.set_xlabel("MPRA effect size (quartile)")
    ax.set_ylabel("baseline enhancer activity (quartile)")
    ax.set_title("Failure map: where model & assay disagree")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > np.nanmean(piv.values) else "#222", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean |disagreement| (z-units)")
    fig.tight_layout()
    return fig, ax, piv


if __name__ == "__main__":
    # Synthetic smoke test: 200 variants, moderate correlation + noise.
    rng = np.random.default_rng(0)
    n = 200
    latent = rng.normal(size=n)
    df = pd.DataFrame({
        "variant_id": [f"v{i}" for i in range(n)],
        "mpra_effect": latent + rng.normal(scale=0.6, size=n),
        "model_effect": latent + rng.normal(scale=0.8, size=n),
        "mpra_fdr": rng.uniform(1e-4, 0.5, size=n),
        "baseline_activity": rng.normal(size=n),
        "is_dav": rng.random(n) < 0.2,
    })
    w = reliability_weights(df)
    print("agreement:", summarize_agreement(df, weights=w))
    print("discrimination:", discrimination(df))
    ff = failure_frame(df)
    print("failure frame cols:", [c for c in ff.columns if c in
          ("disagreement", "effect_bin", "activity_bin")],
          "| mean disagreement:", round(ff.disagreement.mean(), 3))
    import matplotlib
    matplotlib.use("Agg")
    fig1, _, _ = plot_agreement(df, weights=w)
    fig1.savefig("/tmp/_test_agreement.png", dpi=110)
    fig2, _, piv = plot_failure_map(ff)
    fig2.savefig("/tmp/_test_failuremap.png", dpi=110)
    print("figures rendered OK; failure-map shape:", piv.shape)
