"""
Figure-generation code: figure3_cross_estimator
Reconstructed from artifact lineage (version 2f5b2d35-5203-471f-ac34-b9f7114a4fb7).
Source file at generation time: F9_cross_estimator_synthesis.png
Environment: python
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

# Load data
BC = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/d15f0081-af30-40f1-a143-5f2c75c9f64d/va46016a7_benchmark_canonical.parquet')
BZ = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/5c2459a6-f47a-4ab4-912b-f3d961e8a7ae/v6aed2165_borzoi_merged.parquet')
EQ = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/d38bc9ff-89c4-4985-9122-0521fa6fd965/vd477d164_benchmark2_results.parquet')
ENF = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/2e88aefb-d7ae-4956-8962-aef5321b9f84/v44546281_enformer_merged.parquet')
EV = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/3ae2ebb9-1451-41a8-8115-5dae5b645ffb/vd4d4f22d_evo2_merged_256bp.parquet')

# Build unified frame on |effect| (magnitude)
U = BC[["rsid", "model_effect_canonical", "mpra_logFC", "is_dav"]].copy()
U = U.merge(BZ[["rsid", "borzoi_abs"]], on="rsid", how="left")
U = U.merge(ENF[["rsid", "enformer_brain_abs"]], on="rsid", how="left")
U = U.merge(EV[["rsid", "evo2_abs"]], on="rsid", how="left")
U["ag_abs"] = U["model_effect_canonical"].abs()
U["reporter_abs"] = U["mpra_logFC"].abs()

est = {"AlphaGenome": "ag_abs", "Borzoi": "borzoi_abs", "Enformer": "enformer_brain_abs",
       "Evo 2": "evo2_abs", "lentiMPRA": "reporter_abs"}
cols = list(est.keys())

M = np.full((len(cols), len(cols)), np.nan)
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if i == j:
            M[i, j] = 1.0
            continue
        s = U[[est[a], est[b]]].dropna()
        M[i, j] = spearmanr(s[est[a]].values, s[est[b]].values)[0]
Mdf = pd.DataFrame(M, index=cols, columns=cols)

# eQTL referee column
me = EQ.copy()
me["eqtl_abs"] = me["posterior_mean"].abs()
me = me.merge(ENF[["rsid", "enformer_brain_abs"]], on="rsid", how="left")
me = me.merge(EV[["rsid", "evo2_abs"]], on="rsid", how="left")
me["ag_abs"] = me["model_effect_canonical"].abs()
me["reporter_abs"] = me["mpra_logFC"].abs()
eqtl_rho = {}
for name, c in [("AlphaGenome", "ag_abs"), ("Borzoi", "borzoi_abs"), ("Enformer", "enformer_brain_abs"),
                ("Evo 2", "evo2_abs"), ("lentiMPRA", "reporter_abs")]:
    s = me[[c, "eqtl_abs"]].dropna()
    eqtl_rho[name] = float(spearmanr(s[c].values, s["eqtl_abs"].values)[0])

# Build display matrix
labels = ["AlphaGenome", "Borzoi", "Enformer", "Evo 2", "lentiMPRA"]
Mv = Mdf.loc[labels, labels].values.copy()
eq = np.array([eqtl_rho[l] for l in labels]).reshape(-1, 1)
disp = np.hstack([Mv, eq])
col_labels = labels + ["eQTL\nreferee"]

fig, ax = plt.subplots(figsize=(3.4, 2.7))
cmap = plt.cm.RdBu_r
norm = mpl.colors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.85)
im = ax.imshow(disp, cmap=cmap, norm=norm, aspect="auto")
ax.axvline(4.5, color="white", lw=3)
ax.axvline(4.5, color="#333", lw=0.8)
for i in range(disp.shape[0]):
    for j in range(disp.shape[1]):
        v = disp[i, j]
        if i == j and j < 5:
            ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="#888")
            continue
        txt = f"{v:+.2f}"
        c = "white" if abs(v) > 0.5 else "#222"
        ax.text(j, i, txt, ha="center", va="center", fontsize=5.4, color=c,
                fontweight="bold" if (i == 4 or j == 5 or j == 4) else "normal")
ax.set_xticks(range(6))
ax.set_xticklabels(col_labels, fontsize=5.2, rotation=35, ha="right")
ax.set_yticks(range(5))
ax.set_yticklabels(labels, fontsize=5.6)
ax.add_patch(plt.Rectangle((-0.5, 3.5), 5, 1, fill=False, edgecolor="#e07b39", lw=1.6))
ax.add_patch(plt.Rectangle((4.5, -0.5), 1, 5, fill=False, edgecolor="#6a5aa0", lw=1.6))
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("magnitude Spearman ρ", fontsize=5.6)
cb.ax.tick_params(labelsize=5)
ax.set_title("Same variants, different estimators: the reporter agrees with nothing", fontsize=5.9, pad=5)
fig.tight_layout(pad=0.3)
fig.savefig('F9_cross_estimator_synthesis.png', dpi=300, bbox_inches="tight")
plt.close(fig)
print("F9 synthesis done")