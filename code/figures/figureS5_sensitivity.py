"""
Figure-generation code: figureS5_sensitivity
Reconstructed from artifact lineage (version 4fa67e54-96e7-4d60-b1cc-b04e3ea11a89).
Source file at generation time: sensitivity_panel.png
Environment: python
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load sensitivity results
r = pd.read_csv("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/df695593-d446-4fef-a5ef-67269397eaeb/v8a1b8cc8_sensitivity_results.csv")

order = [
    "signed: AG+BZ+MB (baseline)",
    "signed: drop motifbreakR",
    "signed: leave-out AlphaGenome",
    "signed: leave-out Borzoi",
    "magnitude: AG+BZ",
    "magnitude: AG+BZ+Enformer",
    "magnitude: AG+BZ+Enformer+MB"
]
r = r.set_index("spec").loc[order].reset_index()

TEAL = "#1b7f8c"
ORANGE = "#e07b39"
PURPLE = "#6a5aa0"
colors = [PURPLE] + [TEAL] * 3 + [ORANGE] * 3
labels = [
    "AG + Borzoi + motifbreakR  (published baseline)",
    "drop motifbreakR",
    "leave out AlphaGenome",
    "leave out Borzoi",
    "AG + Borzoi",
    "AG + Borzoi + Enformer",
    "AG + Borzoi + Enformer + motifbreakR"
]
y = np.arange(len(r))[::-1]

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.axvspan(-0.055, 0.055, color="0.90", zorder=0)
ax.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
for yy, row, c in zip(y, r.itertuples(), colors):
    ax.plot([row.rho_hdi_lo, row.rho_hdi_hi], [yy, yy], color=c, lw=2.4, zorder=3, solid_capstyle="round")
    ax.plot(row.rho_mean, yy, "o", color=c, ms=8, zorder=4, markeredgecolor="white", markeredgewidth=1.1)
    ax.text(0.215, yy, f"{row.rho_mean:+.3f}", va="center", ha="left", fontsize=7, color=c)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r"calibrated model$\leftrightarrow$reporter agreement  $\rho$  (95% HDI)")
ax.set_xlim(-0.16, 0.30)
ax.set_ylim(-0.9, len(r) - 0.1)
ax.set_title("Latent-model agreement stays negligible across every specification", fontsize=9, loc="left")
ax.annotate("signed", xy=(-0.15, y[2]), fontsize=7.5, color=TEAL, rotation=90, va="center", ha="center", fontweight="bold")
ax.annotate("magnitude", xy=(-0.15, y[5]), fontsize=7.5, color=ORANGE, rotation=90, va="center", ha="center", fontweight="bold")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.5, 0.94, r"grey band $|\rho|\leq0.055$;   all R-hat $\leq$ 1.02;   n = 3,976", fontsize=6.8, color="0.4", ha="center")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("sensitivity_panel.png", dpi=300, bbox_inches="tight")