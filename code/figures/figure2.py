"""
Figure-generation code: figure2
Reconstructed from artifact lineage (version 450d936e-aa7c-4eb9-84df-b91e5e0e815d).
Source file at generation time: fig1_triangulation.png
Environment: mpra
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text
import pandas as pd
import numpy as np
import scipy.stats as ss

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


bc = pd.read_parquet("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/d15f0081-af30-40f1-a143-5f2c75c9f64d/va46016a7_benchmark_canonical.parquet")
bz = pd.read_parquet("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/5c2459a6-f47a-4ab4-912b-f3d961e8a7ae/v6aed2165_borzoi_merged.parquet")
m = bz.merge(bc[['rsid','model_effect_canonical','mpra_logFC']], on='rsid', how='left')

apply_figure_style(sizes=(7,6,6))
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(6.82, 3.96))
x = m['model_effect_canonical'].abs(); y = m['mpra_logFC'].abs()
dav = m['is_dav'].astype(bool); ok = x.notna() & y.notna()
axA.scatter(x[ok & ~dav], y[ok & ~dav], s=7, c=META_GREY, alpha=0.35, lw=0, label='tested (n=3,900)')
axA.scatter(x[ok & dav], y[ok & dav], s=26, c='#c44e52', alpha=0.9, lw=0.4, edgecolor='white', label='DAV (n=76)', zorder=3)
axA.set_xlabel('AlphaGenome |effect|  (native locus, 100 kb)')
axA.set_ylabel('lentiMPRA |allelic effect|  ($|\\log_2$FC$|$)')
axA.set_title('Model vs reporter', loc='left')
axA.text(0.04, 0.96, 'Spearman $\\rho$ = $-$0.01\n$p$ = 0.43 (n.s.)',
         transform=axA.transAxes, va='top', ha='left', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', lw=0.6))
axA.legend(frameon=False, loc='upper right', fontsize=6.5, markerscale=1.2)
axA.margins(0.04); panel_letter(axA, 'a')

pairs = [('AlphaGenome ↔ motifbreakR', 0.104, '#2a788e', '$p$<0.01'),
         ('lentiMPRA ↔ motifbreakR', -0.024, '#c44e52', 'n.s.'),
         ('AlphaGenome ↔ lentiMPRA', -0.013, META_GREY, 'n.s.')]
for (lab,rho,col,ptxt),yy in zip(pairs,[2,1,0]):
    axB.plot([0,rho],[yy,yy], color=col, lw=2.5, solid_capstyle='round', zorder=1)
    axB.scatter([rho],[yy], s=130, color=col, zorder=2, edgecolor='white', lw=0.6)
    ha = 'left' if rho>=0 else 'right'; off = 0.008 if rho>=0 else -0.008
    axB.text(rho+off*3, yy+0.22, f'$\\rho$ = {rho:.3f}  ({ptxt})', color=col, va='bottom', ha=ha, fontsize=7)
axB.axvline(0, color='0.6', lw=0.8, zorder=0)
axB.set_yticks([2,1,0]); axB.set_yticklabels([p[0] for p in pairs])
axB.set_ylim(-0.6, 2.7); axB.set_xlim(-0.06, 0.20)
axB.set_xlabel('Spearman $\\rho$  (effect magnitude)')
axB.set_title('Sequence estimators vs reporter', loc='left')
axB.text(0.97, 0.06, 'n = 702 variants with\nmotifbreakR annotation', transform=axB.transAxes, va='bottom', ha='right', fontsize=6.5, color='0.4')
panel_letter(axB, 'b')
fig1.tight_layout()
fig1.savefig('fig1_triangulation.png', dpi=300, bbox_inches='tight')