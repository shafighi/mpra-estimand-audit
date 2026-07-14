"""
Figure-generation code: figure6_acp
Reconstructed from artifact lineage (version a2764d87-5a0b-445a-b8da-b58731397771).
Source file at generation time: acp_summary.png
Environment: mpra
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import spearmanr
import matplotlib.ticker as mt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re

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


apply_figure_style(sizes=(7, 6, 6))
d = pd.read_parquet("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/e7e0a80f-f689-4f41-b4a4-11451a1305b4/v383c62f1_acp_classified.parquet")
R = json.load(open("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/dc4fab5f-2413-4e85-871f-5bcc3f3cd0c7/v1ada41a2_ladder.json"))['R']
bs = json.load(open("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/5bbc042c-2a62-4dc2-8b2e-d978124b403c/v7e2acac3_acp_bayes_summary.json"))
idata = az.from_netcdf("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/92fa83a8-b3d4-4a82-9004-8a9a2cb75816/va03f560b_acp_idata.nc")
g = np.asarray(idata.posterior["gamma_eff"]).ravel()
enr = pd.read_csv("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/4082e3db-ed20-4ea0-aa3a-c38b2dce2ee4/vb76b132c_enrichment_results.csv")

ws, hs = 0.68, 0.86
fig = plt.figure(figsize=(13*ws, 8.2*hs))
gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.42, top=0.86, bottom=0.09)

axA = fig.add_subplot(gs[0, 0])
sc = axA.scatter(d['M_mpra'], d['Y_eqtl'], s=12+55*d['pip'], c=d['pip'], cmap='viridis', alpha=0.75, edgecolor='white', linewidth=0.3, vmin=0, vmax=1)
axA.axhline(0, color=META_GREY, lw=0.6); axA.axvline(0, color=META_GREY, lw=0.6)
rho = spearmanr(d['M_mpra'], d['Y_eqtl'])[0]
axA.set_xlabel('MPRA allelic effect'); axA.set_ylabel('Endogenous eQTL effect')
axA.set_title('1. MPRA vs endogenous ($\\rho$=%.2f)' % rho, fontsize=7)
cax = inset_axes(axA, width='4%', height='42%', loc='lower right', borderpad=1.0)
cb = fig.colorbar(sc, cax=cax); cb.set_label('PIP', fontsize=5.5); cb.ax.tick_params(labelsize=5)
panel_letter(axA, 'a')

axB = fig.add_subplot(gs[0, 1])
order = ['M0', 'M1', 'M2', 'M3', 'M5', 'M4', 'M6']; labs = ['null', 'MPRA', 'context', 'M+C', 'seq-model', 'M×C', 'all']
means = [np.mean(R[m]) for m in order]; los = [np.percentile(R[m], 2.5) for m in order]; his = [np.percentile(R[m], 97.5) for m in order]
foc = {'M4': '#c0392b', 'M5': '#2e7d32'}; cols = [foc.get(m, '#3b6ea5') for m in order]; x = np.arange(len(order))
axB.errorbar(x, means, yerr=[np.array(means)-np.array(los), np.array(his)-np.array(means)], fmt='none', ecolor='#bbb', elinewidth=1, capsize=2, zorder=1)
axB.scatter(x, means, c=cols, s=45, zorder=3, edgecolor='white', linewidth=0.4); axB.axhline(0, color=META_GREY, lw=0.8, ls='--')
axB.set_xticks(x); axB.set_xticklabels(labs, fontsize=5.6, rotation=30, ha='right'); axB.set_ylabel('held-out R²')
axB.set_title('2. No model beats null', fontsize=7); panel_letter(axB, 'b')

axC = fig.add_subplot(gs[0, 2])
axC.hist(g, bins=55, color='#c0392b', alpha=0.55, density=True); axC.axvline(0, color=META_GREY, lw=1, ls='--'); axC.axvline(g.mean(), color='#c0392b', lw=1.2)
lo, hi = bs['gamma_hdi']; axC.axvspan(lo, hi, color='#c0392b', alpha=0.12)
axC.set_xlabel('γ (MPRA × context)'); axC.set_ylabel('posterior')
axC.set_title('3. Interaction γ ≈ 0 (P=%.2f)' % bs['p_gamma_gt0'], fontsize=7); panel_letter(axC, 'c')

axD = fig.add_subplot(gs[1, 0])
sx = np.asarray(idata.posterior["sigma_extra"]).ravel(); s = d['Y_sd'].values; pip = d['pip'].values.clip(1e-3, 1)
med_meas = np.median(s/np.sqrt(pip))
axD.hist(sx, bins=45, color='#8e44ad', alpha=0.6, density=True, label='σ intrinsic')
axD.axvline(med_meas, color='#2c3e50', lw=1.3, label='σ measurement (med)')
axD.set_xlabel('effect-SD units'); axD.set_ylabel('posterior')
axD.set_title('4. Noise dominates (σ≈%.2f)' % sx.mean(), fontsize=7)
axD.legend(fontsize=5.6, frameon=False); panel_letter(axD, 'd')

axE = fig.add_subplot(gs[1, 1])
cmap = {'Propagating': '#2e7d32', 'Latent potential': '#c0392b', 'Context-only': '#2c3e50', 'Null': '#cccccc', 'Ambiguous': '#ebebeb'}
for cls in ['Ambiguous', 'Null', 'Context-only', 'Propagating', 'Latent potential']:
    sdf = d[d['prop_class'] == cls]; al = 0.3 if cls == 'Ambiguous' else 0.85
    axE.scatter(sdf['M_mpra'], sdf['Y_eqtl'], s=12+40*sdf['pip'], c=cmap[cls], alpha=al, edgecolor='white', linewidth=0.3,
                label=cls+' ('+str(len(sdf))+')', zorder=3 if cls in ('Propagating', 'Latent potential') else 1)
axE.axhline(0, color=META_GREY, lw=0.6); axE.axvline(0, color=META_GREY, lw=0.6)
axE.set_xlabel('MPRA effect'); axE.set_ylabel('endogenous effect')
axE.set_title('5. Latent class dominant ("', fontsize=7)
axE.legend(fontsize=4.8, frameon=False, loc='upper left', handletextpad=0.2, labelspacing=0.25); panel_letter(axE, 'e')

axF = fig.add_subplot(gs[1, 2])
ys = np.arange(len(enr))[::-1]
for i in range(len(enr)):
    r = enr.iloc[i]; col = '#c0392b' if r['p'] < 0.05 else '#3b6ea5'
    axF.plot([r['lo'], r['hi']], [ys[i], ys[i]], color=col, lw=1.4, solid_capstyle='round')
    axF.scatter([r['or']], [ys[i]], color=col, s=30, zorder=3, edgecolor='white', linewidth=0.4)
axF.axvline(1, color=META_GREY, lw=0.9, ls='--'); axF.set_xscale('log')
axF.xaxis.set_minor_locator(mt.NullLocator()); axF.set_xticks([0.25, 0.5, 1, 2, 4]); axF.xaxis.set_major_formatter(mt.FixedFormatter(['.25', '.5', '1', '2', '4']))
axF.set_yticks(ys); axF.set_yticklabels(enr['feature'], fontsize=6); axF.tick_params(axis='x', labelsize=5.6)
axF.set_xlabel('OR for latent potential'); axF.set_title('6. No context explains latent (n=34)', fontsize=7); panel_letter(axF, 'f')

fig.suptitle('Activity–Context–Propagation: reporter potential does not predict endogenous effect (n=370)',
             fontsize=9.5, y=0.96)
fig.savefig('acp_summary.png', dpi=200, bbox_inches='tight')