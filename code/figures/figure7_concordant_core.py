"""
Figure-generation code: figure7_concordant_core
Reconstructed from artifact lineage (version abac44df-c24d-436c-a362-b0324901321d).
Source file at generation time: fig9_comparison_panel.png
Environment: mpra
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# skill:figure-style kernel.py (auto-injected on skill load)
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


def set_frame(ax, style="open"):
    show = {"open": (False, False, True, True),
            "boxed": (True, True, True, True),
            "none": (False, False, False, False)}[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


# Load data
panel = pd.read_csv('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/7bf7b4aa-e88d-48bc-a64c-cebe93e93d8d/v0ac40a74_comparison_panel.csv')

apply_figure_style()
grp_order=['strong_concordant','model_only','mpra_only','both_quiet']
grp_color={'strong_concordant':'#333333','model_only':'#d55e00','mpra_only':'#0072b2','both_quiet':'#999999'}
grp_short={'strong_concordant':'concordant','model_only':'model-only','mpra_only':'MPRA-only','both_quiet':'both-quiet'}
grp_mark={'strong_concordant':'o','model_only':'s','mpra_only':'^','both_quiet':'x'}

fig=plt.figure(figsize=(7.4,3.5))
gs=GridSpec(1,2,width_ratios=[1.25,1.0],wspace=0.32,figure=fig)
axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])

# Panel a: quadrant scatter
axA.axhline(0,color='#ccc',lw=0.7,zorder=0); axA.axvline(0,color='#ccc',lw=0.7,zorder=0)
lim=1.9
axA.fill_between([0,lim],[0,0],[lim,lim],color='#eaf3ea',alpha=0.6,zorder=0)
axA.fill_between([-lim,0],[-lim,-lim],[0,0],color='#eaf3ea',alpha=0.6,zorder=0)
for g in grp_order:
    sub=panel[panel['group']==g]
    axA.scatter(sub['mpra_signed'],sub['ag_signed'],c=grp_color[g],marker=grp_mark[g],s=42,
                label=grp_short[g],edgecolor='white' if g!='both_quiet' else 'none',linewidth=0.5,zorder=3)
axA.set_xlabel('MPRA signed effect (logFC)',fontsize=7)
axA.set_ylabel('AlphaGenome signed effect',fontsize=7)
axA.set_xlim(-1.9,1.9); axA.set_ylim(-0.9,0.55)
axA.text(0.97,0.03,'shaded = same sign',transform=axA.transAxes,fontsize=5.8,ha='right',va='bottom',color='#4a7a4a')
for sp in ['top','right']: axA.spines[sp].set_visible(False)
axA.legend(fontsize=5.8,frameon=False,loc='upper left',handletextpad=0.3,borderpad=0.2)

# Panel b: |effect| by group for MPRA vs model — the dissociation
panel['abs_mpra']=panel['mpra_signed'].abs()
panel['abs_ag']=panel['ag_signed'].abs()
gx=np.arange(len(grp_order)); w=0.36
mpra_m=[panel[panel['group']==g]['abs_mpra'].mean() for g in grp_order]
ag_m=[panel[panel['group']==g]['abs_ag'].mean() for g in grp_order]
axB.bar(gx-w/2,mpra_m,w,color='#4a7a4a',label='|MPRA|')
axB.bar(gx+w/2,ag_m,w,color='#0072b2',label='|AlphaGenome|')
# overlay raw points
for j,g in enumerate(grp_order):
    sub=panel[panel['group']==g]
    axB.scatter(np.full(len(sub),j-w/2)+np.random.uniform(-0.06,0.06,len(sub)),sub['abs_mpra'],s=9,color='#2c4c2c',zorder=3)
    axB.scatter(np.full(len(sub),j+w/2)+np.random.uniform(-0.06,0.06,len(sub)),sub['abs_ag'],s=9,color='#004c7a',zorder=3)
axB.set_xticks(gx); axB.set_xticklabels([grp_short[g] for g in grp_order],fontsize=6.2,rotation=20,ha='right')
axB.set_ylabel('mean |allelic effect|',fontsize=7)
for sp in ['top','right']: axB.spines[sp].set_visible(False)
axB.legend(fontsize=6,frameon=False,loc='upper right')

for ax,L in [(axA,'a'),(axB,'b')]:
    ax.text(-0.16,1.02,L,transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')
fig.suptitle('Matched comparison panel: model and reporter dissociate outside the concordant group',fontsize=8,y=1.02,x=0.5)
fig.savefig('fig9_comparison_panel.png',dpi=300,bbox_inches='tight')