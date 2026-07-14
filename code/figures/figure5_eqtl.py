"""
Figure-generation code: figure5_eqtl
Reconstructed from artifact lineage (version f4e59d05-0e50-4128-82e8-9638004a27cb).
Source file at generation time: fig6_benchmark2_eqtl.png
Environment: mpra
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib as mpl
import matplotlib.pyplot as plt

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
ov = pd.read_parquet("/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/f3b5fd61-9399-4618-9d04-bb8714295a9a/vf44e5c9c_deng_gandal_eqtl_overlap.parquet")

m = ov.merge(bc[['rsid','model_effect_canonical','mpra_logFC']], on='rsid', how='left')
m = m.merge(bz[['rsid','borzoi_abs']], on='rsid', how='left')

apply_figure_style()
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))

# Panel A: model & reporter vs fine-mapped endogenous eQTL (magnitude) — lollipop with CIs
axA = axes[0]
rows=[('AlphaGenome',0.238,0.141,0.328,'#2a788e'),
      ('Borzoi',0.276,None,None,'#3b9ab2'),
      ('lentiMPRA',-0.026,-0.130,0.078,'#c44e52')]
for (lab,r,lo,hi,col),yy in zip(rows,[2,1,0]):
    if lo is not None: axA.plot([lo,hi],[yy,yy],color=col,lw=1.4,alpha=0.6,zorder=1)
    axA.plot([0,r],[yy,yy],color=col,lw=2.6,solid_capstyle='round',zorder=2)
    axA.scatter([r],[yy],s=150,color=col,zorder=3,edgecolor='white',lw=0.7)
    axA.text(r+(0.012 if r>=0 else -0.012),yy+0.24,f'$\\rho$={r:.2f}',color=col,fontsize=8,
             ha='left' if r>=0 else 'right',va='bottom')
axA.axvline(0,color='0.6',lw=0.8,zorder=0)
axA.set_yticks([2,1,0]); axA.set_yticklabels(['AlphaGenome','Borzoi','lentiMPRA'])
axA.set_ylim(-0.6,2.7); axA.set_xlim(-0.20,0.40)
axA.set_xlabel('Spearman $\\rho$ vs fine-mapped\neQTL effect $|$posterior$|$')
axA.set_title('Models track the endogenous eQTL;\nthe reporter does not',loc='left',fontsize=9.5)
axA.text(0.97,0.05,'n = 370 variants\nin eQTL credible sets',transform=axA.transAxes,ha='right',va='bottom',fontsize=6.5,color='0.4')
panel_letter(axA,'a')

# Panel B: direction-of-effect concordance vs PIP threshold
axB = axes[1]
thr=[0.0,0.1,0.5]; conc=[0.505,0.603,0.80]; ns=[370,73,25]; ps=['n.s.','p=0.05','p=0.002']
xs=range(len(thr))
bars=axB.bar(xs,conc,color=['#cfd8dc','#7fb0bf','#2a788e'],edgecolor='0.3',lw=0.6,width=0.62,zorder=3)
axB.axhline(0.5,color='#c44e52',lw=1.2,ls='--',zorder=2,label='chance')
for x,c,n,pt in zip(xs,conc,ns,ps):
    axB.text(x,c+0.015,f'{c:.2f}\n{pt}\n(n={n})',ha='center',va='bottom',fontsize=7)
axB.set_xticks(list(xs)); axB.set_xticklabels(['all','PIP$\\geq$0.1','PIP$\\geq$0.5'])
axB.set_ylim(0,1.0); axB.set_ylabel('AlphaGenome direction-of-effect\nconcordance with eQTL')
axB.set_xlabel('fine-mapping confidence (causal probability)')
axB.set_title('Concordance rises as variants\nbecome confidently causal',loc='left',fontsize=9.5)
axB.legend(frameon=False,loc='upper left',fontsize=7)
panel_letter(axB,'b')

# Panel C: scatter AG |eff| vs eQTL |posterior|, sized by PIP
axC = axes[2]
sz=20+260*m['pip'].clip(0,1)
sc=axC.scatter(m['model_effect_canonical'].abs(), m['posterior_mean'].abs(), s=sz,
               c=m['pip'], cmap='viridis', alpha=0.75, lw=0.3, edgecolor='white')
axC.set_xlabel('AlphaGenome $|$effect$|$ (native locus)')
axC.set_ylabel('fine-mapped eQTL $|$posterior effect$|$')
axC.set_title('Endogenous agreement, $\\rho$=0.24\n(marker size/colour = PIP)',loc='left',fontsize=9.5)
cb=fig.colorbar(sc,ax=axC,fraction=0.046,pad=0.03); cb.set_label('PIP',fontsize=7); cb.ax.tick_params(labelsize=6)
axC.margins(0.05)
panel_letter(axC,'c')

fig.tight_layout()
fig.savefig('fig6_benchmark2_eqtl.png', dpi=300, bbox_inches='tight')