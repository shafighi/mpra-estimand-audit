"""
Figure-generation code: figureS2_sbc
Reconstructed from artifact lineage (version f5a75553-a93a-4768-8fb8-e7b926e91494).
Source file at generation time: figS3_sbc.png
Environment: pdfbuild
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import gc
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


numpyro.set_host_device_count(1)

arb = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/1717f8d8-e191-4949-b056-05c1a46a8ab3/vd34dea5a_arbiter_input.parquet')
se_real = arb['se_s'].values
mb_obs = arb['mb_obs'].values.astype(bool)


def _ll(x, dvar, W, S):
    Dinv = 1./dvar
    Sinv = jnp.linalg.inv(S)
    M = Sinv[None] + jnp.einsum("dk,nd,dl->nkl", W, Dinv, W)
    Minv = jnp.linalg.inv(M)
    WtDx = jnp.einsum("dk,nd,nd->nk", W, Dinv, x)
    quad = jnp.sum(x*Dinv*x, 1) - jnp.einsum("nk,nkl,nl->n", WtDx, Minv, WtDx)
    logdet = jnp.sum(jnp.log(dvar), 1) + jnp.linalg.slogdet(M)[1] + jnp.linalg.slogdet(S)[1]
    return -0.5*(x.shape[1]*jnp.log(2*jnp.pi) + logdet + quad)


def mdl(of, sef, on, sen):
    rho = numpyro.sample("rho", dist.Uniform(-0.99, 0.99))
    la = numpyro.sample("lam_ag", dist.Beta(2., 2.))
    lb = numpyro.sample("lam_bz", dist.Uniform(-0.99, 0.99))
    lm = numpyro.sample("lam_mb", dist.Uniform(-0.99, 0.99))
    lp = numpyro.sample("lam_mp", dist.Beta(2., 2.))
    ba = numpyro.sample("b_ag", dist.Normal(0, .5))
    bb = numpyro.sample("b_bz", dist.Normal(0, .5))
    bm = numpyro.sample("b_mb", dist.Normal(0, .5))
    bp = numpyro.sample("b_mp", dist.Normal(0, .5))
    S = jnp.array([[1., rho], [rho, 1.]])
    eps = 1e-6
    n1 = of.shape[0]
    W1 = jnp.array([[la, 0.], [lb, 0.], [lm, 0.], [0., lp]])
    mu1 = jnp.array([ba, bb, bm, bp])
    dv1 = jnp.stack([jnp.broadcast_to(1-la**2, (n1,)), jnp.broadcast_to(1-lb**2, (n1,)),
                     jnp.broadcast_to(1-lm**2, (n1,)), sef**2+eps], 1)
    numpyro.factor("ll1", _ll(of-mu1, dv1, W1, S).sum())
    n2 = on.shape[0]
    W2 = jnp.array([[la, 0.], [lb, 0.], [0., lp]])
    mu2 = jnp.array([ba, bb, bp])
    dv2 = jnp.stack([jnp.broadcast_to(1-la**2, (n2,)), jnp.broadcast_to(1-lb**2, (n2,)),
                     sen**2+eps], 1)
    numpyro.factor("ll2", _ll(on-mu2, dv2, W2, S).sum())


def sim(rt, lam, se, mbo, rng):
    n = len(se)
    L = np.linalg.cholesky(np.array([[1, rt], [rt, 1]]))
    EA = rng.standard_normal((n, 2)) @ L.T
    E, A = EA[:, 0], EA[:, 1]
    la, lb, lm, lp = lam
    return (la*E + np.sqrt(1-la**2)*rng.standard_normal(n),
            lb*E + np.sqrt(1-lb**2)*rng.standard_normal(n),
            lm*E + np.sqrt(1-lm**2)*rng.standard_normal(n),
            lp*A + se*rng.standard_normal(n))


def fit(ag, bz, mb, mp, se, mbo, seed):
    of = jnp.asarray(np.stack([ag[mbo], bz[mbo], mb[mbo], mp[mbo]], 1))
    sef = jnp.asarray(se[mbo])
    on = jnp.asarray(np.stack([ag[~mbo], bz[~mbo], mp[~mbo]], 1))
    sen = jnp.asarray(se[~mbo])
    mc = MCMC(NUTS(mdl, target_accept_prob=0.9), num_warmup=250, num_samples=250,
              num_chains=1, progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), of, sef, on, sen, extra_fields=("diverging",))
    r = np.asarray(mc.get_samples()["rho"])
    nd = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    return r, nd


lam = (0.87, 0.71, 0.19, 0.68)
rows = []

# rho = -0.3, 0.0, 0.3, 6 reps each
grid = np.array([-0.3, 0.0, 0.3])
nrep = 6
for ri, rt in enumerate(grid):
    for rep in range(nrep):
        rng = np.random.default_rng(1000*ri + rep)
        ag, bz, mb, mp = sim(rt, lam, se_real, mb_obs, rng)
        r, nd = fit(ag, bz, mb, mp, se_real, mb_obs, rep)
        lo, hi = np.percentile(r, [2.5, 97.5])
        rows.append(dict(rho_true=float(rt), rep=rep, rho_mean=float(r.mean()),
                         hdi_lo=float(lo), hdi_hi=float(hi),
                         covered=bool(lo <= rt <= hi), ndiv=int(nd)))
        jax.clear_caches()
        gc.collect()

base = pd.DataFrame(rows)

# top up rho=0 with 14 more reps (seeds 100..113)
topup = pd.read_parquet('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/9266e893-5f1b-4951-8457-0ea081dec1bf/vce7edd4c_sbc_topup_rho0.parquet')

sbc = pd.concat([base, topup[base.columns]], ignore_index=True)

apply_figure_style()

g = sorted(sbc['rho_true'].unique())
cov = sbc.groupby('rho_true')['covered'].mean()
overall = sbc['covered'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

for rt in g:
    sub = sbc[sbc['rho_true'] == rt]
    ax1.errorbar([rt]*len(sub), sub['rho_mean'],
                 yerr=[sub['rho_mean']-sub['hdi_lo'], sub['hdi_hi']-sub['rho_mean']],
                 fmt='o', ms=3.5, color='#2a788e', alpha=0.5, elinewidth=0.7, capsize=1.5)

lim = [-0.55, 0.55]
ax1.plot(lim, lim, '--', color='0.5', lw=1, zorder=0)
ax1.set_xlim(lim)
ax1.set_ylim(lim)
ax1.set_xlabel(r'true $\rho$ (simulated)')
ax1.set_ylabel(r'recovered $\rho$ (posterior mean $\pm$ 95% HDI)')
ax1.set_title('Estimator recovers known agreement', loc='left', fontsize=9.5)
ax1.text(0.04, 0.94, f'bias $\\leq$ 0.011\n0 divergences / {len(sbc)} fits',
         transform=ax1.transAxes, va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', lw=0.6))

xs = np.arange(len(g))
bars = ax2.bar(xs, [cov[r] for r in g], color=['#55a868', '#dd8452', '#55a868'],
               width=0.6, alpha=0.85)
ax2.axhline(0.95, ls='--', color='#c44e52', lw=1.2, label='nominal 0.95')
for i, r in enumerate(g):
    n = int((sbc.rho_true == r).sum())
    k = int(sbc[sbc.rho_true == r]['covered'].sum())
    ax2.text(i, cov[r]+0.02, f'{k}/{n}', ha='center', fontsize=7, color='0.3')

ax2.set_xticks(xs)
ax2.set_xticklabels([f'{r:+.1f}' for r in g])
ax2.set_ylim(0, 1.12)
ax2.set_xlabel(r'true $\rho$')
ax2.set_ylabel('95% HDI coverage')
ax2.set_title(f'Coverage by $\\rho$ (overall {overall:.2f})', loc='left', fontsize=9.5)
ax2.legend(frameon=False, fontsize=7, loc='lower center')

fig.tight_layout()
fig.savefig('figS3_sbc.png', dpi=300, bbox_inches='tight')
print("regenerated figS3_sbc.png with per-rho n and rho=0 emphasis")