"""Sensitivity analysis for the latent-variable model (MPRA-Arbiter).

Re-fits the calibrated model<->reporter agreement rho under altered
specifications: which endogenous indicators load on E, and signed vs
magnitude effects. Same marginalized linear-Gaussian core as latent_arbiter.py:
each endogenous indicator k loads on latent E with loading lam_k and residual
var (1-lam_k^2); the reporter loads on latent A (heteroscedastic via se);
Corr(E,A)=rho is the headline. Per-indicator masks allow a variant to be
missing an indicator (e.g. motifbreakR on 702).
"""
import json, argparse, itertools
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(1)


def make_model(indicators, positive=False):
    """indicators: list of endogenous indicator names loading on E (reporter is separate)."""
    K = len(indicators)

    def model(obs, se, mask):
        # obs: (n, K+1) = [ind_1..ind_K, mpra]; mask: (n, K) 1 if indicator observed
        rho = numpyro.sample("rho", dist.Uniform(-0.99, 0.99))
        # first indicator anchors E to (0,1) for sign identifiability
        lams = []
        for j, name in enumerate(indicators):
            if j == 0 or positive:
                lams.append(numpyro.sample(f"lam_{name}", dist.Beta(2.0, 2.0)))
            else:
                lams.append(numpyro.sample(f"lam_{name}", dist.Uniform(-0.99, 0.99)))
        lam_mp = numpyro.sample("lam_mp", dist.Beta(2.0, 2.0))
        b = [numpyro.sample(f"b_{name}", dist.Normal(0.0, 0.5)) for name in indicators]
        b_mp = numpyro.sample("b_mp", dist.Normal(0.0, 0.5))
        lams = jnp.stack(lams)                       # (K,)
        S = jnp.array([[1.0, rho], [rho, 1.0]])
        Sinv = jnp.linalg.inv(S)
        eps = 1e-6
        n = obs.shape[0]
        # residual variances: indicators (1-lam^2) gated by mask (unobserved -> huge var);
        # mpra = se^2
        BIG = 1e8
        dvar_ind = (1 - lams**2)[None, :]                        # (1,K)
        dvar_ind = jnp.where(mask > 0, dvar_ind, BIG)            # (n,K)
        dvar = jnp.concatenate([dvar_ind, (se**2 + eps)[:, None]], axis=1)  # (n,K+1)
        mu = jnp.concatenate([jnp.stack(b), b_mp[None]])         # (K+1,)
        # W maps [E,A] -> obs means; indicators load on E, mpra on A
        W = jnp.zeros((K + 1, 2))
        W = W.at[:K, 0].set(lams)
        W = W.at[K, 1].set(lam_mp)
        x = obs - mu
        Dinv = 1.0 / dvar
        WtDinvW = jnp.einsum("dk,nd,dl->nkl", W, Dinv, W)
        M = Sinv[None] + WtDinvW
        Minv = jnp.linalg.inv(M)
        WtDinvx = jnp.einsum("dk,nd,nd->nk", W, Dinv, x)
        quad = jnp.sum(x * Dinv * x, 1) - jnp.einsum("nk,nkl,nl->n", WtDinvx, Minv, WtDinvx)
        logdet = jnp.sum(jnp.log(dvar), 1) + jnp.linalg.slogdet(M)[1] + jnp.linalg.slogdet(S)[1]
        ll = -0.5 * ((K + 1) * jnp.log(2 * jnp.pi) + logdet + quad)
        numpyro.factor("ll", ll.sum())

    return model


def fit(df, indicator_cols, se_col, mask_cols, chains=2, warmup=400, samples=500, seed=0, positive=False):
    """indicator_cols: list of df columns loading on E. mask_cols: same length, 1/0 observed."""
    K = len(indicator_cols)
    ind = np.stack([df[c].values for c in indicator_cols], 1)           # (n,K)
    mask = np.stack([df[m].values for m in mask_cols], 1).astype(float) # (n,K)
    ind = np.where(mask > 0, np.nan_to_num(ind), 0.0)
    mpra = df["mpra"].values[:, None]
    obs = jnp.asarray(np.concatenate([ind, mpra], 1))
    se = jnp.asarray(df[se_col].values)
    mask = jnp.asarray(mask)
    names = [c.split("_")[0] for c in indicator_cols]
    model = make_model(names, positive=positive)
    draws = []
    n_div = 0
    for c in range(chains):
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup,
                    num_samples=samples, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed + c), obs, se, mask)
        s = mcmc.get_samples()
        draws.append(np.asarray(s["rho"]))
        ex = mcmc.get_extra_fields(["diverging"]) if False else {}
    rho = np.concatenate(draws)
    # cross-chain R-hat on rho
    arr = np.stack(draws)  # (chains, samples)
    m_chain = arr.mean(1); v_chain = arr.var(1, ddof=1)
    B = arr.shape[1] * m_chain.var(ddof=1); Wv = v_chain.mean()
    var_hat = (arr.shape[1] - 1) / arr.shape[1] * Wv + B / arr.shape[1]
    rhat = float(np.sqrt(var_hat / Wv)) if Wv > 0 else float("nan")
    return {"rho_mean": float(rho.mean()),
            "rho_hdi_lo": float(np.percentile(rho, 2.5)),
            "rho_hdi_hi": float(np.percentile(rho, 97.5)),
            "rho_p_gt_0": float((rho > 0).mean()),
            "rhat": rhat, "n": int(len(df))}


def main():
    df = pd.read_parquet("sensitivity_input.parquet")
    specs = []
    # SIGNED family (matches the published fit's estimand)
    specs.append(("signed: AG+BZ+MB (baseline)", ["ag_s", "bz_s", "mb_s"], "se_s",
                  ["ones", "ones", "mb_obs"]))
    specs.append(("signed: drop motifbreakR", ["ag_s", "bz_s"], "se_s", ["ones", "ones"]))
    specs.append(("signed: leave-out AlphaGenome", ["bz_s", "mb_s"], "se_s", ["ones", "mb_obs"]))
    specs.append(("signed: leave-out Borzoi", ["ag_s", "mb_s"], "se_s", ["ones", "mb_obs"]))
    # MAGNITUDE family (lets Enformer, which we only have as magnitude, enter)
    specs.append(("magnitude: AG+BZ", ["ag_abs_z", "bz_abs_z"], "se_s", ["ones", "ones"]))
    specs.append(("magnitude: AG+BZ+Enformer", ["ag_abs_z", "bz_abs_z", "enf_abs_z"], "se_s",
                  ["ones", "ones", "ones"]))
    specs.append(("magnitude: AG+BZ+Enformer+MB", ["ag_abs_z", "bz_abs_z", "enf_abs_z", "mb_abs_z"],
                  "se_s", ["ones", "ones", "ones", "mb_obs"]))

    df = df.copy()
    df["ones"] = 1.0
    # magnitude reporter column
    df["mpra_signed"] = df["mpra_s"]
    rows = []
    for label, inds, se_col, masks in specs:
        d = df.copy()
        d["mpra"] = d["mpra_s"] if label.startswith("signed") else d["mpra_abs_z"]
        r = fit(d, inds, se_col, masks, positive=label.startswith('magnitude'))
        r["spec"] = label
        rows.append(r)
        print(json.dumps({k: r[k] for k in ["spec", "rho_mean", "rho_hdi_lo", "rho_hdi_hi", "rhat"]}))
    out = pd.DataFrame(rows)[["spec", "n", "rho_mean", "rho_hdi_lo", "rho_hdi_hi", "rho_p_gt_0", "rhat"]]
    out.to_csv("sensitivity_results.csv", index=False)
    print("SAVED sensitivity_results.csv")


if __name__ == "__main__":
    main()
