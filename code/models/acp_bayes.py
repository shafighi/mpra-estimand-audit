"""
Bayesian hierarchical Activity-Context-Propagation (ACP) model.

Y_i ~ Normal(mu_i, s_i)   with s_i the KNOWN SuSiE posterior_sd (measurement error)
mu_i = b0 + bM*M_i + sum_k bC[k]*C_ik + gamma*(M_i x context_favorability_i)

- Regularized horseshoe prior on the context coefficients bC and the interaction
  gamma (many correlated context features, expect most near zero).
- PIP enters as a per-observation likelihood weight (tempering): a low-PIP eQTL
  contributes less evidence about the effect. Implemented by scaling the
  observation precision by PIP (equivalently inflating s_i by 1/sqrt(pip)).

The posterior over gamma (the M x context interaction) is the key inference.
Written from scratch for this analysis (no reuse of prior samplers).
"""
import numpy as np, pandas as pd, json
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

numpyro.set_host_device_count(4)

def load_data(path="acp_table.parquet"):
    mg = pd.read_parquet(path)
    d = mg.dropna(subset=['M_mpra','Y_eqtl','Y_sd']).reset_index(drop=True)
    ctx_cols = ['ctx_ACCESS','ctx_COUPLING','ctx_CONSERV','ctx_REDUND']
    return d, ctx_cols

def model(M, C, fav, s_obs, pip, y=None):
    n, k = C.shape
    b0 = numpyro.sample("b0", dist.Normal(0., 0.5))
    bM = numpyro.sample("bM", dist.Normal(0., 1.))
    # regularized horseshoe on context coefs + interaction
    tau = numpyro.sample("tau", dist.HalfCauchy(0.1))                 # global shrinkage
    with numpyro.plate("ctx", k):
        lam = numpyro.sample("lam", dist.HalfCauchy(1.))              # local shrinkage
        bC  = numpyro.sample("bC", dist.Normal(0., 1.))
    bC = bC * lam * tau
    lam_g = numpyro.sample("lam_g", dist.HalfCauchy(1.))
    gamma = numpyro.sample("gamma", dist.Normal(0., 1.)) * lam_g * tau  # interaction, same shrinkage scale
    numpyro.deterministic("gamma_eff", gamma)

    # intrinsic scatter: biological / propagation noise NOT captured by the eQTL
    # measurement error. This is the scientifically central nuisance parameter --
    # it measures how much endogenous variance MPRA + context leave unexplained.
    sigma_extra = numpyro.sample("sigma_extra", dist.HalfNormal(0.1))

    mu = b0 + bM*M + jnp.dot(C, bC) + gamma*(M*fav)
    # total noise = known measurement error (PIP-tempered) + intrinsic scatter, in quadrature
    sigma_meas = s_obs / jnp.sqrt(jnp.clip(pip, 1e-3, 1.0))
    sigma_tot = jnp.sqrt(sigma_meas**2 + sigma_extra**2)
    numpyro.sample("y", dist.Normal(mu, sigma_tot), obs=y)

def run(seed=0, warmup=1500, samples=2000, chains=4):
    d, ctx_cols = load_data()
    M   = jnp.array(d['M_mpra'].values)
    C   = jnp.array(d[ctx_cols].values)
    fav = jnp.array(d['context_favorability'].values)
    s   = jnp.array(d['Y_sd'].values)
    pip = jnp.array(d['pip'].values)
    y   = jnp.array(d['Y_eqtl'].values)

    kernel = NUTS(model, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), M, C, fav, s, pip, y=y)

    # posterior predictive for calibration
    post = mcmc.get_samples()
    pred = Predictive(model, post)(jax.random.PRNGKey(seed+1), M, C, fav, s, pip)
    idata = az.from_numpyro(mcmc, posterior_predictive={"y": pred["y"][None]},
                            coords={"ctx": ctx_cols}, dims={"bC":["ctx"],"lam":["ctx"]})
    return d, ctx_cols, mcmc, idata

if __name__ == "__main__":
    d, ctx_cols, mcmc, idata = run()
    summ = az.summary(idata, var_names=["b0","bM","gamma_eff","bC","tau"])
    print(summ.to_string())
    idata.to_netcdf("acp_idata.nc")
    # key inference on gamma
    g = np.asarray(idata.posterior["gamma_eff"]).ravel()
    bM = np.asarray(idata.posterior["bM"]).ravel()
    out = dict(
        n=int(len(d)),
        gamma_mean=float(g.mean()), gamma_sd=float(g.std()),
        gamma_hdi=[float(np.percentile(g,2.5)), float(np.percentile(g,97.5))],
        p_gamma_gt0=float((g>0).mean()),
        bM_mean=float(bM.mean()), bM_hdi=[float(np.percentile(bM,2.5)),float(np.percentile(bM,97.5))],
        max_rhat=float(summ['r_hat'].max()), min_ess=float(summ['ess_bulk'].min()),
    )
    json.dump(out, open("acp_bayes_summary.json","w"), indent=2)
    print(json.dumps(out, indent=2))
