"""Day 4: from-scratch Bayesian measurement-error model of MPRA allelic effects.

Written fresh for the hackathon (NumPyro/JAX, no ported sampler).

Setup: the lentiMPRA reports, per variant i, an allelic effect y_i (limma log2FC)
with a moderated standard error se_i. We treat y_i as a NOISY OBSERVATION of a
latent true effect theta_i:

    y_i ~ Normal(theta_i, se_i)              # measurement layer (se known)
    theta_i ~ regularized horseshoe          # sparsity: most effects ~0, few large

Regularized horseshoe (Piironen & Vehtari 2017), non-centered:
    tau         ~ HalfCauchy(tau0)           # global shrinkage
    lambda_i    ~ HalfCauchy(1)              # local shrinkage
    c2          ~ InvGamma(a, b)             # slab (caps large effects)
    lam_tilde_i = sqrt( c2 * lambda_i^2 / (c2 + tau^2 * lambda_i^2) )
    theta_i     = z_i * tau * lam_tilde_i,  z_i ~ Normal(0,1)

Outputs per variant: posterior mean effect (shrunken), posterior SD (calibrated),
and a reliability weight = 1 / posterior-variance. The reliability replaces the
ad-hoc FDR cutoff used in Day 3 with a principled measurement-confidence weight.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def model(se, y=None, tau0=0.01, slab_scale=0.5, slab_df=4):
    n = se.shape[0]
    # global + local shrinkage
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    with numpyro.plate("variants", n):
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n)))
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    # slab: c2 ~ InvGamma(slab_df/2, slab_df/2 * slab_scale^2)
    c2 = numpyro.sample("c2", dist.InverseGamma(slab_df / 2.0,
                                                slab_df / 2.0 * slab_scale ** 2))
    lam_tilde = jnp.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
    theta = numpyro.deterministic("theta", z * tau * lam_tilde)
    with numpyro.plate("obs", n):
        numpyro.sample("y", dist.Normal(theta, se), obs=y)


def fit(y, se, tau0=0.01, slab_scale=0.5, num_warmup=1000, num_samples=1000,
        num_chains=2, seed=0):
    y = jnp.asarray(np.asarray(y, float))
    se = jnp.asarray(np.asarray(se, float))
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), se=se, y=y, tau0=tau0, slab_scale=slab_scale)
    return mcmc


def summarize(mcmc):
    """Per-variant posterior mean, SD, 95% CI, reliability, and P(|theta|>0.1)."""
    th = np.asarray(mcmc.get_samples()["theta"])  # (draws, n)
    post_mean = th.mean(0)
    post_sd = th.std(0)
    lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
    reliability = 1.0 / np.clip(post_sd, 1e-6, None) ** 2
    p_effect = (np.abs(th) > 0.1).mean(0)   # posterior prob of a non-trivial effect
    excludes0 = (lo > 0) | (hi < 0)          # 95% CI excludes zero
    return dict(post_mean=post_mean, post_sd=post_sd, ci_lo=lo, ci_hi=hi,
                reliability=reliability, p_effect=p_effect, ci_excludes_0=excludes0)
