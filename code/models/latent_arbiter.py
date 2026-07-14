"""
Latent-variable generative arbiter for MPRA-Arbiter (from scratch, NumPyro).

Replaces the Day-3 rule-based verdict map with a calibrated generative model.

Generative structure (per variant i):
  (E_i, A_i) ~ BivariateNormal(0, [[1, rho],[rho, 1]])
     E_i = latent endogenous/sequence effect (signed)
     A_i = latent integrated-reporter (lentiMPRA) effect (signed)
     rho = the calibrated model<->assay agreement  <-- headline quantity, with CI

  Signed endogenous observations load on E_i (all z-scored to unit variance,
  so a loading lam IS that source's correlation with the latent):
     ag_s_i   ~ Normal(b_ag + lam_ag*E_i, sqrt(1-lam_ag^2))   lam_ag>0 anchors E
     bz_s_i   ~ Normal(b_bz + lam_bz*E_i, sqrt(1-lam_bz^2))
     mb_s_i   ~ Normal(b_mb + lam_mb*E_i, sqrt(1-lam_mb^2))   [masked: 702 variants]

  Assay observation loads on A_i, heteroscedastic via the limma SE:
     mpra_s_i ~ Normal(b_mp + lam_mp*A_i, se_s_i)             lam_mp>0 anchors A

INFERENCE: the per-variant latents (E_i, A_i) are marginalized ANALYTICALLY.
The whole core is linear-Gaussian, so the marginal of the observation vector
o_i = [ag, bz, (mb), mpra]_i is MultivariateNormal(mu, W Sigma_EA W^T + D_i),
where W maps [E,A] to observation means and D_i is the residual diagonal
(the mpra entry = se_i^2, heteroscedastic). rho is identified through the
cross-source covariance Cov(ag, mpra) = lam_ag*lam_mp*rho -- i.e. rho IS the
calibrated agreement between the endogenous and assay latents. Marginalizing
removes the 8000 free per-variant latents (and their loading<->scale ridge),
so NUTS converges cleanly. Per-variant (E_i, A_i) posteriors for attribution
are recovered post-hoc by Gaussian conditioning.

phyloP is NOT a term in this fitted model -- it is held out as orthogonal
validation (an earlier free-latent variant gave it a negligible ~0.07 loading;
it was demoted to a held-out conservation check here), consistent with its
established role in the project
(Day 4b conservation check).
"""
import json, argparse
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def _sigma_EA(rho):
    return jnp.array([[1.0, rho], [rho, 1.0]])


def _lowrank_loglik(x, dvar, W, S):
    """log N(x | 0, diag(dvar) + W S W^T) per row via Woodbury / matrix
    determinant lemma. x:(n,d), dvar:(n,d) residual variances, W:(d,2), S:(2,2).
    Only a 2x2 inverse per row -- no d-dim Cholesky, so it stays fast at n~4000."""
    n, d = x.shape
    Dinv = 1.0 / dvar                                      # (n,d)
    Sinv = jnp.linalg.inv(S)                               # (2,2)
    WtDinvW = jnp.einsum("dk,nd,dl->nkl", W, Dinv, W)      # (n,2,2)
    M = Sinv[None] + WtDinvW                               # (n,2,2)
    Minv = jnp.linalg.inv(M)                               # (n,2,2)
    WtDinvx = jnp.einsum("dk,nd,nd->nk", W, Dinv, x)       # (n,2)
    quad = jnp.sum(x * Dinv * x, 1) - jnp.einsum("nk,nkl,nl->n", WtDinvx, Minv, WtDinvx)
    logdet = (jnp.sum(jnp.log(dvar), 1)
              + jnp.linalg.slogdet(M)[1]
              + jnp.linalg.slogdet(S)[1])
    return -0.5 * (d * jnp.log(2 * jnp.pi) + logdet + quad)


def model(o_full, se_full, o_nomb, se_nomb):
    """o_full: (n1,4)=[ag,bz,mb,mpra] with mb observed; o_nomb: (n2,3)=[ag,bz,mpra]."""
    rho = numpyro.sample("rho", dist.Uniform(-0.99, 0.99))
    lam_ag = numpyro.sample("lam_ag", dist.Beta(2.0, 2.0))     # (0,1) anchors E
    lam_bz = numpyro.sample("lam_bz", dist.Uniform(-0.99, 0.99))
    lam_mb = numpyro.sample("lam_mb", dist.Uniform(-0.99, 0.99))
    lam_mp = numpyro.sample("lam_mp", dist.Beta(2.0, 2.0))     # (0,1) anchors A
    b_ag = numpyro.sample("b_ag", dist.Normal(0.0, 0.5))
    b_bz = numpyro.sample("b_bz", dist.Normal(0.0, 0.5))
    b_mb = numpyro.sample("b_mb", dist.Normal(0.0, 0.5))
    b_mp = numpyro.sample("b_mp", dist.Normal(0.0, 0.5))
    numpyro.deterministic("sigma_ag", jnp.sqrt(1 - lam_ag**2))
    numpyro.deterministic("sigma_bz", jnp.sqrt(1 - lam_bz**2))
    numpyro.deterministic("sigma_mb", jnp.sqrt(1 - lam_mb**2))

    S = _sigma_EA(rho)
    d_ag = 1 - lam_ag**2; d_bz = 1 - lam_bz**2; d_mb = 1 - lam_mb**2
    eps = 1e-6

    # --- group with motifbreakR observed: [ag, bz, mb, mpra] ---
    n1 = o_full.shape[0]
    W1 = jnp.array([[lam_ag, 0.0], [lam_bz, 0.0], [lam_mb, 0.0], [0.0, lam_mp]])
    mu1 = jnp.array([b_ag, b_bz, b_mb, b_mp])
    dvar1 = jnp.stack([jnp.broadcast_to(d_ag, (n1,)), jnp.broadcast_to(d_bz, (n1,)),
                       jnp.broadcast_to(d_mb, (n1,)), se_full**2 + eps], 1)
    numpyro.factor("ll1", _lowrank_loglik(o_full - mu1, dvar1, W1, S).sum())

    # --- group without motifbreakR: [ag, bz, mpra] ---
    n2 = o_nomb.shape[0]
    W2 = jnp.array([[lam_ag, 0.0], [lam_bz, 0.0], [0.0, lam_mp]])
    mu2 = jnp.array([b_ag, b_bz, b_mp])
    dvar2 = jnp.stack([jnp.broadcast_to(d_ag, (n2,)), jnp.broadcast_to(d_bz, (n2,)),
                       se_nomb**2 + eps], 1)
    numpyro.factor("ll2", _lowrank_loglik(o_nomb - mu2, dvar2, W2, S).sum())


def posterior_latents(post, t, thin=5):
    """Recover per-variant (E,A) posterior mean by Gaussian conditioning.
    Vectorized over variants (batched inversion); draws thinned for speed."""
    ag = t.ag_s.values; bz = t.bz_s.values; mp = t.mpra_s.values
    se = t.se_s.values; mb = t.mb_s.values; mb_obs = t.mb_obs.values.astype(bool)
    D = len(post["rho"]); N = len(t)
    keep = np.arange(0, D, thin)
    E_draws = np.zeros((len(keep), N)); A_draws = np.zeros((len(keep), N))
    for gi, grp_mask in enumerate([mb_obs, ~mb_obs]):
        idx = np.where(grp_mask)[0]
        if idx.size == 0:
            continue
        has_mb = (gi == 0)
        for kk, d in enumerate(keep):
            rho = post["rho"][d]
            la, lb, lm, lp = (post["lam_ag"][d], post["lam_bz"][d],
                              post["lam_mb"][d], post["lam_mp"][d])
            ba, bb, bm, bp = (post["b_ag"][d], post["b_bz"][d],
                              post["b_mb"][d], post["b_mp"][d])
            S = np.array([[1.0, rho], [rho, 1.0]])
            if has_mb:
                W = np.array([[la, 0], [lb, 0], [lm, 0], [0, lp]])
                mu = np.array([ba, bb, bm, bp])
                dvec = np.array([1-la**2, 1-lb**2, 1-lm**2, 0.0])
                obs = np.stack([ag[idx], bz[idx], mb[idx], mp[idx]], 1)
                mp_pos = 3
            else:
                W = np.array([[la, 0], [lb, 0], [0, lp]])
                mu = np.array([ba, bb, bp])
                dvec = np.array([1-la**2, 1-lb**2, 0.0])
                obs = np.stack([ag[idx], bz[idx], mp[idx]], 1)
                mp_pos = 2
            base = W @ S @ W.T
            cross = S @ W.T                                  # (2,dobs)
            dobs = base.shape[0]
            cov = np.broadcast_to(base + np.diag(dvec), (idx.size, dobs, dobs)).copy()
            cov[:, mp_pos, mp_pos] += se[idx]**2
            cinv = np.linalg.inv(cov)                        # (n,dobs,dobs)
            resid = obs - mu                                 # (n,dobs)
            tmp = np.einsum("ndk,nk->nd", cinv, resid)       # (n,dobs)
            mpost = tmp @ cross.T                            # (n,2) conditional mean
            # conditional cov = S - cross cinv cross^T  (per variant, 2x2)
            ccov = S[None] - np.einsum("dk,nkl,ml->ndm", cross, cinv, cross)
            Lc = np.linalg.cholesky(ccov + 1e-8*np.eye(2)[None])
            eps = np.random.default_rng(d).standard_normal((idx.size, 2))
            samp = mpost + np.einsum("ndm,nm->nd", Lc, eps)  # one draw per param draw
            E_draws[kk, idx] = samp[:, 0]; A_draws[kk, idx] = samp[:, 1]
    return E_draws, A_draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="arbiter_input.parquet")
    ap.add_argument("--out_prefix", default="latent_arbiter")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = pd.read_parquet(args.input)
    mb_obs = t.mb_obs.values.astype(bool)
    tf = t[mb_obs]; tn = t[~mb_obs]
    o_full = jnp.asarray(np.stack([tf.ag_s, tf.bz_s, tf.mb_s, tf.mpra_s], 1))
    se_full = jnp.asarray(tf.se_s.values)
    o_nomb = jnp.asarray(np.stack([tn.ag_s, tn.bz_s, tn.mpra_s], 1))
    se_nomb = jnp.asarray(tn.se_s.values)

    # Run chains as independent single-chain MCMCs (multi-chain XLA compilation
    # stalls on this sandbox; single-chain is fast). Different PRNG seed per
    # chain; stack posteriors along a chain axis for cross-chain R-hat.
    scalar = ["rho","lam_ag","lam_bz","lam_mb","lam_mp",
              "b_ag","b_bz","b_mb","b_mp","sigma_ag","sigma_bz","sigma_mb"]
    chain_samps = []
    n_div = 0
    for c in range(args.chains):
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9),
                    num_warmup=args.warmup, num_samples=args.samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(args.seed + c), o_full, se_full, o_nomb, se_nomb,
                 extra_fields=("diverging",))
        chain_samps.append({k: np.asarray(v) for k, v in mcmc.get_samples().items()})
        n_div += int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    import arviz as az
    # stack scalar params into (chain, draw) arrays for arviz
    post_by_chain = {k: np.stack([cs[k] for cs in chain_samps], 0) for k in scalar}
    idata = az.from_dict(posterior=post_by_chain)
    summ = az.summary(idata, var_names=scalar, hdi_prob=0.95)
    lo_col = [c for c in summ.columns if c.startswith("hdi_") and c.endswith("%") and float(c[4:-1]) < 50][0]
    hi_col = [c for c in summ.columns if c.startswith("hdi_") and c.endswith("%") and float(c[4:-1]) > 50][0]
    max_rhat = float(summ["r_hat"].max())

    # pool draws across chains for per-variant recovery + rho posterior
    post = {k: np.concatenate([cs[k] for cs in chain_samps], 0) for k in chain_samps[0]}
    rho = post["rho"]
    E, A = posterior_latents(post, t)

    thr = 0.5
    pE = (np.abs(E) > thr).mean(0)
    pA = (np.abs(A) > thr).mean(0)
    Emag, Amag = np.abs(E), np.abs(A)
    p_concord   = ((Emag > thr) & (Amag > thr)).mean(0)
    p_endo_only = ((Emag > thr) & (Amag <= thr)).mean(0)
    p_rep_only  = ((Emag <= thr) & (Amag > thr)).mean(0)
    p_quiet     = ((Emag <= thr) & (Amag <= thr)).mean(0)
    stack = np.vstack([p_endo_only, p_rep_only, p_concord, p_quiet])
    labels = np.array(["endogenous-only","reporter-only","concordant","both-quiet"])
    hard = labels[stack.argmax(0)]

    out = pd.DataFrame({
        "variant_id": t.variant_id, "rsid": t.rsid, "is_dav": t.is_dav,
        "E_mean": E.mean(0), "E_sd": E.std(0),
        "A_mean": A.mean(0), "A_sd": A.std(0),
        "p_endo_moves": pE, "p_assay_moves": pA,
        "p_endogenous_only": p_endo_only, "p_reporter_only": p_rep_only,
        "p_concordant": p_concord, "p_both_quiet": p_quiet,
        "attribution_calibrated": hard,
    })
    out.to_parquet(f"{args.out_prefix}_pervariant.parquet", index=False)

    summary = {
        "n_variants": int(len(t)), "n_div": n_div, "max_rhat": max_rhat,
        "rho_mean": float(rho.mean()),
        "rho_hdi95": [float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))],
        "rho_p_gt_0": float((rho > 0).mean()),
        "loadings": {k: {"mean": float(summ.loc[k, "mean"]),
                         "hdi_lo": float(summ.loc[k, lo_col]),
                         "hdi_hi": float(summ.loc[k, hi_col]),
                         "r_hat": float(summ.loc[k, "r_hat"])} for k in scalar},
        "attribution_counts_all": out.attribution_calibrated.value_counts().to_dict(),
        "attribution_counts_dav":
            out[out.is_dav == 1].attribution_calibrated.value_counts().to_dict(),
    }
    json.dump(summary, open(f"{args.out_prefix}_summary.json", "w"), indent=2)
    az.to_netcdf(idata, f"{args.out_prefix}_idata.nc")
    np.savez(f"{args.out_prefix}_draws.npz",
             **{k: post[k] for k in scalar})
    print(json.dumps({"rho_mean": summary["rho_mean"], "rho_hdi95": summary["rho_hdi95"],
                      "max_rhat": max_rhat, "n_div": n_div,
                      "dav_attr": summary["attribution_counts_dav"]}, indent=2))


if __name__ == "__main__":
    main()
