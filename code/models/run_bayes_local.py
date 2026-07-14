#!/usr/bin/env python
"""Local run of the hierarchical Bayesian factor model on the pre-subsetted
Stim8hr top-1500-variance effect matrix. Produces calibrated activation-program
loadings, novel-regulator nominations, and known-regulator recovery validation."""
import os, json, time
import numpy as np
T0=time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

K=25; STEPS=6000; NPOST=800; SEED=0
ACT_SIGNATURE = ["IL2","IL2RA","IL2RB","IFNG","TNF","CD69","TNFRSF9","TNFRSF4","MKI67",
                 "GZMB","PRF1","MYC","CD40LG","ICOS","LTA","CSF2","IL2RG","TNFRSF18","NR4A1","EGR2"]
KNOWN_REG = ["CD28","LCK","ZAP70","LAT","PLCG1","PRKCQ","CARD11","MALT1","BCL10","LCP2",
             "NFKB1","REL","RELA","NFATC1","NFATC2","JUN","FOS","JUNB","MYC","IRF4","BATF",
             "STAT5A","STAT5B","IL2RA","VAV1","ITK","TRAF6","CHUK","IKBKB",
             "PTPN6","PTPN22","CBLB","CTLA4","PDCD1","TNFAIP3","NR4A1","NR4A2","NR4A3","TOX","EGR2","UBASH3A"]

d = np.load("data/effect_Stim8hr_top1500.npz", allow_pickle=True)
M = np.nan_to_num(d["Z"].astype(np.float32))
panel_genes = d["genes"].astype(str); targets = d["targets"].astype(str)
M = ((M - M.mean(0))/(M.std(0)+1e-6)).astype(np.float32)
log("panel matrix:", M.shape, "K=", K)

import jax, jax.numpy as jnp
from jax import random
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
numpyro.set_host_device_count(1)

def model(M):
    n,p = M.shape
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.ones(K)))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    W = numpyro.sample("W", dist.Normal(jnp.zeros((n,K)),1.0).to_event(2))
    H = numpyro.sample("H", dist.Normal(jnp.zeros((K,p)), tau[:,None]).to_event(2))
    numpyro.sample("obs", dist.Normal(W@H, sigma).to_event(2), obs=M)

log("fitting SVI ...")
guide = autoguide.AutoNormal(model)
svi = SVI(model, guide, numpyro.optim.Adam(5e-3), Trace_ELBO())
res = svi.run(random.PRNGKey(SEED), STEPS, jnp.asarray(M), progress_bar=False)
log("final ELBO:", float(res.losses[-1]))
post = guide.sample_posterior(random.PRNGKey(SEED+1), res.params, sample_shape=(NPOST,))
W_s = np.asarray(post["W"]); H_s = np.asarray(post["H"]); tau_m = np.asarray(post["tau"]).mean(0)
H_mean = H_s.mean(0)
log("posterior:", W_s.shape)

sig_idx = np.array([i for i,g in enumerate(panel_genes) if g in set(ACT_SIGNATURE)])
zs = [ (H_mean[k][sig_idx].mean()-H_mean[k].mean())/(H_mean[k].std()+1e-6) for k in range(K)] if len(sig_idx)>=3 else [0]*K
kstar = int(np.argmax(np.abs(zs)))
z_sig = float(zs[kstar]); orient = 1.0 if z_sig>=0 else -1.0
log("activation factor k*=",kstar,"sig genes in panel=",len(sig_idx),"z=",round(z_sig,3),"orient",orient)

Wk = orient*W_s[:,:,kstar]
w_mean=Wk.mean(0); w_lo,w_hi=np.percentile(Wk,[2.5,97.5],axis=0)
confident=(w_lo>0)|(w_hi<0)
import pandas as pd
tab=pd.DataFrame({"perturbed_gene":targets,"loading_mean":w_mean,"loading_lo":w_lo,
                  "loading_hi":w_hi,"confident":confident})
tab["abs_loading"]=tab["loading_mean"].abs()
tab["known_regulator"]=tab["perturbed_gene"].isin(set(KNOWN_REG))
tab=tab.sort_values("abs_loading",ascending=False).reset_index(drop=True)
nom=tab[(tab["confident"])&(~tab["known_regulator"])].head(30).copy()
nom["predicted_direction"]=np.where(nom["loading_mean"]>0,"promotes activation program","suppresses activation program")

from scipy.stats import mannwhitneyu
km=tab["known_regulator"].values; absL=tab["abs_loading"].values
kp=int(km.sum())
try: U,pval=mannwhitneyu(absL[km],absL[~km],alternative="greater")
except Exception: pval=float("nan")

summary={"condition":"Stim8hr","K":K,"n_genes_panel":int(M.shape[1]),"n_perturbations":int(M.shape[0]),
  "activation_factor":kstar,"activation_signature_z":round(z_sig,3),"signature_genes_in_panel":int(len(sig_idx)),
  "final_elbo":float(res.losses[-1]),"tau_top5":[round(float(x),3) for x in np.sort(tau_m)[::-1][:5]],
  "n_confident_loadings":int(tab["confident"].sum()),"known_regulators_present":kp,
  "frac_known_confident":round(float(tab.loc[km,"confident"].mean()) if kp else float("nan"),3),
  "frac_all_confident":round(float(tab["confident"].mean()),3),
  "validation_mannwhitney_p_known_gt_rest":float(pval),
  "top_novel_nominations":nom[["perturbed_gene","loading_mean","loading_lo","loading_hi","predicted_direction"]].head(15).to_dict("records")}
tab.to_csv("out/activation_loadings_all.csv",index=False)
nom.to_csv("out/novel_regulator_nominations.csv",index=False)
json.dump(summary,open("out/bayes_model_summary.json","w"),indent=2,default=float)
np.savez_compressed("out/posterior_activation.npz",w_mean=w_mean,w_lo=w_lo,w_hi=w_hi,
                    targets=targets,kstar=kstar,tau_mean=tau_m,elbo=np.asarray(res.losses))
log("SUMMARY:",json.dumps(summary,indent=2,default=float))
log("DONE")
