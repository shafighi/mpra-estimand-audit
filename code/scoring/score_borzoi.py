"""Day 6: score all variants with Borzoi (ref vs alt), brain-track delta.

Runs on the cluster (L40S). Reads variants.parquet (variant_id,chrom,pos,ref,alt),
writes borzoi_scores.parquet with per-variant effect scalars. Checkpoints every 200.
"""
import os, sys, time
import numpy as np, pandas as pd, torch, pysam
from borzoi_pytorch import Borzoi
from borzoi_pytorch.pytorch_borzoi_model import TRACKS_DF  # track metadata

REF = "/mnt/scratche/slow/fmlab/darvis01/borzoi/refs/hg38.fa"
# Persistent output dir (survives job-tracking cancellation / session resets).
OUTDIR = "/mnt/scratche/slow/fmlab/darvis01/borzoi/out"
os.makedirs(OUTDIR, exist_ok=True)
STABLE = os.path.join(OUTDIR, "borzoi_scores.parquet")
L = 524288; HALF = L // 2
M = {"A": 0, "C": 1, "G": 2, "T": 3}

def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    idx = np.array([M.get(ch, -1) for ch in s])
    ok = idx >= 0
    x[idx[ok], np.where(ok)[0]] = 1.0
    return x

def main():
    df = pd.read_parquet("variants.parquet")
    fa = pysam.FastaFile(REF)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0").to(dev).eval()

    # brain-relevant track mask: DNASE/CAGE/RNA in brain/neuron/cortex biosamples
    td = TRACKS_DF.copy()
    desc = (td["description"].astype(str) + " " + td.get("assay", pd.Series([""]*len(td))).astype(str)).str.lower()
    brain = desc.str.contains("brain|cortex|neuron|neural|astrocyte|glia|cerebr")
    assay = desc.str.contains("dnase|cage|rna|atac")
    mask = (brain & assay).values
    if mask.sum() == 0:  # fallback: any brain track
        mask = brain.values
    print(f"brain-track mask: {mask.sum()} of {len(td)} tracks", flush=True)

    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples(index=False)):
        try:
            start = int(r.pos) - 1 - HALF; end = start + L
            seq = fa.fetch(r.chrom, max(0, start), end).upper()
            if len(seq) < L: seq = seq + "N" * (L - len(seq))
            ci = int(r.pos) - 1 - start
            ref_oh = onehot(seq)
            alt_seq = seq[:ci] + r.alt + seq[ci+1:]
            alt_oh = onehot(alt_seq)
            with torch.no_grad():
                yr = model(torch.tensor(ref_oh[None]).to(dev)).cpu().numpy()[0]  # (T, bins)
                ya = model(torch.tensor(alt_oh[None]).to(dev)).cpu().numpy()[0]
            d = ya - yr                       # (T, bins)
            db = d[mask]                       # brain tracks only
            # center bins around the variant (Borzoi crops edges; use central 10 bins)
            cbin = db.shape[1] // 2
            win = db[:, cbin-5:cbin+5]
            rows.append({
                "variant_id": r.variant_id,
                "borzoi_signed": float(win.mean()),
                "borzoi_abs": float(np.abs(win).mean()),
                "borzoi_max_abs": float(np.abs(win).max()),
                "center_ref_ok": bool(seq[ci] == r.ref),
                "n_brain_tracks": int(mask.sum()),
            })
        except Exception as e:
            rows.append({"variant_id": r.variant_id, "borzoi_signed": np.nan,
                         "borzoi_abs": np.nan, "borzoi_max_abs": np.nan,
                         "center_ref_ok": False, "n_brain_tracks": int(mask.sum())})
        if (i+1) % 200 == 0:
            pd.DataFrame(rows).to_parquet(STABLE); pd.DataFrame(rows).to_parquet("borzoi_scores.parquet")
            el = time.time()-t0
            print(f"{i+1}/{len(df)}  {el/ (i+1):.2f}s/var  eta {el/(i+1)*(len(df)-i-1)/60:.1f}min", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(STABLE); out.to_parquet("borzoi_scores.parquet")
    print(f"DONE {len(out)} variants, ref_ok={out.center_ref_ok.mean():.3f}, wall={(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
