#!/usr/bin/env python
import sys, os, numpy as np, pandas as pd, torch, pysam
from enformer_pytorch import Enformer

FASTA = "/mnt/scratche/slow/fmlab/darvis01/borzoi/refs/hg38.fa"
SEQLEN = 196608
BIN = 128
N_CENTER_BINS = 3   # ~384bp around the variant
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA not available — refusing to run on CPU (would exceed wall limit). "
             "Check torch/driver compatibility (need cu121 torch for this node's driver).")
dev = "cuda"
print("device", dev, "GPU", torch.cuda.get_device_name(0), flush=True)

fa = pysam.FastaFile(FASTA)
chroms = set(fa.references)

_LUT = np.full(256, -1, dtype=np.int64)
for _b,_j in {"A":0,"C":1,"G":2,"T":3,"a":0,"c":1,"g":2,"t":3}.items():
    _LUT[ord(_b)] = _j
def one_hot(seq):
    idx = _LUT[np.frombuffer(seq.encode("ascii","replace"), dtype=np.uint8)]
    a = np.zeros((len(seq),4), dtype=np.float32)
    valid = idx >= 0
    a[np.arange(len(seq))[valid], idx[valid]] = 1.0
    return a

model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").to(dev).eval()

# brain track selection from Enformer human targets
import urllib.request
tgt_path = "targets_human.txt"
if not os.path.exists(tgt_path):
    urllib.request.urlretrieve(
      "https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt", tgt_path)
tgt = pd.read_csv(tgt_path, sep="\t")
desc = tgt["description"].astype(str).str.lower()
is_brain = desc.str.contains("brain|cortex|neuron|astrocy|cerebell|hippocamp|glia", regex=True)
assay = tgt["description"].astype(str)
# split by assay type using the 'description' prefix (CAGE:, DNASE:, CHIP:)
brain_idx = np.where(is_brain.values)[0]
cage_brain = np.array([i for i in brain_idx if desc.iloc[i].startswith("cage")])
dnase_brain= np.array([i for i in brain_idx if desc.iloc[i].startswith("dnase")])
print("brain tracks:", len(brain_idx), "cage_brain", len(cage_brain), "dnase_brain", len(dnase_brain), flush=True)

vt = pd.read_csv("variants.csv")
out = []
center = 896//2  # enformer outputs 896 bins
lo = center - N_CENTER_BINS//2 - 1
hi = center + N_CENTER_BINS//2 + 1

@torch.no_grad()
def predict(seq_oh):
    x = torch.from_numpy(seq_oh).to(dev)
    y = model(x)["human"]  # (896, 5313)
    return y.cpu().numpy()

for n,(_,r) in enumerate(vt.iterrows()):
    ch, pos, ref, alt = r["chrom"], int(r["pos"]), r["ref"], r["alt"]
    if ch not in chroms:
        out.append((r["rsid"], np.nan, np.nan, np.nan)); continue
    start = pos-1 - SEQLEN//2
    end = start + SEQLEN
    if start < 0 or end > fa.get_reference_length(ch):
        out.append((r["rsid"], np.nan, np.nan, np.nan)); continue
    seq = fa.fetch(ch, start, end)
    ref_oh = one_hot(seq)
    vidx = SEQLEN//2  # variant at center
    # sanity: reference base
    alt_oh = ref_oh.copy()
    m = {"A":0,"C":1,"G":2,"T":3}
    if alt.upper() in m:
        alt_oh[vidx,:] = 0; alt_oh[vidx, m[alt.upper()]] = 1.0
    yr = predict(ref_oh); ya = predict(alt_oh)
    diff = ya - yr  # (896,5313)
    seg = diff[lo:hi,:].mean(axis=0)  # mean over center bins per track
    e_all = float(np.abs(seg[list(cage_brain)+list(dnase_brain)]).mean()) if len(cage_brain)+len(dnase_brain)>0 else np.nan
    e_cage = float(np.abs(seg[cage_brain]).mean()) if len(cage_brain)>0 else np.nan
    e_dnase= float(np.abs(seg[dnase_brain]).mean()) if len(dnase_brain)>0 else np.nan
    out.append((r["rsid"], e_all, e_cage, e_dnase))
    if n % 200 == 0: print("done", n, "/", len(vt), flush=True)

res = pd.DataFrame(out, columns=["rsid","enformer_brain_abs","enformer_cage_brain","enformer_dnase_brain"])
res.to_csv("enformer_scores.csv", index=False)
print("WROTE enformer_scores.csv", res.shape, "non-null", res["enformer_brain_abs"].notna().sum(), flush=True)
