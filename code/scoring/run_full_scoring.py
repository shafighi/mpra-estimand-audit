"""Day 2 driver: score the full biallelic benchmark set with AlphaGenome and
extract one summary scalar per variant. Written fresh for the hackathon.

Per-variant summary = mean allelic raw_score over brain-relevant tracks, computed
separately per output type (DNASE/ATAC/CAGE/CHIP_HISTONE) plus a combined
accessibility scalar (DNASE+ATAC+CAGE) — the endogenous analogue of episomal
reporter activity. Full per-track tidy tables stay on disk in cache/ (via
score_alphagenome); here we keep only the compact summary to merge with the MPRA.
"""
from __future__ import annotations
import os, sys, time, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_alphagenome as sa

# Curated CNS/neural biosamples present in AlphaGenome tracks (developing-cortex context).
BRAIN_BIOSAMPLES = {
    "astrocyte", "astrocyte of the cerebellum", "astrocyte of the cerebral cortex",
    "astrocyte of the hippocampus", "astrocyte of the spinal cord", "brain",
    "brain microvascular endothelial cell", "brain pericyte", "cerebellar cortex",
    "cerebrospinal fluid", "dorsolateral prefrontal cortex", "ecto neural progenitor cell",
    "frontal cortex", "glutamatergic neuron", "layer of hippocampus", "motor neuron",
    "neural cell", "neural crest cell", "neural progenitor cell", "neuron",
    "neuronal stem cell", "oligodendrocyte precursor cell", "perineural cell",
    "smooth muscle cell of the brain vasculature",
}
ACCESS_TYPES = ("DNASE", "ATAC", "CAGE")  # chromatin accessibility + transcription initiation


def summarize_variant(tidy: pd.DataFrame, variant_id: str) -> dict:
    """One compact row per variant: brain-track mean raw_score per output type,
    all-track mean per output type, plus track counts."""
    out = {"variant_id": variant_id}
    brain = tidy[tidy["biosample_name"].isin(BRAIN_BIOSAMPLES)]
    for ot in ("DNASE", "ATAC", "CAGE", "CHIP_HISTONE"):
        b = brain[brain["output_type"] == ot]["raw_score"]
        a = tidy[tidy["output_type"] == ot]["raw_score"]
        out[f"ag_brain_{ot}"] = float(b.mean()) if len(b) else np.nan
        out[f"ag_all_{ot}"] = float(a.mean()) if len(a) else np.nan
        out[f"n_brain_{ot}"] = int(len(b))
    # Combined accessibility scalar (primary model_effect): mean over brain DNASE+ATAC+CAGE tracks.
    acc = brain[brain["output_type"].isin(ACCESS_TYPES)]["raw_score"]
    out["ag_brain_access"] = float(acc.mean()) if len(acc) else np.nan
    out["n_brain_total"] = int(len(brain))
    return out


# Settled decision (prior session): 100KB context — MPRA is episomal/local, so
# distal 1MB context is not the right analogue and is ~6x slower for no gain here.
from alphagenome.models import dna_client as _dc
SCORING_WIDTH = _dc.SEQUENCE_LENGTH_100KB


def run(variants_path: str, out_path: str, api_key: str | None = None,
        checkpoint_every: int = 100, width: int = SCORING_WIDTH):
    vt = pd.read_parquet(variants_path)
    client = sa.get_client(api_key)
    rows, t0 = [], time.time()
    for i, r in enumerate(vt.itertuples(index=False)):
        tidy = sa.score_one(client, r.chrom, int(r.pos), r.ref, r.alt,
                            name=str(r.variant_id), width=width)
        rows.append(summarize_variant(tidy, str(r.variant_id)))
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_parquet(out_path)
            el = time.time() - t0
            print(f"  {i+1}/{len(vt)}  ({el/ (i+1):.2f}s/var, {el/60:.1f} min elapsed)", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_parquet(out_path)
    print(f"DONE {len(summ)} variants -> {out_path}", flush=True)
    return summ


if __name__ == "__main__":
    vp = sys.argv[1] if len(sys.argv) > 1 else "mpra_arbiter/data/processed/variant_table_biallelic.parquet"
    op = sys.argv[2] if len(sys.argv) > 2 else "mpra_arbiter/data/processed/alphagenome_scores.parquet"
    run(vp, op)
