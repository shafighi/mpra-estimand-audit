"""Day 4b: fetch base-level phyloP100way conservation at each variant position.

UCSC REST (api.genome.ucsc.edu, allowlisted). phyloP is an independent, sequence-
level conservation signal — NOT confounded by the brain-QTL ascertainment that
saturates the eQTL/GWAS columns. Cached to disk. hg38, 0-based half-open.
"""
from __future__ import annotations
import time, requests, pandas as pd

UCSC = "https://api.genome.ucsc.edu/getData/track"


def phylop_at(chrom, pos, session, track="phyloP100way"):
    """pos is 1-based (VCF-style). UCSC is 0-based half-open -> start=pos-1."""
    try:
        r = session.get(UCSC, params={"genome": "hg38", "track": track,
                                       "chrom": chrom, "start": int(pos) - 1, "end": int(pos)},
                        timeout=30)
        if r.status_code == 200:
            items = r.json().get(track, [])
            if items:
                return float(items[0]["value"])
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def fetch_table(df, out_path, chrom_col="chrom", pos_col="pos", id_col="variant_id",
                pause=0.02, checkpoint_every=250):
    s = requests.Session()
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        v = phylop_at(getattr(r, chrom_col), getattr(r, pos_col), s)
        rows.append({id_col: getattr(r, id_col), "phylop": v})
        time.sleep(pause)
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_parquet(out_path)
            print(f"  phyloP {i+1}/{len(df)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(out_path)
    print(f"DONE phyloP {len(out)} -> {out_path}", flush=True)
    return out
