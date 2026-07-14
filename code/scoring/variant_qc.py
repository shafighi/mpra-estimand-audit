"""Day-1 gate: verify a variant table is analysis-ready before any scoring.

The single most dangerous silent bug in a variant-effect benchmark is an
allele-orientation / build mismatch that inverts effect signs. This module:
  1. checks the tidy schema and coordinate convention,
  2. normalizes chromosome naming (chr-prefix),
  3. fetches the hg38 reference base at each position (UCSC REST primary,
     Ensembl REST fallback; both allowlisted) and confirms it equals the
     table's `ref` allele; flags mismatches,
  4. reports anything that should be dropped or lifted over.

Run this BEFORE score_alphagenome.score_table(). No API key needed.
"""
from __future__ import annotations
import time
import requests
import pandas as pd

ENSEMBL = "https://rest.ensembl.org"
UCSC = "https://api.genome.ucsc.edu"
REQUIRED = ["variant_id", "chrom", "pos", "ref", "alt"]


def check_schema(df: pd.DataFrame) -> list[str]:
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems
    if df["pos"].isna().any():
        problems.append("null positions present")
    if not df["ref"].str.match(r"^[ACGTacgt]+$").all():
        problems.append("non-ACGT ref alleles present")
    if not df["alt"].str.match(r"^[ACGTacgt]+$").all():
        problems.append("non-ACGT alt alleles present")
    dup = df.duplicated(["chrom", "pos", "ref", "alt"]).sum()
    if dup:
        problems.append(f"{dup} duplicate variant rows")
    return problems


def norm_chrom(c: str) -> str:
    c = str(c)
    return c if c.startswith("chr") else "chr" + c


def hg38_ref_base(chrom: str, pos: int, ref_len: int = 1,
                  session: requests.Session | None = None) -> str | None:
    """Fetch the hg38 reference base(s) at a 1-based genomic position.

    Primary: UCSC REST (api.genome.ucsc.edu) — 0-based half-open, chr-prefixed,
    reliable and allowlisted. Fallback: Ensembl REST (1-based, no chr-prefix).
    Input `pos` is 1-based (VCF-style); we convert per API.
    """
    s = session or requests
    c = norm_chrom(chrom)
    # --- UCSC (primary): 0-based half-open -> start = pos-1, end = pos-1+ref_len ---
    try:
        u = (f"{UCSC}/getData/sequence?genome=hg38;chrom={c};"
             f"start={pos - 1};end={pos - 1 + ref_len}")
        r = s.get(u, timeout=20)
        if r.status_code == 200:
            dna = r.json().get("dna")
            if dna:
                return dna.upper()
    except (requests.RequestException, ValueError):
        pass
    # --- Ensembl (fallback): 1-based inclusive, no chr-prefix ---
    try:
        reg = f"{c.replace('chr','')}:{pos}..{pos + ref_len - 1}"
        r = s.get(f"{ENSEMBL}/sequence/region/human/{reg}",
                  headers={"Content-Type": "text/plain"}, timeout=20)
        if r.status_code == 200 and r.text and r.text[0] in "ACGTNacgtn":
            return r.text.strip().upper()
    except requests.RequestException:
        pass
    return None


def check_ref_alleles(df: pd.DataFrame, assume_one_based: bool = True,
                      pause: float = 0.05) -> pd.DataFrame:
    """Return df with added columns: hg38_base, ref_match. Mismatches are the
    rows to investigate (wrong build, wrong strand, or 0-vs-1-based off-by-one)."""
    out = df.copy()
    sess = requests.Session()
    bases, matches = [], []
    for _, row in out.iterrows():
        pos = int(row["pos"]) + (0 if assume_one_based else 1)  # Ensembl is 1-based
        b = hg38_ref_base(norm_chrom(row["chrom"]), pos, len(str(row["ref"])), sess)
        bases.append(b)
        matches.append(None if b is None else (b == str(row["ref"]).upper()))
        time.sleep(pause)
    out["hg38_base"] = bases
    out["ref_match"] = matches
    return out


def summarize(checked: pd.DataFrame) -> dict:
    n = len(checked)
    matched = int((checked["ref_match"] == True).sum())
    mism = int((checked["ref_match"] == False).sum())
    unk = int(checked["ref_match"].isna().sum())
    return {"n": n, "ref_match": matched, "ref_mismatch": mism,
            "unresolved": unk, "match_rate": round(matched / n, 3) if n else 0.0}


if __name__ == "__main__":
    import sys
    df = pd.read_parquet(sys.argv[1]) if len(sys.argv) > 1 else None
    if df is None:
        print("usage: python variant_qc.py <variants.parquet>")
    else:
        print("schema problems:", check_schema(df) or "none")
        checked = check_ref_alleles(df)
        print(summarize(checked))
