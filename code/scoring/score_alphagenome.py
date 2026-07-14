"""Score single-base variants with AlphaGenome, in native genomic context.

Written fresh for the hackathon. API surface confirmed against alphagenome 0.7.0:
  - dna_client.create(api_key) -> DnaClient
  - client.score_variant(interval, variant, variant_scorers=[...]) -> list[AnnData]
  - variant_scorers.CenterMaskScorer / RECOMMENDED_VARIANT_SCORERS
  - variant_scorers.tidy_scores(list_of_anndata) -> tidy DataFrame

Design notes:
  * Every response is cached to disk keyed by (variant, scorer-set, interval width).
    Re-runs are free; we never re-hit the API for an already-scored variant.
  * AlphaGenome is fine for thousands of predictions; 164 DAVs + negatives is well
    within budget.
  * genome.Variant takes the genomic position; genome.Interval is 0-based half-open.
    We center the interval on the variant position.
"""
from __future__ import annotations
import os, json, hashlib, pathlib

# --- Network: AlphaGenome talks gRPC to gdmscience.googleapis.com:443.
# In a proxied sandbox, gRPC must be told to use the native resolver and tunnel
# through the HTTP proxy, else channel setup hangs forever. Set before grpc import.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
if os.environ.get("http_proxy"):
    os.environ.setdefault("grpc_proxy", os.environ["http_proxy"])

import pandas as pd
from alphagenome.models import dna_client, variant_scorers
from alphagenome.data import genome

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 1 Mb native context centered on each variant (max supported = SEQUENCE_LENGTH_1MB).
DEFAULT_WIDTH = dna_client.SEQUENCE_LENGTH_1MB


def get_client(api_key: str | None = None) -> "dna_client.DnaClient":
    """Create an AlphaGenome client. Key from arg or ALPHAGENOME_API_KEY env var."""
    api_key = api_key or os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No AlphaGenome API key. Add it under Customize -> Credentials "
            "(name it so it appears as ALPHAGENOME_API_KEY), or pass api_key=."
        )
    return dna_client.create(api_key)


def default_scorers():
    """Regulatory-activity scorers most relevant to an enhancer MPRA:
    chromatin accessibility (DNASE/ATAC) + transcription initiation (CAGE).
    These are the endogenous analogues of episomal reporter activity."""
    rec = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    return [rec["DNASE"], rec["ATAC"], rec["CAGE"], rec["CHIP_HISTONE"]]


def _cache_key(variant: "genome.Variant", width: int, scorer_names: tuple[str, ...]) -> str:
    raw = f"{variant.chromosome}:{variant.position}:{variant.reference_bases}>{variant.alternate_bases}|{width}|{','.join(scorer_names)}"
    return hashlib.md5(raw.encode()).hexdigest()


def score_one(client, chrom: str, pos: int, ref: str, alt: str,
              name: str = "", width: int = DEFAULT_WIDTH,
              scorers=None) -> pd.DataFrame:
    """Score a single variant; returns the tidy per-track score DataFrame.
    Cached to disk as parquet keyed by variant+scorers+width."""
    scorers = scorers or default_scorers()
    scorer_names = tuple(getattr(s, "name", type(s).__name__) for s in scorers)
    variant = genome.Variant(chromosome=chrom, position=int(pos),
                             reference_bases=ref, alternate_bases=alt, name=name)
    key = _cache_key(variant, width, scorer_names)
    cache_path = CACHE_DIR / f"ag_{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Interval centered on the variant, clamped to a supported length.
    half = width // 2
    interval = genome.Interval(chromosome=chrom,
                               start=max(0, int(pos) - half),
                               end=int(pos) - half + width)
    ann_list = client.score_variant(interval=interval, variant=variant,
                                    variant_scorers=scorers)
    tidy = variant_scorers.tidy_scores(ann_list)
    if tidy is None:
        tidy = pd.DataFrame()
    # tidy_scores (alphagenome >=0.7) returns some columns holding AlphaGenome
    # objects (Variant, Interval) that parquet/arrow cannot serialize. Stringify
    # any object column whose first non-null value is not a plain str/number.
    for col in tidy.columns:
        if tidy[col].dtype == object:
            nonnull = tidy[col].dropna()
            if len(nonnull) and not isinstance(nonnull.iloc[0], (str, bytes, int, float, bool)):
                tidy[col] = tidy[col].astype(str)
    tidy["variant_name"] = name
    tidy["chrom"] = chrom
    tidy["pos"] = int(pos)
    tidy["ref"] = ref
    tidy["alt"] = alt
    tidy.to_parquet(cache_path)
    return tidy


def score_table(variants_df: pd.DataFrame, api_key: str | None = None,
                width: int = DEFAULT_WIDTH, scorers=None,
                cols=("chrom", "pos", "ref", "alt", "variant_id")) -> pd.DataFrame:
    """Score every row of a tidy variant table. Expects columns:
    chrom, pos (hg38), ref, alt, variant_id. Returns concatenated tidy scores."""
    client = get_client(api_key)
    c_chrom, c_pos, c_ref, c_alt, c_id = cols
    out = []
    for i, row in variants_df.iterrows():
        df = score_one(client, row[c_chrom], row[c_pos], row[c_ref], row[c_alt],
                       name=str(row[c_id]), width=width, scorers=scorers)
        out.append(df)
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(variants_df)}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


if __name__ == "__main__":
    # Quick-start smoke test from the AlphaGenome docs: chr22:36201698 A>C.
    client = get_client()
    df = score_one(client, "chr22", 36201698, "A", "C", name="smoke_test")
    print(df.head())
    print("scored rows:", len(df))
