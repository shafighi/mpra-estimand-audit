# When ground truth is a different estimand

**Auditing DNA foundation models, reporter assays, and endogenous genetics for regulatory-variant effect.**

This repository contains all code, processed data, and the manuscript for a benchmark that treats
DNA foundation models (AlphaGenome, Borzoi, Enformer, Evo 2) and a developing-human-cortex
integrated lentiMPRA as **two imperfect estimators** of regulatory-variant effect, rather than
treating either as ground truth. The central finding: the three activity-predicting sequence models
agree strongly with each other (rank rho 0.55 to 0.81) but are essentially uncorrelated with the
reporter assay (rho approximately -0.02, n = 3,976), and this gap is not explained by the reporter's
reported per-variant uncertainty. An independent fine-mapped fetal-cortex eQTL anchor gives modest,
independent support for the model-derived ranking (rho 0.24 to 0.28; direction concordance 20/25 at
PIP >= 0.5).

## Headline numbers

| Quantity | Value |
|---|---|
| Benchmark variants (biallelic SNVs) | 3,976 |
| Differentially active variants (DAVs) | 76 |
| Model-reporter agreement (Spearman) | -0.020 |
| Inter-model agreement (rank rho) | 0.55 to 0.81 |
| eQTL anchor agreement (AlphaGenome / Borzoi) | 0.24 / 0.28 |
| High-PIP direction concordance | 20/25 (PIP >= 0.5) |
| Calibrated latent-model agreement | negligible across 7 specifications (|rho| <= 0.052) |

## Repository layout

```
mpra-estimand-audit/
  code/
    scoring/     variant QC + sequence-model scoring (AlphaGenome, Borzoi, Enformer, phyloP)
    models/      Bayesian de-noising, latent measurement model, ACP, sensitivity analysis
    analysis/    agreement / concordance computation
    figures/     one script per published figure (reconstructed from run lineage)
    tools/       citation / figure / LaTeX consistency linters
    requirements-*.txt
  data/
    processed/   per-variant scores, benchmark table, posteriors, concordance map, model stats
    eqtl/        fine-mapped eQTL benchmark + overlap tables
  manuscript/
    manuscript_rev.pdf              compiled paper
    manuscript_selfcontained.html   standalone HTML (figures embedded)
    manuscript.tex                  LaTeX source (two-column)
    figures/                        fig1-7 + figS1-S5 (PNG, 300 dpi)
  README.md, DATA_DICTIONARY.md, LICENSE, .gitignore
```

## Reproducing the results

1. **Environments.** Three requirement sets are provided because the pipeline spans CPU analysis,
   probabilistic modeling, and GPU sequence scoring:
   - `pip install -r code/requirements-analysis.txt`  (tables, plots)
   - `pip install -r code/requirements-bayes.txt`  (NumPyro models; `code/models/`)
   - `pip install -r code/requirements-scoring.txt`  (GPU; `code/scoring/`)

2. **From processed data (fast).** Every figure and headline number can be regenerated from
   `data/processed/` and `data/eqtl/` without re-scoring: run any script in `code/figures/`.

3. **From raw inputs (full).** Sequence scoring requires the raw variant library and reference
   genome (see "External data" below), a GPU, and an AlphaGenome API key. Scripts in
   `code/scoring/` produce the per-model score parquets in `data/processed/`.

## External data (not redistributed here)

These third-party inputs are required only for a full re-scoring and are available from their
original sources:

| Dataset | Source |
|---|---|
| Developing-cortex lentiMPRA variant library + effects | Deng et al. 2023, bioRxiv 2023.02.15.528663 |
| Developing-brain xQTL atlas (fine-mapped eQTL) | github.com/gandallab/devBrain_xQTL; Science 2024, doi:10.1126/science.adh0829 |
| SCZ neuronal Hi-C loop calls (Fig S4) | bioRxiv 2023.07.17.549339, supplementary S4 |
| hg38 reference genome | UCSC / Ensembl |
| AlphaGenome | API, doi:10.1101/2025.06.25.661532 |

## Notes on figure code

`code/figures/*.py` were reconstructed from the recorded run lineage of each published figure.
Figure 1 (overview schematic) and Figure S3 (plate-notation DAGs) are illustrations; see the
`*_README.txt` files in that folder. Figure scripts reference the original run paths; point them at
`data/processed/` before re-running.
