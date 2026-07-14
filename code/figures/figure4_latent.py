"""
Figure-generation code: figure4_latent
Reconstructed from artifact lineage (version 1484e107-f4c0-450d-a4c6-5b9c178aca80).
Source file at generation time: fig4_latent_arbiter.png
Environment: mpra
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import fitz
import os

doc = fitz.open('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/0efa69b2-7eda-4992-b60d-f9eb46e2b021/vd7f339a8_manuscript.pdf')

recover = {115:'fig2_bayes_denoise.png',119:'fig4_latent_arbiter.png',123:'fig5_concordance_map.png',
           131:'figS1_conservation.png',133:'figS2_multimodal.png'}
for xr,fn in recover.items():
    d=doc.extract_image(xr)
    with open(fn,'wb') as f: f.write(d["image"])
    print(fn, os.path.getsize(fn), "bytes")