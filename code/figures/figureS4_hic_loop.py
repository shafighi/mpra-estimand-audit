"""
Figure-generation code: figureS4_hic_loop
Reconstructed from artifact lineage (version b5a7c28e-9d7e-4095-96b6-c9d76b3e92e5).
Source file at generation time: figure8_3dgenome.png
Environment: python
Note: paths/markers reflect the original run; adjust data paths to data/ before re-running.
"""

import matplotlib.pyplot as plt
from matplotlib import image as mpimg

pa = mpimg.imread('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/a088e14d-e1ad-4fbf-a761-0e6c837442d8/va1cd29f4_F8_a_loop_attenuation.png')
pb = mpimg.imread('/Users/darvis01/.claude-science/orgs/c964c4e1-2411-4c6b-aca7-20ef49b419e8/artifacts/proj_934f6aa814c9/90572455-9380-4f2a-b120-f6464c7b068a/v4e9757d7_F8_b_loop_concordance.png')
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.1))
for ax, im, L in zip(axs, [pa, pb], ["a", "b"]):
    ax.imshow(im); ax.axis("off")
    ax.text(-0.02, 1.02, L, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, wspace=0.03)
fig.savefig('figure8_3dgenome.png', dpi=300, bbox_inches="tight")
plt.close(fig)