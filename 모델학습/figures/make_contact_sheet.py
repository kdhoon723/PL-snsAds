"""square/ 의 모든 정사각형 figure를 한 장 컨택트 시트로 합쳐 미리보기."""
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(__file__)
OUT  = os.path.join(HERE, "square")
pngs = sorted(glob.glob(os.path.join(OUT, "*.png")))
n = len(pngs); cols = 3; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*4.2))
for ax, p in zip(axes.flat, pngs):
    ax.imshow(mpimg.imread(p)); ax.set_axis_off()
    ax.set_title(os.path.basename(p).replace(".png", ""), fontsize=9)
for ax in axes.flat[n:]:
    ax.set_axis_off()
fig.tight_layout()
out = os.path.join(HERE, "_contact_sheet.png")
fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
print("저장:", out, f"({n}종)")
