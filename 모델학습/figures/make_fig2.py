"""
논문용 Figure 2 (2-패널) — v2 단일 모델(cycle41)의 학습 품질·성능 증명.

  (a) 학습 곡선 — train loss(category) 하락 + val category-F1 상승/plateau,
      best epoch(early stop) 표시. "수렴했고 과적합을 막았다"를 증명.
  (b) 카테고리별 test F1 — 7개 카테고리 성능 + macro-F1 기준선.

데이터: runs/cycle41_taskweight_cat/{history.json, result.json} (추론 불필요)
출력: fig2_v2_training_performance.pdf + .png (300dpi)
실행: source .venv/bin/activate && python 모델학습/figures/make_fig2.py
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KR = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({
    "font.family": KR, "axes.unicode_minus": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "savefig.bbox": "tight",
})

C_LOSS = "#C0504D"   # train loss (적)
C_F1   = "#2C6FB5"   # val/test F1 (청)
C_MARK = "#666666"   # best epoch 선
C_BAR  = "#2C6FB5"

RUN = os.path.join(os.path.dirname(__file__), "..", "runs", "cycle41_taskweight_cat")
hist = json.load(open(os.path.join(RUN, "history.json")))
res  = json.load(open(os.path.join(RUN, "result.json")))

ep      = [e["epoch"] for e in hist]
loss    = [e["train_loss"]["cat"] if isinstance(e["train_loss"], dict) else e["train_loss"] for e in hist]
val_f1  = [e["val_cat_f1"] for e in hist]
best_i  = max(range(len(hist)), key=lambda i: hist[i]["val_cat_f1"])
best_ep = hist[best_i]["epoch"]
best_f1 = hist[best_i]["val_cat_f1"]

per_f1 = res["test_cat_per_class_f1"]
macro  = res["test_cat_macro_f1"]
order  = sorted(per_f1.items(), key=lambda x: x[1])   # 작은 값이 아래(barh)
labels = [k.replace("_", "·") for k, _ in order]
vals   = [v for _, v in order]

# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11.0, 4.3))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.30)

# ── (a) 학습 곡선 (twin-axis) ─────────────────────────────────
axL = fig.add_subplot(gs[0, 0])
l1, = axL.plot(ep, loss, color=C_LOSS, marker="o", ms=3.5, lw=1.8)
axL.set_xlabel("epoch", fontsize=10)
axL.set_ylabel("train loss", color=C_LOSS, fontsize=10)
axL.tick_params(axis="y", labelcolor=C_LOSS)
axL.set_ylim(0, max(loss) * 1.08)
axL.set_xticks(ep)
axL.grid(axis="x", color="#EEEEEE")
for s in ("top",): axL.spines[s].set_visible(False)

axR = axL.twinx()
l2, = axR.plot(ep, val_f1, color=C_F1, marker="s", ms=3.5, lw=1.8)
# 곡선 옆 인라인 라벨 (범례 박스 대신)
axL.text(4.2, loss[3] + 0.025, "train loss ↓", color=C_LOSS, fontsize=9.5, fontweight="bold")
axR.text(6.0, val_f1[6] - 0.018, "val F1 ↑ (plateau)", color=C_F1, fontsize=9.5, fontweight="bold")
axR.set_ylabel("validation F1", color=C_F1, fontsize=10)
axR.tick_params(axis="y", labelcolor=C_F1)
axR.set_ylim(0.68, 0.84)
for s in ("top",): axR.spines[s].set_visible(False)

# best epoch (early stop) 표시
axR.axvline(best_ep, color=C_MARK, ls="--", lw=1.0)
axR.annotate(f"best epoch {best_ep}\n(early stop)\nF1={best_f1:.3f}",
             xy=(best_ep, best_f1), xytext=(best_ep + 0.5, 0.715),
             fontsize=8.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color=C_MARK, lw=0.9))
# plateau 음영 (best 이후 = train↓ val 평탄)
axR.axvspan(best_ep, ep[-1], color="#F2F2F2", zorder=0)
axR.text((best_ep + ep[-1]) / 2, 0.83, "val 평탄 · train ↓\n→ 과적합 억제",
         ha="center", va="top", fontsize=8, color="#888888")

axL.set_title("(a) 학습 곡선 (v2 단일 모델, cycle41)", fontsize=11, pad=10)

# ── (b) 카테고리별 test F1 ────────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
y = range(len(labels))
axB.barh(list(y), vals, color=C_BAR, height=0.62, zorder=3)
for i, v in enumerate(vals):
    axB.text(v + 0.004, i, f"{v:.2f}", va="center", ha="left", fontsize=9, color="#222222")
axB.axvline(macro, color=C_LOSS, ls="--", lw=1.1, zorder=2)
axB.text(macro, len(labels) - 0.35, f"macro-F1 {macro:.3f}", color=C_LOSS,
         fontsize=9, ha="center", va="bottom")
axB.set_yticks(list(y)); axB.set_yticklabels(labels, fontsize=10)
axB.set_xlim(0.6, 0.95)
axB.set_xlabel("카테고리별 test F1", fontsize=10)
axB.set_title("(b) 카테고리별 성능 (test, 738건)", fontsize=11, pad=10)
axB.grid(axis="x", color="#EEEEEE", zorder=0)
for s in ("top", "right", "left"): axB.spines[s].set_visible(False)
axB.tick_params(length=0)

out = os.path.join(os.path.dirname(__file__), "fig2_v2_training_performance")
fig.savefig(out + ".svg"); fig.savefig(out + ".png", dpi=300)
print(f"저장: {out}.pdf / .png")
print(f"best epoch {best_ep} (val F1 {best_f1:.4f}) · test macro-F1 {macro:.4f}")
