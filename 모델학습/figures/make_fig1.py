"""
논문용 Figure 1 (2-패널) 생성.

  (a) 시스템 동작 예시 — 입력 광고 1건이 7-카테고리 확률 → 가중 합산 →
      강도 점수로 변환되는 전 과정을 카테고리별 기여도 막대로 시각화.
  (b) 카테고리별 성능 — v1(단순) vs v2(데이터 보강+멀티태스크) F1 비교.
      약한 카테고리(정체성·긴급·권위)의 개선을 한눈에.

출력: fig1_system_performance.pdf (벡터) + .png (300dpi)
폰트: Noto Sans CJK KR
실행: source .venv/bin/activate && python 모델학습/figures/make_fig1.py
"""
from __future__ import annotations
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch

# ── 한글 폰트 (Noto Sans CJK KR) ──────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KR = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({
    "font.family": KR,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

# ── 색상 (색맹·흑백 안전) ─────────────────────────────────────
C_V1   = "#BFBFBF"   # v1 = 회색
C_V2   = "#2C6FB5"   # v2 = 청색 (명도 충분히 차이 → 흑백서도 구분)
C_CTRB = "#5B9BD5"   # 기여도 막대
C_ACC  = "#C0504D"   # 강조(점수)
HL     = "#E8A33D"   # 개선 화살표

# 점수식 모듈에서 예시 계산 (단일 출처 → 본문 수치와 자동 일치)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring_v2 import calculate_ad_score, CategoryResult  # noqa: E402

EXAMPLE_TEXT = "오늘만 50% 할인! 무료배송"
ex = [
    CategoryResult("긴급성",  0.92, True, "긍정", "강"),
    CategoryResult("가격비교", 0.94, True, "긍정", "강"),
    CategoryResult("호혜성",  0.80, True, "긍정", "보통"),
]
res = calculate_ad_score(ex)
syn   = res["synergy_factor"]
final = res["final_score_100"]
# 시너지 적용 전 카테고리별 기여도(×100), 큰 값이 위로
contrib = sorted(
    [(c["category"], c["score"] * 100, c["probability"]) for c in res["per_category"]],
    key=lambda x: x[1],
)
raw_sum = sum(v for _, v, _ in contrib)

# ── 성능 데이터 (확정값) ──────────────────────────────────────
#   v1: Cycle 23 앙상블 / v2: Cycle 50 (README·CLAUDE.md)
PERF = [   # (카테고리, v1, v2)  — v2 기준 내림차순
    ("호혜성",       0.875, 0.88),
    ("희소성",       0.850, 0.86),
    ("권위·신뢰",     0.770, 0.86),
    ("사회적 증명",   0.811, 0.84),
    ("긴급성",       0.726, 0.83),
    ("가격비교",      0.833, 0.83),
    ("사회적 정체성", 0.643, 0.70),
]
V2_MEAN = 0.83

# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11.0, 4.4))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.34)

# ── 패널 (a): 시스템 동작 예시 ────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
cats = [c for c, _, _ in contrib]
vals = [v for _, v, _ in contrib]
ypos = range(len(cats))
bars = axA.barh(ypos, vals, color=C_CTRB, edgecolor="white", height=0.6, zorder=3)
for y, (c, v, p) in zip(ypos, contrib):
    axA.text(v + 0.3, y, f"{v:.1f}  (p={p:.2f})", va="center", ha="left",
             fontsize=9, color="#333333")
axA.set_yticks(list(ypos))
axA.set_yticklabels(cats, fontsize=10)
axA.set_xlim(0, max(vals) * 1.7)
axA.set_xlabel("카테고리 기여도  $w_i\\,·\\,prob_i\\,·\\,p_i\\,·\\,t_i \\times 100$", fontsize=9.5)
axA.set_title(f'(a) 시스템 동작 예시\n"{EXAMPLE_TEXT}"', fontsize=11, pad=10)
axA.grid(axis="x", color="#E6E6E6", zorder=0)
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.tick_params(length=0)

# 합산 → 시너지 → 최종점수 흐름 (막대 아래 박스)
flow = (f"$\\Sigma$ = {raw_sum:.1f}    ×  시너지(3개 동시) = {syn:.2f}"
        f"    →    강도 점수 = {final:.1f} / 100")
axA.text(0.5, -0.30, flow, transform=axA.transAxes, ha="center", va="top",
         fontsize=10.5, color="#222222",
         bbox=dict(boxstyle="round,pad=0.5", fc="#FBF4E6", ec=C_ACC, lw=1.2))

# ── 패널 (b): 카테고리별 성능 v1 vs v2 ────────────────────────
axB = fig.add_subplot(gs[0, 1])
labels = [p[0] for p in PERF]
v1 = [p[1] for p in PERF]
v2 = [p[2] for p in PERF]
y = range(len(labels))
h = 0.38
axB.barh([i + h/2 for i in y], v1, height=h, color=C_V1, label="v1 (단순)", zorder=3)
axB.barh([i - h/2 for i in y], v2, height=h, color=C_V2, label="v2 (보강+멀티태스크)", zorder=3)
for i, (l, a, b) in enumerate(PERF):
    axB.text(b + 0.004, i - h/2, f"{b:.2f}", va="center", ha="left", fontsize=8.5, color=C_V2)
    d = b - a
    if d >= 0.05:   # 큰 개선만 Δ 강조
        axB.annotate(f"+{d:.2f}", xy=(b, i - h/2), xytext=(b + 0.055, i - h/2),
                     va="center", fontsize=8.5, color=HL, fontweight="bold")
axB.axvline(V2_MEAN, color=C_ACC, ls="--", lw=1.0, zorder=2)
axB.text(V2_MEAN, len(labels) - 0.3, f"v2 평균 {V2_MEAN:.2f}", color=C_ACC,
         fontsize=8.5, ha="center", va="bottom")
axB.set_yticks(list(y))
axB.set_yticklabels(labels, fontsize=10)
axB.invert_yaxis()
axB.set_xlim(0.6, 0.95)
axB.set_xlabel("카테고리별 F1", fontsize=9.5)
axB.set_title("(b) 카테고리별 성능 (v1 → v2)", fontsize=11, pad=10)
axB.grid(axis="x", color="#E6E6E6", zorder=0)
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.tick_params(length=0)
axB.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95,
           edgecolor="#CCCCCC")

# ── 저장 ──────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "fig1_system_performance")
fig.savefig(out + ".svg")
fig.savefig(out + ".png", dpi=300)
print(f"저장: {out}.pdf / .png")
print(f"예시 점수: Σ={raw_sum:.2f} × synergy {syn} = {final}/100")
