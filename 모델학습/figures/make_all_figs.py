"""
논문용 figure — 종류별 개별 정사각형 생성 (추론 불필요 9종).

출력: 모델학습/figures/square/<name>.pdf + .png (300dpi, ~5.4×5.4in)
폰트: Noto Sans CJK KR / 색맹·흑백 안전(명도차 + 해칭)
실행: source .venv/bin/activate && python 모델학습/figures/make_all_figs.py

생성 목록:
  01_learning_curve      학습 곡선 (loss↓ + valF1↑, early stop)        ← 학습 잘됨
  02_per_category_f1     카테고리별 test F1 + macro                    ← 성능
  03_v1_vs_v2            v1→v2 카테고리별 F1 개선                       ← 개선 증명
  04_weights_grade       학술 가중치 + 근거 등급(해칭)                  ← 가중치 정당성
  05_synergy_curve       시너지 Inverted-U                             ← 보정 설계
  06_scoring_ablation    점수식 ablation (MAE)                         ← 설계 근거
  07_data_composition    카테고리별 양성 비율(불균형)                   ← 데이터 특성
  08_score_decomposition 점수식 분해(예시 1건)                          ← 식 직관화
  09_irr_kappa           라벨 신뢰도 κ + Landis-Koch                   ← 라벨 품질
"""
from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(__file__)
OUT  = os.path.join(HERE, "square")
os.makedirs(OUT, exist_ok=True)
RUN  = os.path.join(HERE, "..", "runs", "cycle41_taskweight_cat")
sys.path.insert(0, os.path.join(HERE, ".."))

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KR = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({
    "font.family": KR, "axes.unicode_minus": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8, "savefig.bbox": "tight",
    "figure.dpi": 110,
})

# 팔레트
BLUE, RED, GRAY, AMBER, GREEN = "#2C6FB5", "#C0504D", "#BFBFBF", "#E8A33D", "#4C9A5A"
SZ = (5.4, 5.4)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p + ".svg"); fig.savefig(p + ".png", dpi=300)
    plt.close(fig)
    print("  •", name)


def nospine(ax, keep=("bottom", "left")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


# ── 01 학습 곡선 ──────────────────────────────────────────────
def fig_learning():
    hist = json.load(open(os.path.join(RUN, "history.json")))
    ep   = [e["epoch"] for e in hist]
    loss = [e["train_loss"]["cat"] if isinstance(e["train_loss"], dict) else e["train_loss"] for e in hist]
    vf1  = [e["val_cat_f1"] for e in hist]
    bi   = max(range(len(hist)), key=lambda i: hist[i]["val_cat_f1"])
    be, bf = hist[bi]["epoch"], hist[bi]["val_cat_f1"]

    fig, axL = plt.subplots(figsize=SZ)
    axL.plot(ep, loss, color=RED, marker="o", ms=3.5, lw=1.8)
    axL.set_xlabel("epoch"); axL.set_ylabel("train loss", color=RED)
    axL.tick_params(axis="y", labelcolor=RED); axL.set_ylim(0, max(loss)*1.08)
    axL.set_xticks(ep[::2]); nospine(axL, ("bottom", "left"))
    axL.text(4.2, loss[3]+0.03, "train loss ↓", color=RED, fontsize=10, fontweight="bold")

    axR = axL.twinx()
    axR.plot(ep, vf1, color=BLUE, marker="s", ms=3.5, lw=1.8)
    axR.set_ylabel("validation F1", color=BLUE); axR.tick_params(axis="y", labelcolor=BLUE)
    axR.set_ylim(0.68, 0.84); nospine(axR, ("bottom",))
    axR.text(5.5, vf1[6]-0.02, "val F1 ↑ (plateau)", color=BLUE, fontsize=10, fontweight="bold")
    axR.axvline(be, color="#666", ls="--", lw=1.0)
    axR.axvspan(be, ep[-1], color="#F2F2F2", zorder=0)
    axR.annotate(f"best ep {be} (early stop)\nF1={bf:.3f}", xy=(be, bf), xytext=(be+0.4, 0.705),
                 fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#666", lw=0.9))
    axL.set_title("학습 곡선 (v2 단일 모델, cycle41)", fontsize=12, pad=10)
    save(fig, "01_learning_curve")


# ── 02 카테고리별 test F1 ─────────────────────────────────────
def fig_per_cat_f1():
    res = json.load(open(os.path.join(RUN, "result.json")))
    per = res["test_cat_per_class_f1"]; macro = res["test_cat_macro_f1"]
    items = sorted(per.items(), key=lambda x: x[1])
    labels = [k.replace("_", "·") for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=SZ)
    ax.barh(range(len(vals)), vals, color=BLUE, height=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v+0.004, i, f"{v:.2f}", va="center", fontsize=9.5)
    ax.axvline(macro, color=RED, ls="--", lw=1.1)
    ax.text(macro, len(vals)-0.3, f"macro-F1 {macro:.3f}", color=RED, fontsize=9.5, ha="center", va="bottom")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlim(0.6, 0.95); ax.set_xlabel("카테고리별 test F1")
    ax.grid(axis="x", color="#EEE", zorder=0); nospine(ax, ("bottom",)); ax.tick_params(length=0)
    ax.set_title("카테고리별 성능 (test 738건)", fontsize=12, pad=10)
    save(fig, "02_per_category_f1")


# ── 03 v1 vs v2 ──────────────────────────────────────────────
def fig_v1v2():
    PERF = [("호혜성",0.875,0.88),("희소성",0.850,0.86),("권위·신뢰",0.770,0.86),
            ("사회적 증명",0.811,0.84),("긴급성",0.726,0.83),("가격비교",0.833,0.83),
            ("사회적 정체성",0.643,0.70)]
    labels=[p[0] for p in PERF]; v1=[p[1] for p in PERF]; v2=[p[2] for p in PERF]
    y=range(len(labels)); h=0.38
    fig, ax = plt.subplots(figsize=SZ)
    ax.barh([i+h/2 for i in y], v1, height=h, color=GRAY, label="v1 (단순)", zorder=3)
    ax.barh([i-h/2 for i in y], v2, height=h, color=BLUE, label="v2 (보강+멀티태스크)", zorder=3)
    for i,(l,a,b) in enumerate(PERF):
        ax.text(b+0.004, i-h/2, f"{b:.2f}", va="center", fontsize=8.5, color=BLUE)
        if b-a>=0.05: ax.annotate(f"+{b-a:.2f}", xy=(b,i-h/2), xytext=(b+0.05,i-h/2),
                                  va="center", fontsize=8.5, color=AMBER, fontweight="bold")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0.6,0.97); ax.set_xlabel("카테고리별 F1")
    ax.grid(axis="x", color="#EEE", zorder=0); nospine(ax, ("bottom",)); ax.tick_params(length=0)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="#CCC")
    ax.set_title("카테고리별 성능 개선 (v1 → v2)", fontsize=12, pad=10)
    save(fig, "03_v1_vs_v2")


# ── 04 가중치 + 근거 등급 ─────────────────────────────────────
def fig_weights():
    from scoring_v2 import WEIGHTS
    # (카테고리, 등급, 출처) — 등급: A 직접메타분석 / B 문헌고찰·이론 / C 단일연구
    GRADE = {
        "권위_신뢰":("B","Pornpitakpan(2004) 문헌고찰"),
        "사회적_증명":("A","Qiu & Zhang(2024) 메타분석"),
        "사회적_정체성":("A","Aguirre-Rodriguez(2012) 메타분석"),
        "호혜성":("B","Cialdini+Raghubir 이론추정"),
        "희소성":("C","Khalid(2025) 단일연구"),
        "긴급성":("B","Ladeira(2023) 메타분석*"),
        "가격비교":("C","이상수(2023) 단일연구"),
    }
    style={"A":(GREEN,None,"A 직접 메타분석 r"),
           "B":(AMBER,"//","B 문헌고찰·이론 추정"),
           "C":(RED,"xx","C 단일 실증연구 β→r")}
    items=sorted(WEIGHTS.items(), key=lambda x:x[1])
    labels=[k.replace("_","·") for k,_ in items]; vals=[v*100 for _,v in items]
    fig, ax = plt.subplots(figsize=SZ)
    for i,(cat,_) in enumerate(items):
        g=GRADE[cat][0]; c,hatch,_=style[g]
        ax.barh(i, vals[i], color=c, hatch=hatch, edgecolor="white", height=0.62, zorder=3)
        ax.text(vals[i]+0.3, i, f"{vals[i]:.1f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("가중치 (정규화 효과크기 r, %)"); ax.set_xlim(0, max(vals)*1.25)
    ax.grid(axis="x", color="#EEE", zorder=0); nospine(ax, ("bottom",)); ax.tick_params(length=0)
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=style[g][0], hatch=style[g][1], edgecolor="white", label=style[g][2]) for g in "ABC"]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#CCC")
    ax.set_title("학술 가중치 + 근거 등급", fontsize=12, pad=10)
    fig.text(0.12, -0.02, "* Ladeira(2023)은 긴급성 단서 무효 결론 → 보수적 해석", fontsize=7.5, color="#888")
    save(fig, "04_weights_grade")


# ── 05 시너지 Inverted-U ─────────────────────────────────────
def fig_synergy():
    from scoring_v2 import synergy_factor
    ns=list(range(1,7)); ys=[synergy_factor(n) for n in ns]
    fig, ax = plt.subplots(figsize=SZ)
    ax.plot(ns, ys, color=BLUE, marker="o", ms=8, lw=2.2, zorder=3)
    ax.axhline(1.0, color="#999", ls=":", lw=1.0)
    for n,y in zip(ns,ys):
        ax.annotate(f"×{y:.2f}", (n,y), textcoords="offset points", xytext=(0,10),
                    ha="center", fontsize=9.5, fontweight="bold")
    ax.annotate("정점 (3개 동시)", (3,1.25), textcoords="offset points", xytext=(10,-20),
                fontsize=9, color=GREEN, fontweight="bold")
    ax.annotate("거부감·심리적 반발 (4개 이상)", xy=(4,0.90), xytext=(4.2,1.06),
                ha="center", fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_xticks(ns); ax.set_ylim(0.8,1.35); ax.set_xlabel("동시 탐지 카테고리 수 (n)")
    ax.set_ylabel("시너지 보정 계수"); ax.grid(color="#EEE", zorder=0); nospine(ax)
    ax.set_title("시너지 보정 (Inverted-U)", fontsize=12, pad=10)
    save(fig, "05_synergy_curve")


# ── 06 점수식 ablation ───────────────────────────────────────
def fig_ablation():
    r=json.load(open(os.path.join(HERE,"..","scoring_ablation_result.json")))["variants"]
    order=[("v1 단순 (Σ w·prob)×100","v1 단순"),("Full (PDF 확장)","Full (PDF)"),
           ("-방향성","− 방향성"),("-시너지","− 시너지"),("-강도","− 강도"),("-신뢰도 보정","− 신뢰도 (채택)")]
    labels=[s for _,s in order]; vals=[r[k]["mae_vs_true"] for k,_ in order]
    colors=[GRAY,GRAY,GRAY,GRAY,GRAY,GREEN]
    fig, ax = plt.subplots(figsize=SZ)
    bars=ax.bar(range(len(vals)), vals, color=colors, width=0.62, zorder=3)
    for i,v in enumerate(vals):
        ax.text(i, v+0.08, f"{v:.2f}", ha="center", fontsize=9.5,
                fontweight="bold" if i==len(vals)-1 else "normal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("정답과의 평균 오차 MAE  (낮을수록 정확)"); ax.set_ylim(0, max(vals)*1.15)
    ax.grid(axis="y", color="#EEE", zorder=0); nospine(ax)
    ax.set_title("점수식 ablation (test 922건)", fontsize=12, pad=10)
    fig.text(0.5,-0.04,"신뢰도 보정 제거 시 가장 정확 → 최종 채택", ha="center", fontsize=8.5, color=GREEN)
    save(fig, "06_scoring_ablation")


# ── 07 데이터 구성 (양성 비율, v3 6149) ──────────────────────
def fig_data():
    DATA=[("사회적 증명",22.2),("호혜성",21.1),("가격비교",20.6),("권위·신뢰",12.8),
          ("긴급성",12.7),("사회적 정체성",10.5),("희소성",10.4)]
    labels=[d[0] for d in DATA]; vals=[d[1] for d in DATA]
    colors=[RED if v<13 else BLUE for v in vals]   # 희귀 클래스 강조
    fig, ax = plt.subplots(figsize=SZ)
    ax.barh(range(len(vals)), vals, color=colors, height=0.62, zorder=3)
    for i,v in enumerate(vals): ax.text(v+0.2, i, f"{v:.1f}%", va="center", fontsize=9.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("양성 라벨 비율 (전체 6,149건)"); ax.set_xlim(0, 26)
    ax.grid(axis="x", color="#EEE", zorder=0); nospine(ax, ("bottom",)); ax.tick_params(length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED,label="희귀 클래스 (<13%)"),Patch(color=BLUE,label="일반")],
              loc="lower right", fontsize=8.5, frameon=True, edgecolor="#CCC")
    ax.set_title("카테고리별 양성 비율 (클래스 불균형)", fontsize=12, pad=10)
    save(fig, "07_data_composition")


# ── 08 점수식 분해 (예시) ────────────────────────────────────
def fig_decomp():
    from scoring_v2 import calculate_ad_score, CategoryResult
    ex=[CategoryResult("긴급성",0.92,True,"긍정","강"),
        CategoryResult("가격비교",0.94,True,"긍정","강"),
        CategoryResult("호혜성",0.80,True,"긍정","보통")]
    res=calculate_ad_score(ex); syn=res["synergy_factor"]; fin=res["final_score_100"]
    cont=sorted([(c["category"],c["score"]*100,c["probability"]) for c in res["per_category"]], key=lambda x:x[1])
    cats=[c.replace("_","·") for c,_,_ in cont]; vals=[v for _,v,_ in cont]; raw=sum(vals)
    fig, ax = plt.subplots(figsize=SZ)
    ax.barh(range(len(cats)), vals, color=BLUE, height=0.55, zorder=3)
    for i,(c,v,p) in enumerate(cont): ax.text(v+0.3, i, f"{v:.1f} (p={p:.2f})", va="center", fontsize=9)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_xlim(0, max(vals)*1.8); ax.set_xlabel(r"기여도  $w_i·prob_i·p_i·t_i \times 100$")
    ax.grid(axis="x", color="#EEE", zorder=0); nospine(ax, ("bottom",)); ax.tick_params(length=0)
    ax.set_title('점수식 분해\n"오늘만 50% 할인! 무료배송"', fontsize=12, pad=10)
    ax.text(0.5,-0.16, f"Σ={raw:.1f}  ×  시너지(3개)={syn:.2f}  →  {fin:.1f}/100",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FBF4E6", ec=RED, lw=1.2))
    save(fig, "08_score_decomposition")


# ── 09 IRR κ + Landis-Koch ───────────────────────────────────
def fig_irr():
    bands=[(0.0,0.20,"#F2DEDE","slight"),(0.20,0.40,"#FCE8D5","fair"),
           (0.40,0.60,"#FFF7D6,","moderate"),(0.60,0.80,"#E2EFDA","substantial"),
           (0.80,1.0,"#D5E8F2","almost perfect")]
    K=[("방향성",0.666),("강도",0.487)]
    fig, ax = plt.subplots(figsize=SZ)
    for lo,hi,col,name in bands:
        ax.axhspan(lo,hi,color=col.rstrip(","),zorder=0)
        ax.text(1.46, (lo+hi)/2, name, va="center", fontsize=8, color="#777")
    ax.bar([k[0] for k in K], [k[1] for k in K], color=[BLUE,AMBER], width=0.45, zorder=3)
    for i,(n,v) in enumerate(K):
        ax.text(i, v+0.02, f"κ={v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0,1.0); ax.set_ylabel("Cohen's κ (라벨러 간 일치도)")
    ax.set_xlim(-0.6,1.9); nospine(ax)
    ax.set_title("라벨링 신뢰도 (IRR, Landis-Koch 기준)", fontsize=12, pad=10)
    save(fig, "09_irr_kappa")


# ── 12 멀티태스크 3개 head 성능 ──────────────────────────────
def fig_three_tasks():
    res = json.load(open(os.path.join(RUN, "result.json")))
    cat = res["test_cat_per_class_f1"]; pol = res["test_pol_per_class_f1"]; itn = res["test_int_per_class_f1"]
    mc, mp, mi = res["test_cat_macro_f1"], res["test_pol_macro_f1"], res["test_int_macro_f1"]
    cats = sorted(cat, key=lambda c: cat[c])   # 작은 값 아래
    labels = [c.replace("_", "·") for c in cats]
    y = range(len(cats)); h = 0.26
    fig, ax = plt.subplots(figsize=SZ)
    ax.barh([i+h for i in y], [cat[c] for c in cats], height=h, color=BLUE,  label=f"카테고리  (macro {mc:.2f})", zorder=3)
    ax.barh([i    for i in y], [pol[c] for c in cats], height=h, color=GREEN, label=f"방향성    (macro {mp:.2f})", zorder=3)
    ax.barh([i-h for i in y], [itn[c] for c in cats], height=h, color=AMBER, label=f"강도      (macro {mi:.2f})", zorder=3)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05); ax.set_xlabel("카테고리별 test F1")
    ax.grid(axis="x", color="#EEE", zorder=0); nospine(ax, ("bottom",)); ax.tick_params(length=0)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#CCC")
    ax.set_title("멀티태스크 3개 head 성능", fontsize=12, pad=10)
    fig.text(0.5, -0.03, "강도 F1 0.65 = 가장 약한 태스크(주관성) · 방향성은 99% 긍정 편중", ha="center", fontsize=7.5, color="#888")
    save(fig, "12_three_tasks")


if __name__ == "__main__":
    print("정사각형 figure 생성:")
    fig_learning(); fig_per_cat_f1(); fig_v1v2(); fig_weights(); fig_synergy()
    fig_ablation(); fig_data(); fig_decomp(); fig_irr(); fig_three_tasks()
    print(f"→ {OUT}")
