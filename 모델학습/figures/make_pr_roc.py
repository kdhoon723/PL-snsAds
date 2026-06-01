"""
논문용 figure (추론 필요) — PR 곡선 + ROC 곡선 (v2 단일 모델 cycle41).

  10_pr_curves   카테고리별 Precision-Recall 곡선 + AP, micro-average  ← 불균형 데이터 정석
  11_roc_curves  카테고리별 ROC 곡선 + AUC (참고용)

cycle41 best_model.pt로 test 738→실제 922행(줄바꿈 포함) 추론 후 sigmoid 확률로 곡선 산출.
확률·라벨은 test_probs.npz로 캐시 → 재플롯 시 추론 생략.
실행: source .venv/bin/activate && python 모델학습/figures/make_pr_roc.py
"""
from __future__ import annotations
import os, sys, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(__file__)
OUT  = os.path.join(HERE, "square"); os.makedirs(OUT, exist_ok=True)
PROJ = os.path.join(HERE, "..", "..")
RUN  = os.path.join(HERE, "..", "runs", "cycle41_taskweight_cat")
TEST = os.path.join(PROJ, "통합데이터셋", "split_test_v3.csv")
NPZ  = os.path.join(HERE, "test_probs.npz")

CAT = ["사회적_정체성","희소성","긴급성","사회적_증명","가격비교","권위_신뢰","호혜성"]
MODEL_NAME = "klue/roberta-large"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KR = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({"font.family": KR, "axes.unicode_minus": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.8, "savefig.bbox": "tight"})
SZ = (5.6, 5.6)
BLUE, RED, GREEN = "#2C6FB5", "#C0504D", "#4C9A5A"


def run_inference():
    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    class MultiTaskClassifier(nn.Module):
        def __init__(self, model_name, dropout=0.1):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            h = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.cat_head = nn.Linear(h, len(CAT))
            self.pol_head = nn.Linear(h, len(CAT) * 3)
            self.int_head = nn.Linear(h, len(CAT) * 3)
        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.last_hidden_state[:, 0])
            return self.cat_head(cls)

    rows = list(csv.DictReader(open(TEST, encoding="utf-8-sig")))
    texts = [r["text"] for r in rows]
    labels = np.array([[int(float(r[c])) if r.get(c) not in (None, "") else 0 for c in CAT] for r in rows])
    print(f"  test rows: {len(rows)} · 양성합계 {labels.sum(0).tolist()}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MultiTaskClassifier(MODEL_NAME).to(dev).eval()
    state = torch.load(os.path.join(RUN, "best_model.pt"), weights_only=True)
    model.load_state_dict(state)

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], truncation=True, padding="max_length",
                      max_length=256, return_tensors="pt").to(dev)
            logit = model(enc["input_ids"], enc["attention_mask"])
            probs.append(torch.sigmoid(logit).cpu().numpy())
    probs = np.concatenate(probs, 0)
    np.savez(NPZ, probs=probs, labels=labels)
    print(f"  저장: {NPZ}")
    return probs, labels


def load_or_infer():
    if os.path.exists(NPZ):
        d = np.load(NPZ); print("  캐시 사용:", NPZ)
        return d["probs"], d["labels"]
    print("  추론 실행 (cycle41)…")
    return run_inference()


def plot_curves(probs, labels):
    from sklearn.metrics import (precision_recall_curve, average_precision_score,
                                 roc_curve, auc)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    names = [c.replace("_", "·") for c in CAT]
    order = np.argsort(-labels.mean(0))   # 흔한 카테고리부터

    # ── PR ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=SZ)
    aps = []
    for k, idx in enumerate(order):
        y, s = labels[:, idx], probs[:, idx]
        ap = average_precision_score(y, s); aps.append(ap)
        pr, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, pr, color=colors[k], lw=1.6, label=f"{names[idx]} (AP {ap:.2f})")
    # micro-average
    ap_micro = average_precision_score(labels.ravel(), probs.ravel())
    pr, rc, _ = precision_recall_curve(labels.ravel(), probs.ravel())
    ax.plot(rc, pr, color="black", lw=2.4, ls="--", label=f"micro-avg (AP {ap_micro:.2f})")
    base = labels.mean()
    ax.axhline(base, color="#999", ls=":", lw=1.0)
    ax.text(0.02, base+0.01, f"baseline {base:.2f}", fontsize=8, color="#777")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="#CCC")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("Precision-Recall 곡선 (test, cycle41)", fontsize=12, pad=10)
    fig.savefig(os.path.join(OUT, "10_pr_curves.svg")); fig.savefig(os.path.join(OUT, "10_pr_curves.png"), dpi=300)
    plt.close(fig); print(f"  • 10_pr_curves  (macro-AP {np.mean(aps):.3f})")

    # ── ROC ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=SZ)
    aucs = []
    for k, idx in enumerate(order):
        y, s = labels[:, idx], probs[:, idx]
        fpr, tpr, _ = roc_curve(y, s); a = auc(fpr, tpr); aucs.append(a)
        ax.plot(fpr, tpr, color=colors[k], lw=1.6, label=f"{names[idx]} (AUC {a:.2f})")
    fpr, tpr, _ = roc_curve(labels.ravel(), probs.ravel()); a_micro = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="black", lw=2.4, ls="--", label=f"micro-avg (AUC {a_micro:.2f})")
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1.0)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="#CCC")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("ROC 곡선 (test, cycle41) — 참고용", fontsize=12, pad=10)
    fig.savefig(os.path.join(OUT, "11_roc_curves.svg")); fig.savefig(os.path.join(OUT, "11_roc_curves.png"), dpi=300)
    plt.close(fig); print(f"  • 11_roc_curves  (macro-AUC {np.mean(aucs):.3f})")


def fig_confusion(probs, labels):
    """13_confusion — 카테고리별 2×2 혼동행렬 small-multiples (행 정규화=recall)."""
    res = json.load(open(os.path.join(RUN, "result.json")))
    thr = res["best_thresholds"]
    names = [c.replace("_", "·") for c in CAT]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.4))
    for k, c in enumerate(CAT):
        ax = axes.flat[k]
        y = labels[:, k]; pred = (probs[:, k] >= thr[c]).astype(int)
        tn = int(((y==0)&(pred==0)).sum()); fp = int(((y==0)&(pred==1)).sum())
        fn = int(((y==1)&(pred==0)).sum()); tp = int(((y==1)&(pred==1)).sum())
        M = np.array([[tn, fp], [fn, tp]], float)
        Mn = M / M.sum(1, keepdims=True).clip(min=1)   # 행(실제)별 정규화
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for (r, col), v in np.ndenumerate(M):
            ax.text(col, r, f"{int(v)}", ha="center", va="center", fontsize=9,
                    color="white" if Mn[r, col] > 0.55 else "#333")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["음","양"], fontsize=8); ax.set_yticklabels(["음","양"], fontsize=8)
        ax.set_title(names[k], fontsize=9.5, pad=3)
        if k % 4 == 0: ax.set_ylabel("실제", fontsize=8)
        if k >= 3:     ax.set_xlabel("예측", fontsize=8)
    axes.flat[-1].set_axis_off()
    fig.suptitle("카테고리별 혼동행렬 (test, cycle41, 셀=건수)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "13_confusion.svg")); fig.savefig(os.path.join(OUT, "13_confusion.png"), dpi=300)
    plt.close(fig); print("  • 13_confusion")


def fig_calibration(probs, labels):
    """14_calibration — 신뢰도 보정 곡선 (전 카테고리 pooled) + ECE."""
    p = probs.ravel(); y = labels.ravel()
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1; idx = np.clip(idx, 0, 9)
    conf, acc, cnt = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() == 0: conf.append(np.nan); acc.append(np.nan); cnt.append(0); continue
        conf.append(p[m].mean()); acc.append(y[m].mean()); cnt.append(int(m.sum()))
    conf = np.array(conf); acc = np.array(acc); cnt = np.array(cnt)
    ece = np.nansum(cnt * np.abs(acc - conf)) / cnt.sum()

    fig, ax = plt.subplots(figsize=SZ)
    ax.plot([0,1],[0,1], color="#999", ls=":", lw=1.2, label="완벽 보정")
    valid = ~np.isnan(conf)
    ax.plot(conf[valid], acc[valid], color=BLUE, marker="o", ms=6, lw=1.8, label="모델 raw 확률")
    ax.bar(bins[:-1]+0.05, cnt/cnt.max()*0.25, width=0.09, color="#DDD", zorder=0, align="center")
    ax.set_xlabel("예측 확률 (raw prob)"); ax.set_ylabel("실제 양성 빈도")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.04, 0.92, f"ECE = {ece:.3f}\n(낮을수록 보정 우수)", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FBF4E6", ec=RED, lw=1.0))
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, edgecolor="#CCC")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title("신뢰도 보정 곡선 (raw 확률, cycle41)", fontsize=12, pad=10)
    fig.text(0.5, -0.02, "raw 확률이 대각선에 가까움 → 후처리 보정 불필요(제거 정당화)", ha="center", fontsize=7.5, color=GREEN)
    fig.savefig(os.path.join(OUT, "14_calibration.svg")); fig.savefig(os.path.join(OUT, "14_calibration.png"), dpi=300)
    plt.close(fig); print(f"  • 14_calibration  (ECE {ece:.3f})")


if __name__ == "__main__":
    print("PR/ROC figure 생성:")
    probs, labels = load_or_infer()
    plot_curves(probs, labels)
    fig_confusion(probs, labels)
    fig_calibration(probs, labels)
    print("→", OUT)
