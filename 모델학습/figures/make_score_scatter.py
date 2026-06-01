"""
15_score_scatter — 예측 강도 점수 vs 정답 점수 산점도 (회귀 검증).

채택 공식(신뢰도 보정 OFF) + 4-모델 앙상블 = scoring_ablation/headline(MAE 5.80, r 0.838)과 동일.
예측·정답 점수는 score_scatter.npz로 캐시 → 재플롯 시 추론 생략.
실행: source .venv/bin/activate && python 모델학습/figures/make_score_scatter.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(__file__)
OUT  = os.path.join(HERE, "square"); os.makedirs(OUT, exist_ok=True)
ML   = os.path.join(HERE, "..")
PROJ = os.path.join(ML, "..")
RUNS = os.path.join(ML, "runs")
CACHE = os.path.join(HERE, "score_scatter.npz")
sys.path.insert(0, ML)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KR = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({"font.family": KR, "axes.unicode_minus": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.8, "savefig.bbox": "tight"})


def infer():
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from train_v3_multitask import MultiTaskClassifier, AdsDataset, load_csv, CAT
    from scoring_ablation import calc_score, true_score, MEMBERS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test_rows = load_csv(os.path.join(PROJ, "통합데이터셋", "split_test_v3.csv"))
    cat_acc = pol_acc = int_acc = None
    cat_lab = pol_lab = int_lab = None
    for mname in MEMBERS:
        rdir = os.path.join(RUNS, mname)
        cfg = json.load(open(os.path.join(rdir, "config.json"), encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        m = MultiTaskClassifier(cfg["model"], cfg["dropout"]).to(dev).eval()
        m.load_state_dict(torch.load(os.path.join(rdir, "best_model.pt"), weights_only=True, map_location=dev))
        loader = DataLoader(AdsDataset(test_rows, tok, cfg["max_length"]), batch_size=cfg["batch"])
        cps, pps, ips, cls, pls, ils = [], [], [], [], [], []
        with torch.no_grad():
            for b in loader:
                cl, pl, il = m(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                cps.append(torch.sigmoid(cl).cpu().numpy())
                pps.append(F.softmax(pl, -1).cpu().numpy()); ips.append(F.softmax(il, -1).cpu().numpy())
                cls.append(b["category"].numpy()); pls.append(b["polarity"].numpy()); ils.append(b["intensity"].numpy())
        cp, pp, ip = np.concatenate(cps), np.concatenate(pps), np.concatenate(ips)
        cat_acc = cp if cat_acc is None else cat_acc + cp
        pol_acc = pp if pol_acc is None else pol_acc + pp
        int_acc = ip if int_acc is None else int_acc + ip
        if cat_lab is None:
            cat_lab, pol_lab, int_lab = np.concatenate(cls), np.concatenate(pls), np.concatenate(ils)
        del m; torch.cuda.empty_cache(); print(f"  ✓ {mname}")
    cat_probs = cat_acc/len(MEMBERS); pol_probs = pol_acc/len(MEMBERS); int_probs = int_acc/len(MEMBERS)

    ens = json.load(open(os.path.join(RUNS, "cycle50_ensemble_41_48_39_46", "result.json"), encoding="utf-8"))
    thr = np.array([ens["thresholds"][c] for c in CAT])
    cat_preds = (cat_probs >= thr).astype(int)
    pol_preds = pol_probs.argmax(-1); int_preds = int_probs.argmax(-1)

    pred = np.array([calc_score(cat_probs[i], cat_preds[i], pol_preds[i], int_preds[i], use_calibration=False)
                     for i in range(len(test_rows))])
    true = np.array([true_score(cat_lab[i], pol_lab[i], int_lab[i]) for i in range(len(test_rows))])
    np.savez(CACHE, pred=pred, true=true); print("  저장:", CACHE)
    return pred, true


def load_or_infer():
    if os.path.exists(CACHE):
        d = np.load(CACHE); print("  캐시 사용:", CACHE); return d["pred"], d["true"]
    print("  앙상블 추론 (4 모델)…"); return infer()


def plot(pred, true):
    BLUE, RED = "#2C6FB5", "#C0504D"
    r = float(np.corrcoef(pred, true)[0, 1]); r2 = r**2
    mae = float(np.mean(np.abs(pred - true)))
    lim = max(pred.max(), true.max()) * 1.05
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, lim], [0, lim], color="#999", ls="--", lw=1.2, label="y = x (완벽)")
    ax.scatter(true, pred, s=14, alpha=0.35, color=BLUE, edgecolors="none")
    # 회귀선
    a, b = np.polyfit(true, pred, 1)
    xs = np.array([0, lim]); ax.plot(xs, a*xs+b, color=RED, lw=1.6, label=f"회귀선 (기울기 {a:.2f})")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("정답 강도 점수 (라벨 기반)"); ax.set_ylabel("예측 강도 점수 (모델)")
    ax.text(0.04*lim, 0.86*lim, f"r = {r:.3f}\nR² = {r2:.3f}\nMAE = {mae:.2f}", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FBF4E6", ec=RED, lw=1.1))
    ax.legend(loc="lower right", fontsize=9, frameon=True, edgecolor="#CCC")
    ax.set_aspect("equal")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("예측 vs 정답 강도 점수 (test, 앙상블)", fontsize=12, pad=10)
    fig.savefig(os.path.join(OUT, "15_score_scatter.svg")); fig.savefig(os.path.join(OUT, "15_score_scatter.png"), dpi=300)
    plt.close(fig); print(f"  • 15_score_scatter  (r={r:.3f}, MAE={mae:.2f})")


if __name__ == "__main__":
    print("점수 산점도 생성:")
    pred, true = load_or_infer()
    plot(pred, true)
    print("→", OUT)
