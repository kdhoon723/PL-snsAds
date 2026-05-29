"""점수 공식 Ablation — 보정 공식의 각 항이 점수에 미치는 영향 분리 측정.

공식: score = Σ(w × ĉ × p × t) × synergy(n) × 100

변형:
1. Full (PDF + 팀원 A 융합)
2. -신뢰도 보정 (ĉ → prob)
3. -방향성 (p → 1.0)
4. -강도 (t → 1.0)
5. -시너지 (synergy → 1.0)
6. v1 단순 공식 Σ(w × prob) (0~1 → ×100)

평가:
- Test 922건 점수 분포 (mean, std, min, max)
- True 점수 (라벨 기반)와 MAE
- 변형별 점수 변화 크기
"""
import json, sys, csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, "PROJECT_ROOT/모델학습")
from train_v3_multitask import (
    MultiTaskClassifier, AdsDataset, load_csv,
    CAT, POLARITY, INTENSITY, WEIGHTS,
    POLARITY_SCORE, INTENSITY_SCORE,
)
from scoring_v2 import (
    CategoryResult, synergy_factor, calibrate_confidence,
)

PROJECT = Path("PROJECT_ROOT")
DATA = PROJECT / "통합데이터셋"
RUNS = PROJECT / "모델학습" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEMBERS = ["cycle41_taskweight_cat", "cycle48_seed7", "cycle39_taskweight_balanced", "cycle46_dropout0.2"]


def calc_score(cat_probs, cat_preds, pol_preds, int_preds,
               use_calibration=True, use_polarity=True, use_intensity=True, use_synergy=True):
    """카테고리별 예측에서 점수 계산. 각 항 toggle."""
    n_pos = int(cat_preds.sum())
    syn = synergy_factor(n_pos) if use_synergy else 1.0
    raw = 0.0
    for i, c in enumerate(CAT):
        if not cat_preds[i]:
            continue
        prob = float(cat_probs[i])
        c_hat = calibrate_confidence(c, prob) if use_calibration else prob
        p = POLARITY_SCORE[POLARITY[pol_preds[i]]] if use_polarity else 1.0
        t = INTENSITY_SCORE[INTENSITY[int_preds[i]]] if use_intensity else 1.0
        raw += WEIGHTS[c] * c_hat * p * t
    return raw * syn * 100


def simple_v1_score(cat_probs):
    """v1 단순 공식 Σ(w × prob) × 100 (0~84 점)."""
    return float(sum(WEIGHTS[c] * cat_probs[i] for i, c in enumerate(CAT))) * 100


def true_score(cat_label, pol_label, int_label):
    """라벨 기반 정답 점수."""
    n_pos = int(cat_label.sum())
    syn = synergy_factor(n_pos)
    raw = 0.0
    for i, c in enumerate(CAT):
        if cat_label[i] != 1:
            continue
        # mask -100인 경우 기본값
        p = POLARITY_SCORE[POLARITY[pol_label[i]]] if pol_label[i] != -100 else 1.0
        t = INTENSITY_SCORE[INTENSITY[int_label[i]]] if int_label[i] != -100 else 1.0
        raw += WEIGHTS[c] * 1.0 * p * t  # ĉ=1 for true
    return raw * syn * 100


@torch.no_grad()
def main():
    test_rows = load_csv(DATA / "split_test_v3.csv")
    val_rows = load_csv(DATA / "split_val_v3.csv")
    print(f"Test: {len(test_rows)}")

    # 4-앙상블 추론
    cat_probs_acc = pol_probs_acc = int_probs_acc = None
    cat_labels = pol_labels = int_labels = None
    for mname in MEMBERS:
        rdir = RUNS / mname
        cfg = json.loads((rdir / "config.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        m = MultiTaskClassifier(cfg["model"], cfg["dropout"]).to(DEVICE)
        m.load_state_dict(torch.load(rdir / "best_model.pt", weights_only=True, map_location=DEVICE))
        m.eval()
        ds = AdsDataset(test_rows, tok, cfg["max_length"])
        loader = DataLoader(ds, batch_size=cfg["batch"])
        cps, pps, ips, cls, pls, ils = [], [], [], [], [], []
        for b in loader:
            cl, pl, il = m(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            cps.append(torch.sigmoid(cl).cpu().numpy())
            pps.append(F.softmax(pl, dim=-1).cpu().numpy())
            ips.append(F.softmax(il, dim=-1).cpu().numpy())
            cls.append(b["category"].numpy()); pls.append(b["polarity"].numpy()); ils.append(b["intensity"].numpy())
        cp = np.concatenate(cps); pp = np.concatenate(pps); ip = np.concatenate(ips)
        cat_probs_acc = cp if cat_probs_acc is None else cat_probs_acc + cp
        pol_probs_acc = pp if pol_probs_acc is None else pol_probs_acc + pp
        int_probs_acc = ip if int_probs_acc is None else int_probs_acc + ip
        if cat_labels is None:
            cat_labels = np.concatenate(cls); pol_labels = np.concatenate(pls); int_labels = np.concatenate(ils)
        del m; torch.cuda.empty_cache()
        print(f"  ✓ {mname}")

    cat_probs = cat_probs_acc / len(MEMBERS)
    pol_probs = pol_probs_acc / len(MEMBERS)
    int_probs = int_probs_acc / len(MEMBERS)

    # Threshold from cycle 50 result
    ens_result = json.loads((RUNS / "cycle50_ensemble_41_48_39_46" / "result.json").read_text(encoding="utf-8"))
    thr = np.array([ens_result["thresholds"][c] for c in CAT])
    cat_preds = (cat_probs >= thr).astype(int)
    pol_preds = pol_probs.argmax(-1)
    int_preds = int_probs.argmax(-1)

    # 각 변형으로 점수 계산
    variants = {
        "Full (PDF+팀원 A)": lambda i: calc_score(cat_probs[i], cat_preds[i], pol_preds[i], int_preds[i]),
        "-신뢰도 보정": lambda i: calc_score(cat_probs[i], cat_preds[i], pol_preds[i], int_preds[i], use_calibration=False),
        "-방향성": lambda i: calc_score(cat_probs[i], cat_preds[i], pol_preds[i], int_preds[i], use_polarity=False),
        "-강도": lambda i: calc_score(cat_probs[i], cat_preds[i], pol_preds[i], int_preds[i], use_intensity=False),
        "-시너지": lambda i: calc_score(cat_probs[i], cat_preds[i], pol_preds[i], int_preds[i], use_synergy=False),
        "v1 단순 (Σ w·prob)×100": lambda i: simple_v1_score(cat_probs[i]),
    }
    true_scores = np.array([true_score(cat_labels[i], pol_labels[i], int_labels[i]) for i in range(len(cat_labels))])

    print(f"\n{'=' * 80}")
    print(f"📊 점수 공식 Ablation (Test 922건)")
    print(f"{'=' * 80}")
    print(f"True 점수 분포: mean={true_scores.mean():.2f}, std={true_scores.std():.2f}, min={true_scores.min():.2f}, max={true_scores.max():.2f}")
    print()
    print(f"{'변형':<24} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'MAE vs true':>12} {'pred vs true r':>15}")
    print("-" * 95)
    results = {}
    for name, fn in variants.items():
        scores = np.array([fn(i) for i in range(len(test_rows))])
        mae = float(np.mean(np.abs(scores - true_scores)))
        # 상관관계
        if scores.std() > 0 and true_scores.std() > 0:
            corr = float(np.corrcoef(scores, true_scores)[0, 1])
        else:
            corr = 0.0
        print(f"{name:<24} {scores.mean():>7.2f} {scores.std():>7.2f} {scores.min():>7.2f} {scores.max():>7.2f} {mae:>12.4f} {corr:>15.4f}")
        results[name] = {
            "mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "max": float(scores.max()),
            "mae_vs_true": mae, "corr_vs_true": corr,
        }

    # 저장
    out = PROJECT / "모델학습" / "scoring_ablation_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_test": len(test_rows),
            "true_score": {
                "mean": float(true_scores.mean()), "std": float(true_scores.std()),
                "min": float(true_scores.min()), "max": float(true_scores.max()),
            },
            "variants": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {out}")

    # 해석
    print("\n" + "=" * 80)
    print("📝 해석")
    print("=" * 80)
    base = results["Full (PDF+팀원 A)"]["mae_vs_true"]
    print(f"기준 (Full 공식) MAE: {base:.4f}")
    for name, r in results.items():
        if name == "Full (PDF+팀원 A)":
            continue
        diff = r["mae_vs_true"] - base
        sign = "악화" if diff > 0 else "개선"
        print(f"  {name:<24}: MAE Δ {diff:+.4f} ({sign})")


if __name__ == "__main__":
    main()
