"""Multitask 모델 + scoring_v2 통합 추론 CLI.

사용:
  python 모델학습/predict_v3.py --text "오늘만 50% 할인 무료배송"
  python 모델학습/predict_v3.py --text-file inputs.txt
  python 모델학습/predict_v3.py --text "..." --run cycle50_ensemble_41_48_39_46
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3_multitask import (
    MultiTaskClassifier, CAT, POLARITY, INTENSITY,
    WEIGHTS, POLARITY_SCORE, INTENSITY_SCORE,
)
from scoring_v2 import (
    CategoryResult, calculate_ad_score, calibrate_confidence, synergy_factor,
)

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "모델학습" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_single_model(run_name):
    run_dir = RUNS / run_name
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = MultiTaskClassifier(cfg["model"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True, map_location=DEVICE))
    model.eval()
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return tok, model, cfg, result.get("best_thresholds", {})


def load_ensemble(ensemble_name="cycle50_ensemble_41_48_39_46"):
    """Cycle 50 4-모델 앙상블."""
    members = ["cycle41_taskweight_cat", "cycle48_seed7", "cycle39_taskweight_balanced", "cycle46_dropout0.2"]
    # 첫 모델로 threshold + cfg
    result = json.loads((RUNS / ensemble_name / "result.json").read_text(encoding="utf-8"))
    thresholds = result.get("thresholds", {})
    loaded = []
    for m in members:
        tok, model, cfg, _ = load_single_model(m)
        loaded.append((tok, model, cfg))
    return loaded, thresholds


@torch.no_grad()
def predict_one(loaded, text, thresholds):
    """앙상블 (또는 단일) 추론. loaded: [(tok, model, cfg)]."""
    cat_probs_acc = None
    pol_probs_acc = None
    int_probs_acc = None
    n = len(loaded)
    for tok, model, cfg in loaded:
        enc = tok(text, truncation=True, padding="max_length", max_length=cfg["max_length"], return_tensors="pt")
        cat_l, pol_l, int_l = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        cp = torch.sigmoid(cat_l).cpu().numpy()[0]
        pp = F.softmax(pol_l, dim=-1).cpu().numpy()[0]
        ip = F.softmax(int_l, dim=-1).cpu().numpy()[0]
        cat_probs_acc = cp if cat_probs_acc is None else cat_probs_acc + cp
        pol_probs_acc = pp if pol_probs_acc is None else pol_probs_acc + pp
        int_probs_acc = ip if int_probs_acc is None else int_probs_acc + ip
    cat_probs = cat_probs_acc / n
    pol_probs = pol_probs_acc / n
    int_probs = int_probs_acc / n

    # Threshold 적용
    thr = np.array([thresholds.get(c, 0.5) for c in CAT])
    cat_preds = (cat_probs >= thr).astype(int)

    # 카테고리별 polarity·intensity argmax
    pol_preds = pol_probs.argmax(-1)
    int_preds = int_probs.argmax(-1)

    # CategoryResult 생성
    results = []
    for i, c in enumerate(CAT):
        results.append(CategoryResult(
            category=c,
            probability=float(cat_probs[i]),
            is_positive=bool(cat_preds[i]),
            polarity=POLARITY[pol_preds[i]] if cat_preds[i] else None,
            intensity=INTENSITY[int_preds[i]] if cat_preds[i] else None,
        ))

    # 점수 계산
    score_result = calculate_ad_score(results)

    # 상세 출력
    return {
        "text": text,
        "score": score_result,
        "details": [
            {
                "category": r.category,
                "probability": round(r.probability, 4),
                "threshold": round(thr[i], 3),
                "is_positive": r.is_positive,
                "polarity": r.polarity,
                "intensity": r.intensity,
                "calibrated_confidence": round(r.calibrated_confidence, 4) if r.is_positive else 0.0,
                "score_contribution": round(r.score, 4) if r.is_positive else 0.0,
            } for i, r in enumerate(results)
        ],
    }


def format_output(result):
    s = result["score"]
    lines = [
        "=" * 70,
        f"📝 텍스트: {result['text'][:100]}",
        "=" * 70,
        f"💯 최종 점수: {s['final_score_100']}/100",
        f"   양성 카테고리: {s['n_positive_categories']}개 → 시너지 ×{s['synergy_factor']}",
        f"   raw score: {s['raw_score']}",
        "",
        f"{'카테고리':<15} {'확률':>7} {'thr':>5} {'양성':>5} {'방향':>5} {'강도':>5} {'점수':>10}",
        "-" * 70,
    ]
    for d in result["details"]:
        pos = "✓" if d["is_positive"] else ""
        pol = d["polarity"] or "-"
        ints = d["intensity"] or "-"
        sc = f"{d['score_contribution']:+.4f}" if d["is_positive"] else "-"
        lines.append(f"{d['category']:<15} {d['probability']:>7.3f} {d['threshold']:>5.2f} {pos:>5} {pol:>5} {ints:>5} {sc:>10}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--run", default="cycle50_ensemble_41_48_39_46", help="Run name (default: 4-모델 앙상블)")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    args = ap.parse_args()

    if not args.text and not args.text_file:
        print("❌ --text 또는 --text-file 필요"); sys.exit(1)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            texts.extend([l.strip() for l in f if l.strip()])

    print(f"📥 모델 로드: {args.run}")
    if "ensemble" in args.run:
        loaded, thresholds = load_ensemble(args.run)
    else:
        tok, model, cfg, thresholds = load_single_model(args.run)
        loaded = [(tok, model, cfg)]
    print(f"✅ {len(loaded)} 모델 로드 / {DEVICE}\n")

    for text in texts:
        result = predict_one(loaded, text, thresholds)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_output(result))
            print()


if __name__ == "__main__":
    main()
