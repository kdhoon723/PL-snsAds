"""학습된 모델로 광고 텍스트 → 7 카테고리 확률 + 강도 점수 추론.

사용:
  python predict.py --text "오늘만 50% 할인! 무료배송"
  python predict.py --text-file inputs.txt
  python predict.py --run cycle23_ensemble_gamma1  # default: cycle 19 단일 (가장 빠름)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

PROJECT = Path("PROJECT_ROOT")
RUNS = PROJECT / "모델학습" / "runs"
CAT = ["사회적_정체성", "희소성", "긴급성", "사회적_증명", "가격비교", "권위_신뢰", "호혜성"]
WEIGHTS = {"권위_신뢰":0.221,"사회적_증명":0.159,"사회적_정체성":0.156,"호혜성":0.151,
           "희소성":0.126,"긴급성":0.096,"가격비교":0.091}


class AdsClassifier(nn.Module):
    def __init__(self, m, n, d=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(m)
        self.dropout = nn.Dropout(d)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, n)
    def forward(self, i, a):
        o = self.backbone(input_ids=i, attention_mask=a)
        return self.classifier(self.dropout(o.last_hidden_state[:, 0]))


def load_single_model(run_name: str, device):
    run = RUNS / run_name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AdsClassifier(cfg["model_name"], len(CAT), cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(run / "best_model.pt", weights_only=True, map_location=device))
    model.eval()
    thresholds = [result["best_thresholds"][c] for c in CAT]
    return tok, model, cfg["max_length"], thresholds


def load_ensemble(run_name: str, device):
    """ensemble run 결과는 model 가중치 없음 → cycle 23 = 19+13+18 hardcoded."""
    if run_name == "cycle23_ensemble_gamma1":
        members = ["cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]
    elif run_name == "cycle35_weighted_19x2":
        members = ["cycle19_dropout0.1", "cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]
    else:
        raise ValueError(f"Ensemble {run_name} 모름")
    models = []
    for m in members:
        models.append(load_single_model(m, device))
    # ensemble의 threshold는 result.json에서
    ens_res = json.loads((RUNS / run_name / "result.json").read_text(encoding="utf-8"))
    thresholds = [ens_res["thresholds"][c] for c in CAT]
    return models, thresholds


@torch.no_grad()
def predict_single(text: str, tok, model, max_len, device):
    enc = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return torch.sigmoid(logits).cpu().numpy()[0]


def intensity(probs):
    return float(sum(WEIGHTS[c] * probs[i] for i, c in enumerate(CAT)))


def report(text, probs, thresholds):
    preds = (probs >= np.array(thresholds)).astype(int)
    inten = intensity(probs)
    print(f"\n📄 텍스트: {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"\n{'카테고리':<14} {'확률':>8} {'임계':>6} {'예측':>4} {'가중치':>7}")
    print("-" * 50)
    for i, c in enumerate(CAT):
        marker = "✓" if preds[i] else " "
        print(f"{c:<14} {probs[i]:>8.3f} {thresholds[i]:>6.2f} {marker:>4} {WEIGHTS[c]:>7.3f}")
    print("-" * 50)
    print(f"📊 강도 점수: {inten:.4f}  (0.0~0.84 범위, 평균 0.146)")
    intensity_level = "강함" if inten >= 0.4 else "중간" if inten >= 0.2 else "약함" if inten >= 0.05 else "거의없음"
    print(f"🎯 자극 강도: {intensity_level}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="단일 광고 텍스트")
    ap.add_argument("--text-file", help="줄별 광고 텍스트 파일")
    ap.add_argument("--run", default="cycle19_dropout0.1",
                    help="모델 (default: cycle19, 단일 best). 앙상블: cycle23_ensemble_gamma1")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.run}")

    is_ensemble = "ensemble" in args.run or "weighted" in args.run
    if is_ensemble:
        models, thresholds = load_ensemble(args.run, device)
        print(f"  Ensemble: {len(models)} sub-models")
    else:
        tok, model, max_len, thresholds = load_single_model(args.run, device)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        texts.extend([l.strip() for l in Path(args.text_file).read_text(encoding="utf-8").splitlines() if l.strip()])
    if not texts:
        # 데모
        texts = [
            "오늘만 50% 할인! 무료배송 ✨",
            "갓생러들의 필수템! 직장인 100명이 선택",
            "박보영이 직접 추천한 우리 동네 맛집",
            "한정판 콜라보 굿즈, 선착순 300명",
        ]

    for text in texts:
        if is_ensemble:
            probs_list = []
            for tok, model, max_len, _ in models:
                probs_list.append(predict_single(text, tok, model, max_len, device))
            probs = np.mean(probs_list, axis=0)
        else:
            probs = predict_single(text, tok, model, max_len, device)
        report(text, probs, thresholds)


if __name__ == "__main__":
    main()
