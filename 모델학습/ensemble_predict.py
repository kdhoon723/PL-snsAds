"""여러 학습 모델의 test 예측을 평균(앙상블)해 평가.

사용:
  python ensemble_predict.py cycle9_focal_gamma1 cycle11_focal_gamma1.5 cycle8_focal_gamma2 \
      --run-name ensemble_top3
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

PROJECT = Path("PROJECT_ROOT")
DATA = PROJECT / "통합데이터셋"
RUNS = PROJECT / "모델학습" / "runs"
CAT = ["사회적_정체성","희소성","긴급성","사회적_증명","가격비교","권위_신뢰","호혜성"]
WEIGHTS = {"권위_신뢰":0.221,"사회적_증명":0.159,"사회적_정체성":0.156,"호혜성":0.151,"희소성":0.126,"긴급성":0.096,"가격비교":0.091}


class AdsClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0]))


class AdsDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(r["labels"], dtype=torch.float)}


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try: labels = [int(r[c]) for c in CAT]
            except: continue
            if not r["text"].strip(): continue
            rows.append({"text": r["text"], "labels": labels, "unified_id": r.get("unified_id","")})
    return rows


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        all_logits.append(logits.cpu()); all_labels.append(b["labels"])
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    return torch.sigmoid(logits).numpy(), labels


def tune_thresholds(probs, labels):
    ts = []
    for i in range(len(CAT)):
        best_f1, best_t = 0.0, 0.5
        if labels[:, i].sum() == 0:
            ts.append(0.5); continue
        for t in np.arange(0.2, 0.71, 0.02):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        ts.append(float(best_t))
    return ts


def evaluate(probs, labels, thresholds):
    preds = (probs >= np.array(thresholds)).astype(int)
    per_f1, per_p, per_r = [], [], []
    for i, c in enumerate(CAT):
        if labels[:, i].sum() == 0:
            per_f1.append(0); per_p.append(0); per_r.append(0); continue
        per_f1.append(f1_score(labels[:, i], preds[:, i], zero_division=0))
        per_p.append(precision_score(labels[:, i], preds[:, i], zero_division=0))
        per_r.append(recall_score(labels[:, i], preds[:, i], zero_division=0))
    macro_f1 = float(np.mean(per_f1))
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    intensity_pred = np.array([sum(WEIGHTS[c]*probs[i, j] for j, c in enumerate(CAT)) for i in range(len(probs))])
    intensity_true = np.array([sum(WEIGHTS[c]*labels[i, j] for j, c in enumerate(CAT)) for i in range(len(labels))])
    mae = float(np.mean(np.abs(intensity_pred - intensity_true)))
    return {
        "macro_f1": macro_f1, "micro_f1": float(micro_f1),
        "per_class_f1": dict(zip(CAT, per_f1)),
        "per_class_precision": dict(zip(CAT, per_p)),
        "per_class_recall": dict(zip(CAT, per_r)),
        "test_intensity_mae": mae,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="앙상블할 run 디렉토리 이름들")
    ap.add_argument("--run-name", required=True, help="결과 저장 이름")
    ap.add_argument("--data-dir", default=str(DATA))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    test_rows = load_csv(data_dir / "split_test.csv")
    val_rows = load_csv(data_dir / "split_val.csv")
    print(f"Test {len(test_rows)} / Val {len(val_rows)}")

    out = RUNS / args.run_name; out.mkdir(parents=True, exist_ok=True)

    all_test_probs, all_val_probs = [], []
    cfgs = []
    for run_name in args.runs:
        run = RUNS / run_name
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        cfgs.append({"name": run_name, "model": cfg["model_name"], "max_length": cfg["max_length"]})
        print(f"\n=== {run_name} ({cfg['model_name']}, max_len {cfg['max_length']}) ===")
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        model = AdsClassifier(cfg["model_name"], len(CAT), cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(run / "best_model.pt", weights_only=True, map_location=device))
        test_loader = DataLoader(AdsDataset(test_rows, tok, cfg["max_length"]), batch_size=cfg["batch_size"])
        val_loader = DataLoader(AdsDataset(val_rows, tok, cfg["max_length"]), batch_size=cfg["batch_size"])
        tp, tl = predict_probs(model, test_loader, device)
        vp, vl = predict_probs(model, val_loader, device)
        all_test_probs.append(tp); all_val_probs.append(vp)

    # 평균 앙상블
    test_avg = np.mean(all_test_probs, axis=0)
    val_avg = np.mean(all_val_probs, axis=0)
    thr = tune_thresholds(val_avg, vl)
    result = evaluate(test_avg, tl, thr)

    print(f"\n=== 앙상블 결과 ({len(args.runs)} models) ===")
    print(f"Test Macro F1: {result['macro_f1']:.4f}")
    print(f"Test Micro F1: {result['micro_f1']:.4f}")
    print(f"Test Intensity MAE: {result['test_intensity_mae']:.4f}")
    for c in CAT:
        print(f"  {c:<14} F1 {result['per_class_f1'][c]:.3f} | P {result['per_class_precision'][c]:.3f} | R {result['per_class_recall'][c]:.3f}")

    # 저장
    saved = {
        "run_name": args.run_name,
        "ensemble_members": cfgs,
        "thresholds": dict(zip(CAT, thr)),
        "test_macro_f1": result["macro_f1"],
        "test_micro_f1": result["micro_f1"],
        "test_per_class_f1": result["per_class_f1"],
        "test_per_class_precision": result["per_class_precision"],
        "test_per_class_recall": result["per_class_recall"],
        "test_intensity_mae": result["test_intensity_mae"],
        "config": {"ensemble_size": len(args.runs)},
        "weights": WEIGHTS,
    }
    (out / "result.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {out}/result.json")


if __name__ == "__main__":
    main()
