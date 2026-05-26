"""카테고리별로 가장 잘하는 모델의 예측을 골라 합성하는 hybrid ensemble.

각 카테고리별로 val F1이 가장 좋은 모델의 test prob을 사용 → 상한 측정.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import torch, torch.nn as nn
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
    def __init__(self, m, n, d=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(m)
        self.dropout = nn.Dropout(d)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, n)
    def forward(self, i, a):
        o = self.backbone(input_ids=i, attention_mask=a)
        return self.classifier(self.dropout(o.last_hidden_state[:, 0]))


class AdsDataset(torch.utils.data.Dataset):
    def __init__(self, r, t, m): self.r, self.t, self.m = r, t, m
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        r = self.r[i]
        e = self.t(r["text"], truncation=True, padding="max_length", max_length=self.m, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(r["labels"], dtype=torch.float)}


def load_csv(p):
    rows = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try: labels = [int(r[c]) for c in CAT]
            except: continue
            if r["text"].strip():
                rows.append({"text": r["text"], "labels": labels})
    return rows


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    pl, ll = [], []
    for b in loader:
        pl.append(model(b["input_ids"].to(device), b["attention_mask"].to(device)).cpu())
        ll.append(b["labels"])
    return torch.sigmoid(torch.cat(pl)).numpy(), torch.cat(ll).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_rows = load_csv(DATA / "split_test.csv")
    val_rows = load_csv(DATA / "split_val.csv")
    print(f"Test {len(test_rows)} / Val {len(val_rows)}")

    out = RUNS / args.run_name; out.mkdir(parents=True, exist_ok=True)

    # 각 모델 prob 수집
    test_probs, val_probs = [], []
    cfgs = []
    for rn in args.runs:
        run = RUNS / rn
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        cfgs.append(cfg)
        print(f"\nLoading {rn}...")
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        model = AdsClassifier(cfg["model_name"], len(CAT), cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(run / "best_model.pt", weights_only=True, map_location=device))
        tl = DataLoader(AdsDataset(test_rows, tok, cfg["max_length"]), batch_size=cfg["batch_size"])
        vl = DataLoader(AdsDataset(val_rows, tok, cfg["max_length"]), batch_size=cfg["batch_size"])
        tp, tlbl = predict(model, tl, device)
        vp, vlbl = predict(model, vl, device)
        test_probs.append(tp); val_probs.append(vp)

    # 각 카테고리별 val F1 best 모델 선택
    print(f"\n=== 카테고리별 val 최고 모델 ===")
    chosen = []
    for j, c in enumerate(CAT):
        best_f1, best_idx, best_t = 0, 0, 0.5
        for i in range(len(args.runs)):
            for t in np.arange(0.2, 0.71, 0.02):
                preds = (val_probs[i][:, j] >= t).astype(int)
                if vlbl[:, j].sum() == 0: continue
                f1 = f1_score(vlbl[:, j], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_idx, best_t = f1, i, t
        chosen.append((best_idx, best_t))
        print(f"  {c:<14}: {args.runs[best_idx]} @ thr {best_t:.2f} (val F1 {best_f1:.3f})")

    # test 예측: 각 카테고리는 선택된 모델 사용
    final_preds = np.zeros((len(test_rows), len(CAT)), dtype=int)
    final_probs = np.zeros((len(test_rows), len(CAT)))
    for j in range(len(CAT)):
        idx, thr = chosen[j]
        final_probs[:, j] = test_probs[idx][:, j]
        final_preds[:, j] = (test_probs[idx][:, j] >= thr).astype(int)

    # 평가
    per_f1, per_p, per_r = [], [], []
    for j, c in enumerate(CAT):
        per_f1.append(f1_score(tlbl[:, j], final_preds[:, j], zero_division=0))
        per_p.append(precision_score(tlbl[:, j], final_preds[:, j], zero_division=0))
        per_r.append(recall_score(tlbl[:, j], final_preds[:, j], zero_division=0))
    macro_f1 = float(np.mean(per_f1))
    micro_f1 = f1_score(tlbl, final_preds, average="micro", zero_division=0)
    intensity_pred = np.array([sum(WEIGHTS[c]*final_probs[i, j] for j, c in enumerate(CAT)) for i in range(len(test_rows))])
    intensity_true = np.array([sum(WEIGHTS[c]*tlbl[i, j] for j, c in enumerate(CAT)) for i in range(len(test_rows))])
    mae = float(np.mean(np.abs(intensity_pred - intensity_true)))

    print(f"\n=== Hybrid (카테고리별 best 모델) 결과 ===")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Test Micro F1: {float(micro_f1):.4f}")
    print(f"Test Intensity MAE: {mae:.4f}")
    for j, c in enumerate(CAT):
        print(f"  {c:<14} F1 {per_f1[j]:.3f} | P {per_p[j]:.3f} | R {per_r[j]:.3f}")

    result = {
        "run_name": args.run_name,
        "hybrid_choices": {CAT[j]: {"model": args.runs[chosen[j][0]], "threshold": chosen[j][1]} for j in range(len(CAT))},
        "test_macro_f1": macro_f1,
        "test_micro_f1": float(micro_f1),
        "test_per_class_f1": dict(zip(CAT, per_f1)),
        "test_per_class_precision": dict(zip(CAT, per_p)),
        "test_per_class_recall": dict(zip(CAT, per_r)),
        "test_intensity_mae": mae,
        "weights": WEIGHTS,
        "config": {"hybrid": True, "n_models": len(args.runs)},
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {out}/result.json")


if __name__ == "__main__":
    main()
