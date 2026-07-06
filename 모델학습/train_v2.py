"""KLUE-RoBERTa 다중 라벨 분류 학습 v2

통합데이터셋/ 입력, 자동 사이클을 위한 결과 저장 강화.

사용:
  python train_v2.py [--run-name NAME] [--model MODEL] [--epochs N] [--lr LR] [--batch B]
                     [--max-length L] [--pos-weight-scale S]

출력:
  모델학습/runs/{run_name}/
    ├── config.json           실행 설정
    ├── result.json           최종 메트릭 + 카테고리별 F1
    ├── best_model.pt         가중치 (gitignored)
    ├── history.json          epoch별 학습 곡선
    └── test_predictions.csv  test 셋 예측 + 정답 (오류 분석용)
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "통합데이터셋"
TRAIN_CSV = DATA / "split_train.csv"
VAL_CSV = DATA / "split_val.csv"
TEST_CSV = DATA / "split_test.csv"
RUNS_DIR = PROJECT / "모델학습" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["사회적_정체성", "희소성", "긴급성", "사회적_증명", "가격비교", "권위_신뢰", "호혜성"]
NUM_LABELS = len(CAT_COLS)

# 학술 가중치 (강도 점수)
WEIGHTS = {
    "권위_신뢰": 0.221, "사회적_증명": 0.159, "사회적_정체성": 0.156,
    "호혜성": 0.151, "희소성": 0.126, "긴급성": 0.096, "가격비교": 0.091,
}


def load_csv(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                labels = [int(r[c]) for c in CAT_COLS]
            except (KeyError, ValueError):
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            rows.append({"text": text, "labels": labels, "unified_id": r.get("unified_id", "")})
    return rows


class AdsDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(r["labels"], dtype=torch.float),
        }


class AdsClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0]))


def compute_pos_weights(train_data, num_labels, scale=1.0):
    counts = [0] * num_labels
    n = len(train_data)
    for r in train_data:
        for i, l in enumerate(r["labels"]):
            counts[i] += l
    weights = [(n - c) / max(c, 1) if c > 0 else 1.0 for c in counts]
    return torch.tensor([w * scale for w in weights], dtype=torch.float)


@torch.no_grad()
def evaluate(model, loader, device, thresholds=None):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_logits.append(logits.cpu()); all_labels.append(batch["labels"])
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.sigmoid(logits).numpy()
    if thresholds is None:
        thresholds = [0.5] * NUM_LABELS
    preds = (probs >= np.array(thresholds)).astype(int)
    per_cls_f1, per_cls_prec, per_cls_rec = [], [], []
    for i, cat in enumerate(CAT_COLS):
        if labels[:, i].sum() == 0:
            f1 = p = r = 0.0
        else:
            f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
            p = precision_score(labels[:, i], preds[:, i], zero_division=0)
            r = recall_score(labels[:, i], preds[:, i], zero_division=0)
        per_cls_f1.append(f1); per_cls_prec.append(p); per_cls_rec.append(r)
    macro_f1 = float(np.mean(per_cls_f1))
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "per_class_f1": dict(zip(CAT_COLS, per_cls_f1)),
        "per_class_precision": dict(zip(CAT_COLS, per_cls_prec)),
        "per_class_recall": dict(zip(CAT_COLS, per_cls_rec)),
        "probs": probs, "labels": labels,
    }


def tune_thresholds(probs, labels):
    thresholds = []
    for i in range(NUM_LABELS):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.2, 0.71, 0.02):
            preds = (probs[:, i] >= t).astype(int)
            if labels[:, i].sum() == 0:
                continue
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(float(best_t))
    return thresholds


def intensity(probs_row):
    return float(sum(WEIGHTS[c] * probs_row[i] for i, c in enumerate(CAT_COLS)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pos-weight-scale", type=float, default=1.0,
                    help="Multiplier on pos_weight (>1 = stronger minority emphasis)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--academic-weighted", action="store_true",
                    help="BCE loss에 학술 가중치(WEIGHTS) 곱하기 (도메인 중요도 반영)")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="Focal Loss γ (0=일반 BCE, 2.0=표준 focal, 어려운 샘플 강조)")
    ap.add_argument("--data-dir", default=str(DATA),
                    help="split_{train,val,test}.csv가 있는 디렉토리 (default: 통합데이터셋)")
    ap.add_argument("--warmup-ratio", type=float, default=0.1, help="warmup ratio")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    return ap.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"run_{int(time.time())}"
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "split_train.csv"
    val_csv = data_dir / "split_val.csv"
    test_csv = data_dir / "split_test.csv"

    # Seed
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    cfg = {
        "model_name": args.model,
        "max_length": args.max_length,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "patience": args.patience,
        "pos_weight_scale": args.pos_weight_scale,
        "academic_weighted": args.academic_weighted,
        "focal_gamma": args.focal_gamma,
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
    }
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {run_name} ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print("\n=== 데이터 로드 ===")
    train = load_csv(train_csv); val = load_csv(val_csv); test = load_csv(test_csv)
    print(f"Train {len(train)} / Val {len(val)} / Test {len(test)} (from {data_dir})")
    if not train:
        print("❌ Train 데이터 없음"); return

    device = cfg["device"]
    print(f"\n=== 모델 초기화 ({cfg['model_name']}) ===")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AdsClassifier(cfg["model_name"], NUM_LABELS, cfg["dropout"]).to(device)

    train_loader = DataLoader(AdsDataset(train, tok, cfg["max_length"]), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(AdsDataset(val, tok, cfg["max_length"]), batch_size=cfg["batch_size"])
    test_loader = DataLoader(AdsDataset(test, tok, cfg["max_length"]), batch_size=cfg["batch_size"])

    pos_w = compute_pos_weights(train, NUM_LABELS, scale=cfg["pos_weight_scale"]).to(device)
    print(f"pos_weights (×{cfg['pos_weight_scale']}): {[round(v, 2) for v in pos_w.cpu().tolist()]}")
    academic_t = None
    if cfg.get("academic_weighted"):
        academic_t = torch.tensor([WEIGHTS[c] for c in CAT_COLS], dtype=torch.float)
        academic_t = (academic_t / academic_t.mean()).to(device)
        print(f"academic_weights (학술 가중치 ↑): {[round(v, 3) for v in academic_t.cpu().tolist()]}")

    if cfg.get("focal_gamma", 0.0) > 0:
        gamma = cfg["focal_gamma"]
        print(f"Focal Loss γ={gamma} (어려운 샘플 강조)")
        def loss_fn(logits, labels):
            # multi-label focal with pos_weight + optional academic weight
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_w, reduction="none"
            )
            p = torch.sigmoid(logits)
            pt = p * labels + (1 - p) * (1 - labels)  # 정답 확률
            focal_term = (1 - pt).pow(gamma)
            loss = focal_term * bce
            if academic_t is not None:
                loss = loss * academic_t
            return loss.mean()
    elif academic_t is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w, weight=academic_t)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optim = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * cfg["warmup_ratio"]), total_steps)

    print(f"\n=== 학습 시작 ({cfg['epochs']} epochs) ===")
    best_f1, best_thr, patience_cnt = 0.0, None, 0
    history = []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        ep_loss = 0
        for batch in train_loader:
            optim.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(train_loader)

        val_m = evaluate(model, val_loader, device)
        thr = tune_thresholds(val_m["probs"], val_m["labels"])
        val_tuned = evaluate(model, val_loader, device, thr)

        hist_entry = {
            "epoch": ep + 1, "train_loss": avg_loss,
            "val_macro_f1_default": val_m["macro_f1"],
            "val_macro_f1_tuned": val_tuned["macro_f1"],
            "val_micro_f1_tuned": val_tuned["micro_f1"],
            "val_per_class_f1_tuned": val_tuned["per_class_f1"],
            "thresholds": thr,
            "epoch_sec": round(time.time() - t0, 1),
        }
        history.append(hist_entry)
        print(f"\nEpoch {ep+1}/{cfg['epochs']} | Loss {avg_loss:.4f} | "
              f"Val MacroF1 {val_tuned['macro_f1']:.4f} (def {val_m['macro_f1']:.4f}) | "
              f"{hist_entry['epoch_sec']}s")
        for c, f in val_tuned["per_class_f1"].items():
            print(f"  {c:<14}: {f:.3f}")

        if val_tuned["macro_f1"] > best_f1:
            best_f1, best_thr = val_tuned["macro_f1"], thr
            patience_cnt = 0
            torch.save(model.state_dict(), out / "best_model.pt")
            print("  ✅ best 갱신")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"  ⏹  Early stopping (patience={cfg['patience']})")
                break

    # 최종 평가
    print("\n=== 최종 평가 (Test, best 모델 + tuned thresholds) ===")
    model.load_state_dict(torch.load(out / "best_model.pt", weights_only=True))
    test_m = evaluate(model, test_loader, device, best_thr)
    print(f"Test Macro F1: {test_m['macro_f1']:.4f}")
    print(f"Test Micro F1: {test_m['micro_f1']:.4f}")
    print("카테고리별 F1 / Precision / Recall:")
    for c in CAT_COLS:
        print(f"  {c:<14} F1 {test_m['per_class_f1'][c]:.3f} | "
              f"P {test_m['per_class_precision'][c]:.3f} | "
              f"R {test_m['per_class_recall'][c]:.3f}")

    # 강도 점수 (test 셋 전체)
    intensity_preds = [intensity(p) for p in test_m["probs"]]
    intensity_true = [sum(WEIGHTS[c] * test_m["labels"][i][j] for j, c in enumerate(CAT_COLS))
                      for i in range(len(test_m["labels"]))]
    mae_intensity = float(np.mean(np.abs(np.array(intensity_preds) - np.array(intensity_true))))
    print(f"\n강도 점수 MAE (test): {mae_intensity:.4f}")

    # test 예측 저장 (오류 분석용)
    pred_path = out / "test_predictions.csv"
    with pred_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unified_id", "text"] + [f"true_{c}" for c in CAT_COLS]
                   + [f"pred_{c}" for c in CAT_COLS]
                   + [f"prob_{c}" for c in CAT_COLS]
                   + ["intensity_true", "intensity_pred"])
        for i, r in enumerate(test):
            pred = (test_m["probs"][i] >= np.array(best_thr)).astype(int).tolist()
            w.writerow([r["unified_id"], r["text"]] + r["labels"] + pred
                       + [round(float(p), 4) for p in test_m["probs"][i]]
                       + [round(intensity_true[i], 4), round(intensity_preds[i], 4)])

    # 결과 JSON
    result = {
        "run_name": run_name,
        "config": cfg,
        "best_val_macro_f1": best_f1,
        "best_thresholds": dict(zip(CAT_COLS, best_thr)),
        "test_macro_f1": test_m["macro_f1"],
        "test_micro_f1": test_m["micro_f1"],
        "test_per_class_f1": test_m["per_class_f1"],
        "test_per_class_precision": test_m["per_class_precision"],
        "test_per_class_recall": test_m["per_class_recall"],
        "test_intensity_mae": mae_intensity,
        "weights": WEIGHTS,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {out}/result.json")
    print(f"💾 {out}/history.json")
    print(f"💾 {out}/best_model.pt")
    print(f"💾 {pred_path}")


if __name__ == "__main__":
    main()
