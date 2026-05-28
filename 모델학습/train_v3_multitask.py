"""KLUE-RoBERTa 멀티태스크 학습 v3 (Cycle 38~)

카테고리 (multi-label sigmoid) + 카테고리별 방향성 (softmax) + 카테고리별 강도 (softmax)
동시 학습. PDF (가중치 계산.pdf) 설계 + 우리 검증된 multi-label 통합.

입력: 통합데이터셋/split_{train,val,test}_v3.csv

출력: 모델학습/runs/{run_name}/
  ├── config.json
  ├── result.json (cat/pol/int F1 + 강도점수 MAE)
  ├── best_model.pt
  └── history.json

사용:
  python train_v3_multitask.py --run-name cycle38_multitask_baseline \
    --model klue/roberta-large --max-length 256 --batch 8 --lr 1e-5 \
    --focal-gamma 1.0 --dropout 0.1 --epochs 15 --patience 5 \
    --task-weights 0.6,0.2,0.2
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np

PROJECT = Path("PROJECT_ROOT")
DATA = PROJECT / "통합데이터셋"
RUNS_DIR = PROJECT / "모델학습" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

CAT = ["사회적_정체성","희소성","긴급성","사회적_증명","가격비교","권위_신뢰","호혜성"]
NUM_CAT = len(CAT)
POLARITY = ["긍정", "중립", "부정"]
INTENSITY = ["강", "보통", "약"]
POL_IDX = {l: i for i, l in enumerate(POLARITY)}
INT_IDX = {l: i for i, l in enumerate(INTENSITY)}

# 학술 가중치 + 점수 공식 상수
WEIGHTS = {"권위_신뢰":0.221,"사회적_증명":0.159,"사회적_정체성":0.156,"호혜성":0.151,"희소성":0.126,"긴급성":0.096,"가격비교":0.091}
POLARITY_SCORE = {"긍정":1.0, "중립":0.5, "부정":-1.0}
INTENSITY_SCORE = {"강":1.5, "보통":1.0, "약":0.5}
CONFIDENCE_BASELINE = {"권위_신뢰":0.90,"사회적_증명":0.85,"희소성":0.85,"긴급성":0.85,"가격비교":0.80,"호혜성":0.75,"사회적_정체성":0.65}
MAX_CALIBRATION = 2.0


def synergy(n):
    if n <= 1: return 1.0
    if n == 2: return 1.15
    if n == 3: return 1.25
    return 0.90  # n >= 4


def calibrate(cat, conf):
    if cat not in CONFIDENCE_BASELINE: return conf
    if conf == 0: return 0.0
    return min(conf * min(CONFIDENCE_BASELINE[cat] / conf, MAX_CALIBRATION), 1.0)


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                cat_labels = [int(r[c]) for c in CAT]
            except (KeyError, ValueError):
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            # polarity / intensity per category — 음성 카테고리는 -1 (mask)
            pol_labels, int_labels = [], []
            for c in CAT:
                if int(r[c]) == 1:
                    pol_str = r.get(f"polarity_{c}", "긍정") or "긍정"
                    int_str = r.get(f"intensity_{c}", "보통") or "보통"
                    pol_labels.append(POL_IDX.get(pol_str, 0))
                    int_labels.append(INT_IDX.get(int_str, 1))
                else:
                    pol_labels.append(-100)  # CE ignore_index
                    int_labels.append(-100)
            rows.append({
                "text": text,
                "category": cat_labels,
                "polarity": pol_labels,
                "intensity": int_labels,
                "unified_id": r.get("unified_id", ""),
            })
    return rows


class AdsDataset(Dataset):
    def __init__(self, rows, tok, max_len=256):
        self.rows, self.tok, self.max_len = rows, tok, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category": torch.tensor(r["category"], dtype=torch.float),
            "polarity": torch.tensor(r["polarity"], dtype=torch.long),
            "intensity": torch.tensor(r["intensity"], dtype=torch.long),
        }


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cat_head = nn.Linear(h, NUM_CAT)               # multi-label sigmoid
        self.pol_head = nn.Linear(h, NUM_CAT * len(POLARITY))  # 카테고리별 polarity softmax
        self.int_head = nn.Linear(h, NUM_CAT * len(INTENSITY)) # 카테고리별 intensity softmax

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        cat_logits = self.cat_head(cls)                                          # (B, 7)
        pol_logits = self.pol_head(cls).view(-1, NUM_CAT, len(POLARITY))         # (B, 7, 3)
        int_logits = self.int_head(cls).view(-1, NUM_CAT, len(INTENSITY))        # (B, 7, 3)
        return cat_logits, pol_logits, int_logits


def compute_pos_weights(rows):
    counts = [0] * NUM_CAT
    n = len(rows)
    for r in rows:
        for i, l in enumerate(r["category"]):
            counts[i] += l
    return torch.tensor([(n - c) / max(c, 1) for c in counts], dtype=torch.float)


def multitask_loss(cat_logits, pol_logits, int_logits, batch, pos_weight, task_weights, focal_gamma=0.0):
    """카테고리 BCE (focal 옵션) + 방향성·강도 CE (양성 카테고리만)."""
    # Category
    if focal_gamma > 0:
        bce = nn.functional.binary_cross_entropy_with_logits(
            cat_logits, batch["category"], pos_weight=pos_weight, reduction="none"
        )
        p = torch.sigmoid(cat_logits)
        pt = p * batch["category"] + (1 - p) * (1 - batch["category"])
        cat_loss = ((1 - pt).pow(focal_gamma) * bce).mean()
    else:
        cat_loss = nn.functional.binary_cross_entropy_with_logits(
            cat_logits, batch["category"], pos_weight=pos_weight
        )

    # Polarity: (B, 7, 3) → flatten (B*7, 3) + label (B*7,)
    pol_labels = batch["polarity"].view(-1)
    pol_flat = pol_logits.reshape(-1, len(POLARITY))
    pol_loss = nn.functional.cross_entropy(pol_flat, pol_labels, ignore_index=-100)

    # Intensity
    int_labels = batch["intensity"].view(-1)
    int_flat = int_logits.reshape(-1, len(INTENSITY))
    int_loss = nn.functional.cross_entropy(int_flat, int_labels, ignore_index=-100)

    w_cat, w_pol, w_int = task_weights
    total = w_cat * cat_loss + w_pol * pol_loss + w_int * int_loss
    return total, {"cat": cat_loss.item(), "pol": pol_loss.item(), "int": int_loss.item()}


@torch.no_grad()
def evaluate(model, loader, device, thresholds=None):
    model.eval()
    all_cat_logits, all_cat_labels = [], []
    all_pol_preds, all_pol_labels = [], []
    all_int_preds, all_int_labels = [], []

    for batch in loader:
        cat_logits, pol_logits, int_logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        all_cat_logits.append(cat_logits.cpu())
        all_cat_labels.append(batch["category"])
        # polarity/intensity argmax
        all_pol_preds.append(pol_logits.argmax(-1).cpu())  # (B, 7)
        all_pol_labels.append(batch["polarity"])
        all_int_preds.append(int_logits.argmax(-1).cpu())
        all_int_labels.append(batch["intensity"])

    cat_logits = torch.cat(all_cat_logits, dim=0)
    cat_labels = torch.cat(all_cat_labels, dim=0).numpy()
    cat_probs = torch.sigmoid(cat_logits).numpy()
    if thresholds is None:
        thresholds = [0.5] * NUM_CAT
    cat_preds = (cat_probs >= np.array(thresholds)).astype(int)

    # category F1
    per_class_f1 = []
    for i, c in enumerate(CAT):
        if cat_labels[:, i].sum() == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(f1_score(cat_labels[:, i], cat_preds[:, i], zero_division=0))
    macro_f1 = float(np.mean(per_class_f1))

    # polarity/intensity F1 — 양성 카테고리만
    pol_preds = torch.cat(all_pol_preds, dim=0).numpy()  # (N, 7)
    pol_labels = torch.cat(all_pol_labels, dim=0).numpy()
    int_preds = torch.cat(all_int_preds, dim=0).numpy()
    int_labels = torch.cat(all_int_labels, dim=0).numpy()

    pol_per_cat, int_per_cat = {}, {}
    for i, c in enumerate(CAT):
        mask = pol_labels[:, i] != -100
        if mask.sum() == 0:
            pol_per_cat[c] = 0.0
            int_per_cat[c] = 0.0
            continue
        pol_per_cat[c] = f1_score(pol_labels[mask, i], pol_preds[mask, i], average="macro", zero_division=0)
        int_per_cat[c] = f1_score(int_labels[mask, i], int_preds[mask, i], average="macro", zero_division=0)
    pol_macro = float(np.mean(list(pol_per_cat.values())))
    int_macro = float(np.mean(list(int_per_cat.values())))

    # 강도 점수 MAE (PDF 공식, full formula)
    n = len(cat_labels)
    intensity_pred_scores = []
    intensity_true_scores = []
    for j in range(n):
        # True score
        true_n = int(cat_labels[j].sum())
        true_syn = synergy(true_n)
        true_score = 0.0
        for i, c in enumerate(CAT):
            if cat_labels[j, i] == 1:
                pl = POLARITY[pol_labels[j, i]] if pol_labels[j, i] != -100 else "긍정"
                il = INTENSITY[int_labels[j, i]] if int_labels[j, i] != -100 else "보통"
                true_score += WEIGHTS[c] * 1.0 * POLARITY_SCORE[pl] * INTENSITY_SCORE[il]  # ĉ=1 for true
        intensity_true_scores.append(true_score * true_syn)

        # Pred score
        pred_n = int(cat_preds[j].sum())
        pred_syn = synergy(pred_n)
        pred_score = 0.0
        for i, c in enumerate(CAT):
            if cat_preds[j, i] == 1:
                pl = POLARITY[pol_preds[j, i]]
                il = INTENSITY[int_preds[j, i]]
                cb = calibrate(c, float(cat_probs[j, i]))
                pred_score += WEIGHTS[c] * cb * POLARITY_SCORE[pl] * INTENSITY_SCORE[il]
        intensity_pred_scores.append(pred_score * pred_syn)

    score_mae = float(np.mean(np.abs(np.array(intensity_pred_scores) - np.array(intensity_true_scores))))

    return {
        "cat_macro_f1": macro_f1,
        "cat_per_class_f1": dict(zip(CAT, per_class_f1)),
        "pol_macro_f1": pol_macro,
        "pol_per_class_f1": pol_per_cat,
        "int_macro_f1": int_macro,
        "int_per_class_f1": int_per_cat,
        "score_mae": score_mae,
        "cat_probs": cat_probs,
        "cat_labels": cat_labels,
    }


def tune_thresholds(probs, labels):
    thresholds = []
    for i in range(NUM_CAT):
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="klue/roberta-large")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--focal-gamma", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--task-weights", default="0.6,0.2,0.2", help="cat,pol,int weights")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(DATA))
    return ap.parse_args()


def main():
    args = parse_args()
    task_w = tuple(float(x) for x in args.task_weights.split(","))
    assert len(task_w) == 3 and abs(sum(task_w) - 1.0) < 0.01, "task-weights는 3개 합 1.0"

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["task_weights"] = list(task_w)
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {args.run_name} ===")
    for k, v in cfg.items(): print(f"  {k}: {v}")

    # Seed
    import random as _r
    _r.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # 데이터
    data_dir = Path(args.data_dir)
    train = load_csv(data_dir / "split_train_v3.csv")
    val = load_csv(data_dir / "split_val_v3.csv")
    test = load_csv(data_dir / "split_test_v3.csv")
    print(f"Train {len(train)} / Val {len(val)} / Test {len(test)}")

    device = cfg["device"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = MultiTaskClassifier(args.model, args.dropout).to(device)

    train_loader = DataLoader(AdsDataset(train, tok, args.max_length), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(AdsDataset(val, tok, args.max_length), batch_size=args.batch)
    test_loader = DataLoader(AdsDataset(test, tok, args.max_length), batch_size=args.batch)

    pos_w = compute_pos_weights(train).to(device)
    print(f"pos_weights: {[round(v, 2) for v in pos_w.cpu().tolist()]}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    print(f"\n=== 학습 ({args.epochs} epochs) ===")
    best_f1, best_thr, patience_cnt = 0.0, None, 0
    history = []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        ep_loss = {"total": 0, "cat": 0, "pol": 0, "int": 0}
        for batch in train_loader:
            optim.zero_grad()
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            cat_l, pol_l, int_l = model(batch_dev["input_ids"], batch_dev["attention_mask"])
            loss, parts = multitask_loss(cat_l, pol_l, int_l, batch_dev, pos_w, task_w, args.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            ep_loss["total"] += loss.item()
            for k in ("cat","pol","int"): ep_loss[k] += parts[k]
        n_batch = len(train_loader)

        val_m = evaluate(model, val_loader, device)
        thr = tune_thresholds(val_m["cat_probs"], val_m["cat_labels"])
        val_tuned = evaluate(model, val_loader, device, thr)

        hist = {
            "epoch": ep + 1,
            "train_loss": {k: round(v / n_batch, 4) for k, v in ep_loss.items()},
            "val_cat_f1": val_tuned["cat_macro_f1"],
            "val_pol_f1": val_tuned["pol_macro_f1"],
            "val_int_f1": val_tuned["int_macro_f1"],
            "val_score_mae": val_tuned["score_mae"],
            "thresholds": thr,
            "epoch_sec": round(time.time() - t0, 1),
        }
        history.append(hist)
        print(f"\nEp {ep+1}/{args.epochs} | loss {hist['train_loss']} | "
              f"CatF1 {val_tuned['cat_macro_f1']:.4f} / PolF1 {val_tuned['pol_macro_f1']:.4f} / "
              f"IntF1 {val_tuned['int_macro_f1']:.4f} / MAE {val_tuned['score_mae']:.4f} | {hist['epoch_sec']}s")

        if val_tuned["cat_macro_f1"] > best_f1:
            best_f1, best_thr = val_tuned["cat_macro_f1"], thr
            patience_cnt = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print("  ✅ best 갱신")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  ⏹ Early stop (patience={args.patience})"); break

    # 최종 평가
    print("\n=== Test (best model + tuned thr) ===")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True))
    test_m = evaluate(model, test_loader, device, best_thr)
    print(f"Cat F1:   {test_m['cat_macro_f1']:.4f}")
    print(f"Pol F1:   {test_m['pol_macro_f1']:.4f}")
    print(f"Int F1:   {test_m['int_macro_f1']:.4f}")
    print(f"Score MAE: {test_m['score_mae']:.4f}")

    result = {
        "run_name": args.run_name,
        "config": cfg,
        "best_val_cat_f1": best_f1,
        "best_thresholds": dict(zip(CAT, best_thr)),
        "test_cat_macro_f1": test_m["cat_macro_f1"],
        "test_cat_per_class_f1": test_m["cat_per_class_f1"],
        "test_pol_macro_f1": test_m["pol_macro_f1"],
        "test_pol_per_class_f1": test_m["pol_per_class_f1"],
        "test_int_macro_f1": test_m["int_macro_f1"],
        "test_int_per_class_f1": test_m["int_per_class_f1"],
        "test_score_mae": test_m["score_mae"],
        "weights": WEIGHTS,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {run_dir}")


if __name__ == "__main__":
    main()
