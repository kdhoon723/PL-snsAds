"""
SNS 광고 텍스트 다중 라벨 분류 모델 학습

모델: klue/roberta-base (KoBERT보다 SNS 광고에 더 적합)
출력: 7개 카테고리 시그모이드 확률
손실: BCEWithLogitsLoss + pos_weight (클래스 불균형 처리)

실행 전 설치 필요:
  pip install torch transformers scikit-learn pandas

실행:
  python train_kobert.py
"""
import csv
import json
from pathlib import Path

# 학습 라이브러리 (필요시 설치)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score, precision_score, recall_score
    import numpy as np
except ImportError as e:
    print(f"❌ 필수 라이브러리 누락: {e}")
    print("설치: pip install torch transformers scikit-learn numpy")
    raise SystemExit(1)

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "문장라벨링"
TRAIN_CSV = ROOT / "split_train.csv"
VAL_CSV = ROOT / "split_val.csv"
TEST_CSV = ROOT / "split_test.csv"

CAT_COLS = ["사회적_정체성", "희소성", "긴급성", "사회적_증명", "가격비교", "권위_신뢰", "호혜성"]
NUM_LABELS = len(CAT_COLS)

# PDF 가중치 (강도 점수 계산용)
WEIGHTS = {
    "권위_신뢰": 0.221,
    "사회적_증명": 0.159,
    "사회적_정체성": 0.156,
    "호혜성": 0.151,
    "희소성": 0.126,
    "긴급성": 0.096,
    "가격비교": 0.091,
}

# 하이퍼파라미터
CONFIG = {
    "model_name": "klue/roberta-base",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "epochs": 10,
    "dropout": 0.3,
    "patience": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Device: {CONFIG['device']}")
print(f"Model: {CONFIG['model_name']}")


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                labels = [int(row[c]) for c in CAT_COLS]
            except (KeyError, ValueError):
                continue
            rows.append({"text": row["text"], "labels": labels})
    return rows


class AdsDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["labels"], dtype=torch.float),
        }


class AdsClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        return self.classifier(self.dropout(pooled))


def compute_pos_weights(train_data, num_labels):
    """클래스 불균형 처리용 양성 가중치"""
    counts = [0] * num_labels
    n = len(train_data)
    for r in train_data:
        for i, l in enumerate(r["labels"]):
            counts[i] += l
    # pos_weight = (n - pos) / pos  (sklearn-like)
    weights = []
    for c in counts:
        if c == 0:
            weights.append(1.0)
        else:
            weights.append((n - c) / max(c, 1))
    return torch.tensor(weights, dtype=torch.float)


def evaluate(model, loader, device, thresholds=None):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"])

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()

    if thresholds is None:
        thresholds = [0.5] * NUM_LABELS

    preds = (probs >= np.array(thresholds)).astype(int)

    per_class_f1 = []
    for i, cat in enumerate(CAT_COLS):
        if labels_np[:, i].sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(labels_np[:, i], preds[:, i], zero_division=0)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))
    micro_f1 = f1_score(labels_np, preds, average="micro", zero_division=0)

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_f1": dict(zip(CAT_COLS, per_class_f1)),
        "probs": probs,
        "labels": labels_np,
    }


def tune_thresholds(probs, labels):
    """검증셋에서 카테고리별 최적 임계값 탐색"""
    thresholds = []
    for i in range(NUM_LABELS):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.2, 0.71, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            if labels[:, i].sum() == 0:
                continue
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(float(best_t))
    return thresholds


def intensity_score(probs):
    """7개 카테고리 확률 → 가중 강도 점수"""
    s = 0.0
    for i, c in enumerate(CAT_COLS):
        s += WEIGHTS[c] * float(probs[i])
    return s


def main():
    print("\n=== 데이터 로드 ===")
    train_data = load_csv(TRAIN_CSV)
    val_data = load_csv(VAL_CSV)
    test_data = load_csv(TEST_CSV)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) == 0:
        print("❌ Train 데이터가 없습니다. merge_datasets.py를 먼저 실행하세요.")
        return

    # 토크나이저 / 모델
    print("\n=== 모델 초기화 ===")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AdsClassifier(CONFIG["model_name"], NUM_LABELS, CONFIG["dropout"])
    model.to(CONFIG["device"])

    # 데이터로더
    train_ds = AdsDataset(train_data, tokenizer, CONFIG["max_length"])
    val_ds = AdsDataset(val_data, tokenizer, CONFIG["max_length"])
    test_ds = AdsDataset(test_data, tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

    # 손실 함수 (클래스 불균형 처리)
    pos_weights = compute_pos_weights(train_data, NUM_LABELS).to(CONFIG["device"])
    print(f"Pos weights: {pos_weights.cpu().tolist()}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 옵티마이저 + 스케줄러
    optimizer = AdamW(model.parameters(),
                      lr=CONFIG["learning_rate"],
                      weight_decay=CONFIG["weight_decay"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 학습 루프
    print("\n=== 학습 시작 ===")
    best_val_f1 = 0.0
    best_thresholds = None
    patience_counter = 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(CONFIG["device"])
            attention_mask = batch["attention_mask"].to(CONFIG["device"])
            labels = batch["labels"].to(CONFIG["device"])

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, CONFIG["device"])
        thresholds = tune_thresholds(val_metrics["probs"], val_metrics["labels"])
        val_metrics_tuned = evaluate(model, val_loader, CONFIG["device"], thresholds)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_macro_f1": val_metrics_tuned["macro_f1"],
            "val_per_class_f1": val_metrics_tuned["per_class_f1"],
        })

        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f}")
        print(f"  Val Macro F1: {val_metrics_tuned['macro_f1']:.4f} "
              f"(default thr) {val_metrics['macro_f1']:.4f}")
        for c, f1 in val_metrics_tuned["per_class_f1"].items():
            print(f"    {c}: {f1:.3f}")

        # Early stopping
        if val_metrics_tuned["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics_tuned["macro_f1"]
            best_thresholds = thresholds
            patience_counter = 0
            torch.save(model.state_dict(), ROOT.parent / "모델학습" / "best_model.pt")
            print("  ✅ best 갱신")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"  ⏹️  Early stopping (patience={CONFIG['patience']})")
                break

    # 최종 평가 (best 모델 + 최적 임계값)
    print(f"\n=== 최종 평가 (Test) ===")
    model.load_state_dict(torch.load(ROOT.parent / "모델학습" / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, CONFIG["device"], best_thresholds)
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Micro F1: {test_metrics['micro_f1']:.4f}")
    for c, f1 in test_metrics["per_class_f1"].items():
        print(f"  {c}: {f1:.3f}")

    # 결과 저장
    result = {
        "config": CONFIG,
        "thresholds": dict(zip(CAT_COLS, best_thresholds)),
        "weights": WEIGHTS,
        "best_val_f1": best_val_f1,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "training_history": history,
    }
    # device 객체는 직렬화 불가 → 문자열로
    result["config"] = {**CONFIG, "device": str(CONFIG["device"])}
    (ROOT.parent / "모델학습" / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n💾 학습 결과 저장: 모델학습/training_result.json")
    print(f"💾 모델 가중치: 모델학습/best_model.pt")

    # 강도 점수 예시
    print(f"\n=== 강도 점수 예시 (Test 첫 5건) ===")
    probs = test_metrics["probs"][:5]
    for i, p in enumerate(probs):
        s = intensity_score(p)
        text = test_data[i]["text"][:50]
        print(f"  [{s:.3f}] {text}")


if __name__ == "__main__":
    main()
