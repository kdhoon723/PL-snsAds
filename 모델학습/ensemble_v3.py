"""Multitask 앙상블 (Cycle 41 + 48 + 39) — Test 평가.

각 모델의 cat_probs/pol_probs/int_probs를 평균.
"""
import json, csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
import numpy as np

import sys
sys.path.insert(0, "PROJECT_ROOT/모델학습")
from train_v3_multitask import (
    load_csv, AdsDataset, MultiTaskClassifier, evaluate, tune_thresholds,
    CAT, POLARITY, INTENSITY, POL_IDX, INT_IDX, NUM_CAT, WEIGHTS,
    POLARITY_SCORE, INTENSITY_SCORE, synergy, calibrate,
)

PROJECT = Path("PROJECT_ROOT")
DATA = PROJECT / "통합데이터셋"
RUNS = PROJECT / "모델학습" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
# Cycle 50: 4 모델 앙상블 (46 dropout 0.2 추가 — 다양성)
# Cycle 49: 3 모델 앙상블 baseline
RUN_NAMES = os.environ.get("ENS_RUNS", "cycle41_taskweight_cat,cycle48_seed7,cycle39_taskweight_balanced,cycle46_dropout0.2").split(",")
ENS_NAME = os.environ.get("ENS_NAME", "cycle50_ensemble_41_48_39_46")
OUT = RUNS / ENS_NAME
OUT.mkdir(parents=True, exist_ok=True)


def load_model(run_name):
    run_dir = RUNS / run_name
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = MultiTaskClassifier(cfg["model"], cfg["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True, map_location=DEVICE))
    model.eval()
    return tok, model, cfg


@torch.no_grad()
def predict_all(tok, model, rows, max_len=256, batch_size=8):
    """Test 셋 전체에 대한 cat_probs, pol_probs, int_probs."""
    ds = AdsDataset(rows, tok, max_len)
    loader = DataLoader(ds, batch_size=batch_size)
    all_cat_probs, all_pol_probs, all_int_probs = [], [], []
    all_cat_labels, all_pol_labels, all_int_labels = [], [], []
    for batch in loader:
        cat_l, pol_l, int_l = model(
            batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
        )
        all_cat_probs.append(torch.sigmoid(cat_l).cpu().numpy())
        all_pol_probs.append(F.softmax(pol_l, dim=-1).cpu().numpy())
        all_int_probs.append(F.softmax(int_l, dim=-1).cpu().numpy())
        all_cat_labels.append(batch["category"].numpy())
        all_pol_labels.append(batch["polarity"].numpy())
        all_int_labels.append(batch["intensity"].numpy())
    return (
        np.concatenate(all_cat_probs, axis=0),
        np.concatenate(all_pol_probs, axis=0),
        np.concatenate(all_int_probs, axis=0),
        np.concatenate(all_cat_labels, axis=0),
        np.concatenate(all_pol_labels, axis=0),
        np.concatenate(all_int_labels, axis=0),
    )


def main():
    test_rows = load_csv(DATA / "split_test_v3.csv")
    val_rows = load_csv(DATA / "split_val_v3.csv")
    print(f"Test: {len(test_rows)} / Val: {len(val_rows)}")

    print(f"\n=== 모델 {len(RUN_NAMES)}개 로드 + Test 예측 ===")
    cat_probs_list, pol_probs_list, int_probs_list = [], [], []
    val_cat_probs_list, val_pol_probs_list, val_int_probs_list = [], [], []
    cat_labels = pol_labels = int_labels = None
    val_cat_labels = val_pol_labels = val_int_labels = None

    for name in RUN_NAMES:
        print(f"  → {name}")
        tok, model, cfg = load_model(name)
        max_len = cfg["max_length"]
        # Test
        cp, pp, ip, cl, pl, il = predict_all(tok, model, test_rows, max_len, cfg["batch"])
        cat_probs_list.append(cp); pol_probs_list.append(pp); int_probs_list.append(ip)
        if cat_labels is None:
            cat_labels, pol_labels, int_labels = cl, pl, il
        # Val (for threshold tuning)
        vcp, vpp, vip, vcl, vpl, vil = predict_all(tok, model, val_rows, max_len, cfg["batch"])
        val_cat_probs_list.append(vcp); val_pol_probs_list.append(vpp); val_int_probs_list.append(vip)
        if val_cat_labels is None:
            val_cat_labels, val_pol_labels, val_int_labels = vcl, vpl, vil
        del model; torch.cuda.empty_cache()

    # 앙상블 (평균)
    cat_probs = np.mean(cat_probs_list, axis=0)
    pol_probs = np.mean(pol_probs_list, axis=0)
    int_probs = np.mean(int_probs_list, axis=0)
    val_cat_probs = np.mean(val_cat_probs_list, axis=0)
    val_pol_probs = np.mean(val_pol_probs_list, axis=0)
    val_int_probs = np.mean(val_int_probs_list, axis=0)

    # threshold 튜닝 (val에서)
    thresholds = tune_thresholds(val_cat_probs, val_cat_labels)
    print(f"\n앙상블 thresholds: {[round(t, 2) for t in thresholds]}")

    # Test 평가
    cat_preds = (cat_probs >= np.array(thresholds)).astype(int)
    pol_preds = pol_probs.argmax(-1)  # (N, 7)
    int_preds = int_probs.argmax(-1)

    # Cat F1
    per_class_f1 = []
    for i, c in enumerate(CAT):
        if cat_labels[:, i].sum() == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(f1_score(cat_labels[:, i], cat_preds[:, i], zero_division=0))
    cat_macro_f1 = float(np.mean(per_class_f1))

    # Pol/Int F1 (양성만)
    pol_per_cat, int_per_cat = {}, {}
    for i, c in enumerate(CAT):
        mask = pol_labels[:, i] != -100
        if mask.sum() == 0:
            pol_per_cat[c] = int_per_cat[c] = 0.0
            continue
        pol_per_cat[c] = f1_score(pol_labels[mask, i], pol_preds[mask, i], average="macro", zero_division=0)
        int_per_cat[c] = f1_score(int_labels[mask, i], int_preds[mask, i], average="macro", zero_division=0)
    pol_macro_f1 = float(np.mean(list(pol_per_cat.values())))
    int_macro_f1 = float(np.mean(list(int_per_cat.values())))

    # Score MAE
    n = len(cat_labels)
    pred_scores, true_scores = [], []
    for j in range(n):
        # true
        true_n = int(cat_labels[j].sum())
        ts = sum(WEIGHTS[c] * 1.0 * POLARITY_SCORE[POLARITY[pol_labels[j, i]] if pol_labels[j, i] != -100 else 0] * INTENSITY_SCORE[INTENSITY[int_labels[j, i]] if int_labels[j, i] != -100 else 1]
                 for i, c in enumerate(CAT) if cat_labels[j, i] == 1)
        true_scores.append(ts * synergy(true_n))
        # pred
        pred_n = int(cat_preds[j].sum())
        ps = sum(WEIGHTS[c] * calibrate(c, float(cat_probs[j, i])) * POLARITY_SCORE[POLARITY[pol_preds[j, i]]] * INTENSITY_SCORE[INTENSITY[int_preds[j, i]]]
                 for i, c in enumerate(CAT) if cat_preds[j, i] == 1)
        pred_scores.append(ps * synergy(pred_n))
    score_mae = float(np.mean(np.abs(np.array(pred_scores) - np.array(true_scores))))

    print(f"\n=== Ensemble Test 결과 ===")
    print(f"Cat F1:    {cat_macro_f1:.4f}")
    print(f"Pol F1:    {pol_macro_f1:.4f}")
    print(f"Int F1:    {int_macro_f1:.4f}")
    print(f"Score MAE: {score_mae:.4f}")
    print(f"\n카테고리별 F1:")
    for c, f in zip(CAT, per_class_f1):
        print(f"  {c}: {f:.4f}")

    # 저장
    result = {
        "run_name": ENS_NAME,
        "ensemble_members": RUN_NAMES,
        "test_cat_macro_f1": cat_macro_f1,
        "test_cat_per_class_f1": dict(zip(CAT, per_class_f1)),
        "test_pol_macro_f1": pol_macro_f1,
        "test_pol_per_class_f1": pol_per_cat,
        "test_int_macro_f1": int_macro_f1,
        "test_int_per_class_f1": int_per_cat,
        "test_score_mae": score_mae,
        "thresholds": dict(zip(CAT, thresholds)),
    }
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"ensemble_members": RUN_NAMES}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {OUT}")


if __name__ == "__main__":
    main()
