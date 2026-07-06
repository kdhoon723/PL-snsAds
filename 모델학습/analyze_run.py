"""학습 결과 분석 + 다음 사이클 개선 방향 제안.

사용:
  python analyze_run.py <run_name>
  python analyze_run.py --compare run1 run2 run3
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "모델학습" / "runs"

CAT_COLS = ["사회적_정체성", "희소성", "긴급성", "사회적_증명", "가격비교", "권위_신뢰", "호혜성"]


def analyze_run(run_name: str):
    run = RUNS / run_name
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    print(f"=== {run_name} ===")
    print(f"Model: {result['config']['model_name']}")
    print(f"pos_weight_scale: {result['config']['pos_weight_scale']}")
    print(f"Test Macro F1: {result['test_macro_f1']:.4f}")
    print(f"Test Micro F1: {result['test_micro_f1']:.4f}")
    print(f"Test Intensity MAE: {result['test_intensity_mae']:.4f}\n")

    print(f"{'카테고리':<14} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Thr':>6}")
    print("-" * 45)
    for c in CAT_COLS:
        f1 = result["test_per_class_f1"][c]
        p = result["test_per_class_precision"][c]
        r = result["test_per_class_recall"][c]
        thr = result["best_thresholds"][c]
        print(f"{c:<14} {f1:>6.3f} {p:>6.3f} {r:>6.3f} {thr:>6.2f}")

    # 약한 카테고리
    weak = [c for c in CAT_COLS if result["test_per_class_f1"][c] < 0.5]
    print(f"\n약한 카테고리 (F1<0.5): {weak}")

    # 오류 분석
    pred_path = run / "test_predictions.csv"
    if pred_path.exists():
        analyze_predictions(pred_path)

    # 개선 제안
    print("\n=== 다음 사이클 개선 제안 ===")
    if result["test_macro_f1"] < 0.4:
        print("  - Macro F1 너무 낮음: pos_weight 강화 또는 더 긴 학습")
    if any(result["test_per_class_recall"][c] < 0.3 for c in weak):
        print("  - 약한 카테고리 recall 낮음: pos_weight_scale 1.5~2.0 시도")
    if result["test_intensity_mae"] > 0.15:
        print("  - 강도 점수 MAE 큼: threshold 튜닝 강화 또는 ensemble")
    if result["config"]["model_name"] == "klue/roberta-base":
        print("  - 더 큰 모델: klue/roberta-large 시도 검토")


def analyze_predictions(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"\n=== 오류 분석 (총 {len(rows)} 테스트 행) ===")
    fp_by_cat = Counter(); fn_by_cat = Counter()
    for r in rows:
        for c in CAT_COLS:
            t = int(r[f"true_{c}"]); p = int(r[f"pred_{c}"])
            if t == 0 and p == 1: fp_by_cat[c] += 1
            if t == 1 and p == 0: fn_by_cat[c] += 1
    print(f"{'카테고리':<14} {'FP':>5} {'FN':>5}")
    for c in CAT_COLS:
        print(f"{c:<14} {fp_by_cat[c]:>5} {fn_by_cat[c]:>5}")


def compare_runs(names):
    print(f"=== 사이클 비교 ===")
    print(f"{'run':<25} {'macro_F1':>10} {'micro_F1':>10} {'MAE':>8}")
    best = None
    for n in names:
        try:
            r = json.loads((RUNS / n / "result.json").read_text(encoding="utf-8"))
            print(f"{n:<25} {r['test_macro_f1']:>10.4f} {r['test_micro_f1']:>10.4f} {r['test_intensity_mae']:>8.4f}")
            if not best or r["test_macro_f1"] > best[1]:
                best = (n, r["test_macro_f1"])
        except FileNotFoundError:
            print(f"{n:<25} (없음)")
    if best:
        print(f"\n🏆 Best: {best[0]} (Macro F1 {best[1]:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", nargs="?")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    if args.compare:
        compare_runs(args.compare)
    elif args.run_name:
        analyze_run(args.run_name)
    else:
        # 모든 run 비교
        all_runs = sorted(p.name for p in RUNS.iterdir() if (p / "result.json").exists())
        if all_runs:
            compare_runs(all_runs)
        else:
            print("실행된 run 없음")


if __name__ == "__main__":
    main()
