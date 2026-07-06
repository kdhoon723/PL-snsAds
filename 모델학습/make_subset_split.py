"""기존 split에서 특정 조건의 부분집합만 추출.

사용:
  python make_subset_split.py --filter human-only --out-dir 통합데이터셋_human
  python make_subset_split.py --filter ad-only   --out-dir 통합데이터셋_ad
  python make_subset_split.py --filter sentence-only --out-dir 통합데이터셋_sentence
  python make_subset_split.py --filter claude-only --out-dir 통합데이터셋_claude
"""
import argparse, csv, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "통합데이터셋"
CAT = ["사회적_정체성","희소성","긴급성","사회적_증명","가격비교","권위_신뢰","호혜성"]


def filter_row(r, mode):
    if mode == "human-only":
        return "HUMAN" in r["label_source"]
    if mode == "claude-only":
        return r["label_source"] == "CLAUDE"
    if mode == "ad-only":
        return r["unit"] == "ad"
    if mode == "sentence-only":
        return r["unit"] == "sentence"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True,
                    choices=["human-only","claude-only","ad-only","sentence-only"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = PROJECT / args.out_dir
    out.mkdir(exist_ok=True)

    stats = {}
    for split in ["split_train","split_val","split_test"]:
        with open(SRC / f"{split}.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if filter_row(r, args.filter)]
        fields = list(rows[0].keys())
        with open(out / f"{split}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(kept)
        stats[split] = len(kept)
        # 카테고리 분포
        cats = {c: sum(1 for r in kept if r[c] == "1") for c in CAT}
        print(f"{split}: {len(rows)} → {len(kept)} | {cats}")

    (out / "filter_meta.json").write_text(json.dumps({
        "filter": args.filter, "splits": stats
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {out}/")


if __name__ == "__main__":
    main()
