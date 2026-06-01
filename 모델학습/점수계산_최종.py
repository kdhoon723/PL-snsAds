"""
SNS 광고 소비심리 자극 측정 — 점수 계산 (최종 확정 버전)

실험 결과 반영:
- 신뢰도 보정 제거 (오차 6.83 → 5.80, 과보정 문제)
- 시너지: 1개×1.0 / 2개×1.15 / 3개×1.25 / 4개↑×0.90
- 강도(t) 유지 (제거 시 가장 부정확, 오차 8.12)
- 방향성(p) 유지 (공익광고·부정 맥락 대응)

최종 계산식 (광고 단위):
    Score = Σ_i (w_i × prob_i × p_i × t_i) × synergy(n) × 100
      i : 그 광고에서 탐지된(양성) 카테고리
      n : 양성 카테고리 수

────────────────────────────────────────────────────────────
⚠️ 팀 원본 설계서(PDF)와 다른 점
  1. 분류 방식: 원본=단일 라벨(문장당 1 카테고리). 우리=멀티 라벨(한 광고에서
     여러 카테고리 동시 탐지). 실험상 멀티 라벨 F1이 더 높아 채택(0.830).
  2. 채점 단위: 원본=문장 분리 후 ÷n_sentences. 우리=광고 통째 1단위(÷n 없음).
  3. 신뢰도 보정(ĉ): 원본=있음. 우리=제거(ablation 결과 과보정).
  4. 시너지: 원본=단조증가(1.30/1.45). 우리=Inverted-U(1.25/0.90, '과다 시
     역효과' 의견 반영).
  ※ 단일 라벨이면 문장당 카테고리=1 → synergy(n)이 항상 1로 작동 불가.
    멀티 라벨이라 시너지가 자연스럽게 작동함.

실제 모델 추론·학습 코드:
  - 모델 학습:  train_v3_multitask.py  (멀티태스크: 카테고리 sigmoid + 방향성·강도 softmax)
  - 추론·점수:  predict_v3.py          (Cycle 50 4-모델 앙상블 + 본 점수식)
  - 서버:       server.py              (FastAPI, localhost:7860)
  - 점수 모듈:  scoring_v2.py          (본 파일과 동일 공식의 운영본)
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# 1. 가중치 설정 (선행 메타분석 r값 기반)
#    w_i = r_i / Σr   (정규화 → 합 1.0)
# ============================================================
R_VALUES = {
    "권위_신뢰":     0.44,   # Pornpitakpan (2004)
    "사회적_증명":   0.317,  # Qiu & Zhang (2024) — review volume r
    "사회적_정체성": 0.31,   # Aguirre-Rodriguez et al. (2012)
    "호혜성":        0.30,   # Cialdini(1984) + Raghubir(2004) 추정
    "희소성":        0.25,   # Khalid et al. (2025) β→r 변환
    "긴급성":        0.19,   # Ladeira et al. (2023)
    "가격비교":      0.18,   # 이상수 (2023) β→r 변환
}
SIGMA_R = sum(R_VALUES.values())                          # 1.987
WEIGHTS = {cat: r / SIGMA_R for cat, r in R_VALUES.items()}


# ============================================================
# 2. 맥락 설정 (방향성 p × 강도 t)
# ============================================================

# 방향성 (p) — 광고 대부분 긍정이라 점수 영향은 작지만,
#              공익광고·부정 맥락 대응 위해 유지 (제거해도 오차 동일)
POLARITY = {
    "긍정":  1.0,   # 구매 권유
    "중립":  0.5,   # 정보 안내
    "부정": -1.0,   # 억제 (공익광고 등)
}

# 강도 (t) — 실험에서 가장 중요한 항 (제거 시 오차 8.12로 가장 부정확)
INTENSITY = {
    "강":   1.5,   # 구체적·극단적 (70% 할인, 100명 한정)
    "보통": 1.0,   # 일반적 표현
    "약":   0.5,   # 추상적·약한 표현
}


# ============================================================
# 3. 시너지 보정 (Inverted-U)
#    2~3개 동시: 상승 / 4개 이상: 과다 자극 → 거부감 → 감점
#    실험: 시너지 제거 시 오차 7.31 (유지가 맞음)
# ============================================================
def synergy_factor(n: int) -> float:
    """동시 탐지 카테고리 수 n에 따른 보정 계수.

    n=1 → ×1.00 (보정 없음)
    n=2 → ×1.15 (상승)
    n=3 → ×1.25 (정점)
    n≥4 → ×0.90 (거부감, 감점)
    """
    if n <= 1:
        return 1.00
    elif n == 2:
        return 1.15
    elif n == 3:
        return 1.25
    else:
        return 0.90


# ============================================================
# 4. 점수 계산 (광고 단위, 멀티 라벨)
#
# 문장당 1 카테고리(단일 라벨)가 아니라, 한 광고에서 여러 카테고리를
# 동시에 탐지(멀티 라벨)하므로 광고를 1단위로 채점한다.
#
# 카테고리 점수 = w_i × prob_i × p_i × t_i   (신뢰도 보정 없음)
# 최종 점수     = Σ_i(카테고리 점수) × synergy(n) × 100
# ============================================================

@dataclass
class CategoryResult:
    """한 광고의 카테고리별 분류 결과 (멀티태스크 모델 출력)."""
    category:   str            # 카테고리명
    probability: float         # 모델 시그모이드 확률 (보정 없이 그대로)
    is_positive: bool          # threshold 통과 여부 (양성)
    polarity:   Optional[str] = None   # 긍정/중립/부정 (양성일 때만)
    intensity:  Optional[str] = None   # 강/보통/약   (양성일 때만)
    score:      float = field(init=False, default=0.0)


def calculate_ad_score(category_results: List[CategoryResult]) -> dict:
    """광고 단위 최종 소비심리 자극 점수.

    실험 결과 반영:
        - 신뢰도 보정 제거 (오차 6.83 → 5.80)
        - 시너지 유지 (제거 시 7.31)
        - 강도 유지 (제거 시 8.12, 가장 중요)
        - 방향성 유지 (공익광고 대응)
    """
    positives = [c for c in category_results if c.is_positive]
    n = len(positives)
    synergy = synergy_factor(n)

    per_category = []
    raw_sum = 0.0
    for c in positives:
        w   = WEIGHTS.get(c.category, 0.0)
        p   = POLARITY.get(c.polarity or "긍정", 0.5)
        t   = INTENSITY.get(c.intensity or "보통", 1.0)
        c.score = w * c.probability * p * t        # 신뢰도 보정 없음
        raw_sum += c.score
        per_category.append({
            "카테고리":  c.category,
            "가중치":    round(w, 4),
            "확률":      round(c.probability, 4),
            "방향성":    c.polarity,
            "강도":      c.intensity,
            "점수":      round(c.score, 4),
        })

    raw_score   = raw_sum * synergy
    final_score = raw_score * 100   # 0~100점 (부정 방향이면 음수 가능)

    return {
        "final_score":     round(final_score, 2),
        "raw_score":       round(raw_score, 4),
        "n_categories":    n,
        "synergy_factor":  synergy,
        "per_category":    per_category,
    }


# ============================================================
# 5. 실행 예시
# ============================================================
if __name__ == "__main__":

    print("=" * 64)
    print(f"카테고리별 가중치  (Σr = {SIGMA_R:.3f})")
    print("=" * 64)
    for cat, r in sorted(R_VALUES.items(), key=lambda x: -x[1]):
        print(f"  {cat:<14} r={r:.3f}  →  가중치 {WEIGHTS[cat]*100:5.1f}%")

    print("\n" + "=" * 64)
    print("시너지 보정 계수 (Inverted-U, 최종 확정)")
    print("=" * 64)
    labels = {1: "단일", 2: "2개 동시 (상승)", 3: "3개 동시 (정점)", 4: "4개 이상 (거부감)"}
    for n in range(1, 5):
        print(f"  n={n}  {labels[n]:<18} → ×{synergy_factor(n):.2f}")

    print("\n" + "=" * 64)
    print("점수 공식 ablation 실험 결과 (테스트 922건, 정답과의 평균 오차)")
    print("=" * 64)
    table = [
        ("전체 (PDF 원본)",  6.83, "기준"),
        ("신뢰도 보정 제거", 5.80, "✅ 채택 — 1.03점 더 정확 (지금 공식)"),
        ("강도(t) 제거",     8.12, "❌ 가장 부정확 (가장 중요한 항)"),
        ("시너지 제거",      7.31, "❌ 유지가 맞음"),
        ("방향성(p) 제거",   6.83, "= 변화 없음 (광고 99% 긍정)"),
    ]
    print(f"  {'공식':<18} {'오차':>5}  해석")
    print(f"  {'-'*58}")
    for name, err, note in table:
        print(f"  {name:<18} {err:>5.2f}  {note}")

    print("\n" + "=" * 64)
    print("계산 예시:  '서울대 치과 전문의 추천! 오늘만 70% 할인, 100명 한정'")
    print("=" * 64)
    example = [
        CategoryResult("권위_신뢰", 0.95, True, "긍정", "강"),
        CategoryResult("긴급성",   0.92, True, "긍정", "강"),
        CategoryResult("가격비교", 0.94, True, "긍정", "강"),
        CategoryResult("희소성",   0.90, True, "긍정", "강"),
        CategoryResult("사회적_증명",  0.10, False),  # 양성 아님 (점수 미반영)
    ]
    result = calculate_ad_score(example)

    print(f"\n  {'카테고리':<12} {'가중치':>6} {'확률':>6} {'방향':>4} {'강도':>4} {'점수':>9}")
    print(f"  {'-'*52}")
    for c in result["per_category"]:
        print(f"  {c['카테고리']:<12} {c['가중치']:>6.3f} {c['확률']:>6.2f} "
              f"{c['방향성']:>4} {c['강도']:>4} {c['점수']:>+9.4f}")
    print(f"\n  양성 카테고리 {result['n_categories']}개 → 시너지 ×{result['synergy_factor']}")
    print(f"  최종 점수: {result['final_score']} / 100점")

    print("\n" + "=" * 64)
    print("부정 케이스 예시:  공익광고 '당신은 정말 비흡연자입니까?'")
    print("=" * 64)
    public = [CategoryResult("사회적_정체성", 0.80, True, "부정", "강")]
    r2 = calculate_ad_score(public)
    print(f"  사회적_정체성 (부정·강) → 최종 점수 {r2['final_score']} / 100점 "
          f"(음수 = 행동 억제 방향)")
