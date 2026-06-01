"""SNS 광고 소비심리 자극 점수 계산 v2 (PDF + 팀원 A 융합)

공식:
  score = Σ(w_i × ĉ_i × p_i × t_i) × synergy(n) × 100

여기서:
  - w_i: 학술 가중치 (메타분석 r값 정규화)
  - ĉ_i: baseline 보정된 신뢰도 (calibrated confidence)
  - p_i: 방향성 점수 (+1.0 긍정 / +0.5 중립 / -1.0 부정)
  - t_i: 강도 점수 (1.5 강 / 1.0 보통 / 0.5 약)
  - n: 양성 카테고리 수
  - synergy(n): 4단계 Inverted-U (1.0 / 1.15 / 1.25 / 0.90)

학술 근거:
  - 가중치 w: Pornpitakpan(2004), Ao et al.(2023), Cialdini(1984) 등 메타분석
  - synergy: Berlyne(1970) inverted-U + Cialdini synergy + Brehm reactance + Friestad & Wright(1994)
  - calibration: PDF (가중치 계산.pdf) 설계자 baseline

한계:
  - Framework dependence: Singla et al.(2023) 7-category 한정 ad-hoc 보정
  - Calibration baseline은 임의 값 — 데이터 fit 없음
  - Inverted-U 5단계 → 4단계 단순화 (학술 정직성)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 1. 학술 가중치 (메타분석 r값 → 정규화)
# ============================================================
R_VALUES = {
    "권위_신뢰":     0.44,   # Pornpitakpan (2004)
    "사회적_증명":   0.317,  # Qiu & Zhang (2024) — review volume r
    "사회적_정체성": 0.31,   # Aguirre-Rodriguez et al. (2012)
    "호혜성":        0.30,   # Cialdini (1984) + Raghubir (2004) 추정
    "희소성":        0.25,   # Khalid et al. (2025) β→r 변환
    "긴급성":        0.19,   # Ladeira et al. (2023)
    "가격비교":      0.18,   # 이상수 (2023) β→r 변환
}
_SIGMA_R = sum(R_VALUES.values())
WEIGHTS = {cat: r / _SIGMA_R for cat, r in R_VALUES.items()}


# ============================================================
# 2. 방향성·강도 점수
# ============================================================
POLARITY_SCORE = {
    "긍정": 1.0,
    "중립": 0.5,
    "부정": -1.0,
}
INTENSITY_SCORE = {
    "강": 1.5,
    "보통": 1.0,
    "약": 0.5,
}


# ============================================================
# 3. 신뢰도 보정 baseline (PDF 설계 — 임의 값)
#    ⚠️ DEPRECATED: ablation 결과 과보정으로 오차 악화 (MAE 6.83→5.80).
#    점수 계산(calculate_ad_score)에서 더 이상 사용하지 않음. 참고용으로만 유지.
# ============================================================
CONFIDENCE_BASELINE = {
    "권위_신뢰":     0.90,
    "사회적_증명":   0.85,
    "희소성":        0.85,
    "긴급성":        0.85,
    "가격비교":      0.80,
    "호혜성":        0.75,
    "사회적_정체성": 0.65,
}
MAX_CALIBRATION = 2.0


def calibrate_confidence(category: str, raw_confidence: float) -> float:
    """카테고리별 baseline으로 신뢰도 보정.

    약한 카테고리(정체성)는 confidence 낮게 나오기 쉬우니 baseline까지 끌어올림.
    공식: min(raw_confidence × min(baseline/raw_confidence, 2.0), 1.0)
    """
    if category not in CONFIDENCE_BASELINE:
        return raw_confidence
    baseline = CONFIDENCE_BASELINE[category]
    if raw_confidence == 0:
        return 0.0
    calibration = min(baseline / raw_confidence, MAX_CALIBRATION)
    return min(raw_confidence * calibration, 1.0)


# ============================================================
# 4. Inverted-U 시너지 (4단계, framework-dependent ad-hoc 보정)
# ============================================================
def synergy_factor(n_categories: int) -> float:
    """n개 카테고리 동시 활성화에 따른 시너지 보정.

    근거:
      - n=2~3: Cialdini(1984) multi-cue synergy + Maheswaran & Chaiken(1991)
      - n≥4: Brehm(1966) reactance + Friestad & Wright(1994) persuasion knowledge
      - 모양: Berlyne(1970) inverted-U

    한계: framework-dependent. Singla 7-category 한정.
    """
    if n_categories <= 1:
        return 1.00
    elif n_categories == 2:
        return 1.15
    elif n_categories == 3:
        return 1.25
    else:  # n >= 4
        return 0.90


# ============================================================
# 5. 점수 계산
# ============================================================
@dataclass
class CategoryResult:
    """카테고리별 분류 결과."""
    category: str
    probability: float       # 모델 시그모이드 확률
    is_positive: bool        # threshold 통과 여부
    polarity: Optional[str] = None     # "긍정" / "중립" / "부정"
    intensity: Optional[str] = None    # "강" / "보통" / "약"
    calibrated_confidence: float = field(init=False, default=0.0)
    score: float = field(init=False, default=0.0)


def calculate_ad_score(
    category_results: list[CategoryResult],
    use_synergy: bool = True,
) -> dict:
    """광고 단위 최종 점수.

    신뢰도 보정(ĉ)은 제거됨 — ablation 결과 과보정으로 오차 악화 (MAE 6.83→5.80).
    모델 원본 확률(prob)을 그대로 사용.

    Args:
      use_synergy: Inverted-U 시너지 보정 사용 (기본 True). False는 MAE +0.48 악화.

    Pipeline:
      1. 양성 카테고리만 추출
      2. 점수 계산: w × prob × p × t  (신뢰도 보정 없음)
      3. (옵션) 시너지 보정 적용 (use_synergy=True)
      4. 100점 만점 정규화

    Returns:
      {
        "final_score_100": float,    # 0~100점 (음수 가능 — 부정 방향)
        "raw_score": float,
        "n_positive_categories": int,
        "synergy_factor": float,
        "options": {"use_synergy": bool},
        "per_category": [{...}]
      }
    """
    positives = [c for c in category_results if c.is_positive]
    n_pos = len(positives)
    synergy = synergy_factor(n_pos) if use_synergy else 1.0

    per_cat = []
    raw_sum = 0.0
    for c in positives:
        prob = c.probability
        c.calibrated_confidence = prob  # 호환용: 보정 없이 원본 확률
        w = WEIGHTS.get(c.category, 0.0)
        p = POLARITY_SCORE.get(c.polarity or "긍정", 0.5)
        t = INTENSITY_SCORE.get(c.intensity or "보통", 1.0)
        c.score = w * prob * p * t
        raw_sum += c.score
        per_cat.append({
            "category": c.category,
            "weight": round(w, 4),
            "probability": round(prob, 4),
            "polarity": c.polarity,
            "intensity": c.intensity,
            "score": round(c.score, 4),
        })

    raw_score = raw_sum * synergy
    final_100 = raw_score * 100

    syn_note = "시너지 ON" if use_synergy else "시너지 OFF"

    return {
        "final_score_100": round(final_100, 2),
        "raw_score": round(raw_score, 4),
        "n_positive_categories": n_pos,
        "synergy_factor": synergy,
        "options": {"use_synergy": use_synergy},
        "per_category": per_cat,
        "note": f"점수 공식 (신뢰도 보정 제거). {n_pos}개 양성. {syn_note}",
    }


# ============================================================
# 6. 단순 호환 함수 (기존 v1과 비교용)
# ============================================================
def simple_score_v1(category_probs: dict[str, float]) -> float:
    """v1 단순 점수 (방향성·강도·시너지 없음). 기존 시스템과 동일.

    intensity_score = Σ(w_i × prob_i)
    """
    return sum(WEIGHTS.get(c, 0) * p for c, p in category_probs.items())


# ============================================================
# 실행 예시
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📊 카테고리별 가중치 (Σr = {:.3f})".format(_SIGMA_R))
    print("=" * 60)
    for cat in sorted(WEIGHTS, key=lambda c: -WEIGHTS[c]):
        b = CONFIDENCE_BASELINE.get(cat, "-")
        print(f"  {cat:<14} w={WEIGHTS[cat]:.3f}  baseline={b}")

    print("\n" + "=" * 60)
    print("📐 Inverted-U 시너지 계수")
    print("=" * 60)
    for n in range(1, 6):
        label = f"n={n}" if n < 4 else f"n≥{n}"
        print(f"  {label} → ×{synergy_factor(n):.2f}")

    print("\n" + "=" * 60)
    print("🧪 테스트: '서울대 치과 전문의 추천 오늘만 70% 할인 100명 한정'")
    print("=" * 60)
    test = [
        CategoryResult("권위_신뢰", 0.95, True, "긍정", "강"),
        CategoryResult("희소성", 0.92, True, "긍정", "강"),
        CategoryResult("긴급성", 0.88, True, "긍정", "강"),
        CategoryResult("가격비교", 0.94, True, "긍정", "강"),
    ]
    result = calculate_ad_score(test)
    print(f"양성 카테고리: {result['n_positive_categories']} → 시너지 ×{result['synergy_factor']}")
    print(f"\n카테고리별 점수:")
    for c in result["per_category"]:
        print(f"  {c['category']:<14} w={c['weight']:.3f} × prob={c['probability']:.3f} "
              f"× p({c['polarity']})={POLARITY_SCORE[c['polarity']]:+.1f} "
              f"× t({c['intensity']})={INTENSITY_SCORE[c['intensity']]:.1f} = {c['score']:+.4f}")
    print(f"\n💯 최종 점수: {result['final_score_100']}/100")

    print("\n" + "=" * 60)
    print("🧪 테스트: 부정 케이스 (공익광고 — 정체성 부정)")
    print("=" * 60)
    test_neg = [
        CategoryResult("사회적_정체성", 0.80, True, "부정", "강"),
    ]
    result_neg = calculate_ad_score(test_neg)
    print(f"양성 카테고리: {result_neg['n_positive_categories']} → 시너지 ×{result_neg['synergy_factor']}")
    print(f"최종 점수: {result_neg['final_score_100']}/100 (음수 — 행동 억제 방향)")
