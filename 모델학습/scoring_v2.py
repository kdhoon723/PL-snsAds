"""SNS 광고 소비심리 자극 점수 계산 v2 (PDF + 팀원 A 융합)

공식:
  score = Σ(w_i × ĉ_i × p_i × t_i) × synergy(n) × 100

여기서:
  - w_i: a priori 가중치 (이론·메타분석 정보 기반 사전 설정; 실측 r 직접 투입 아님)
  - ĉ_i: baseline 보정된 신뢰도 (calibrated confidence)
  - p_i: 방향성 점수 (+1.0 긍정 / +0.5 중립 / -1.0 부정)
  - t_i: 강도 점수 (1.5 강 / 1.0 보통 / 0.5 약)
  - n: 양성 카테고리 수
  - synergy(n): 4단계 Inverted-U (1.0 / 1.15 / 1.25 / 0.90)

학술 근거 (상세·검증이력: 점수식_학술근거.md):
  - 가중치 w: a priori. 검증된 출처 = Aguirre-Rodriguez et al.(2012)·Ladeira et al.(2023)·
    Babić Rosario et al.(2016)·Wilson & Sherrell(1993) 메타분석 + Cialdini(1984) 이론 등.
  - synergy 형태: Shu & Carlson(2014) 정점3/반전4 + Berlyne(1971) 역U + Brehm(1966) reactance
    + Friestad & Wright(1994)·Campbell(1995) 설득지식/조작의도.
  - calibration: 운영본에선 제거됨 (ablation 결과 과보정 — 점수계산_최종.py 참조).

한계 (점수식_학술근거.md §5):
  - 가중치·배수 구체값은 a priori 설계 파라미터 (데이터 fit 아님) → 민감도 분석으로 방어.
  - 실측 메타분석 r은 a priori와 다르며(대체로 더 작음) 그대로 정규화 시 카테고리 순위 역전.
  - ablation은 부분 동어반복(true_score가 동일 매핑 공유) → 구조 기여도 확인 용도.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 1. a priori 가중치 (이론·메타분석 정보 기반 사전 설정)
#    값은 검증된 실측 r이 아니라 a priori (점수식_학술근거.md §3.1).
#    주석의 cf.= 검증된 실측 효과크기 (정규화 분모로는 미사용).
# ============================================================
R_VALUES = {
    "권위_신뢰":     0.44,   # a priori; cf. Wilson & Sherrell(1993) 메타 expertise r≈.40 (η²→r)
    "사회적_증명":   0.317,  # a priori; cf. Babić Rosario et al.(2016) eWOM volume ρ=.141
    "사회적_정체성": 0.31,   # Aguirre-Rodriguez et al.(2012) 메타분석 r=.31 (검증 일치)
    "호혜성":        0.30,   # a priori (Cialdini 1984 이론); cf. DITF r=.126 (Feeley et al. 2012)
    "희소성":        0.25,   # a priori; cf. Ladeira et al.(2023) 메타 limited-quantity r≈.142
    "긴급성":        0.19,   # a priori; cf. Sun et al.(2023) zero-order r≈.29 (1차연구·DV 충동구매)
    "가격비교":      0.18,   # a priori (잠정 — 기존 β→r 무효); 후보 Santini et al.(2016) 페이월
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

    근거 (점수식_학술근거.md §3.4):
      - 정점3/반전4 위치: Shu & Carlson(2014, J. Marketing) "3 charms, 4 alarms"
      - n=2~3 상승: Petty & Cacioppo(1984) 주변경로 주장수 효과(저관여)
      - n≥4 반전: Brehm(1966) reactance + Friestad & Wright(1994) + Campbell(1995) 조작의도
      - 역U 형태: Berlyne(1971) Aesthetics and Psychobiology (※1970 논문 아님)

    한계: 배수(1.15/1.25/0.90)는 a priori 휴리스틱. 단위=카테고리 수(claim 수의 프록시).
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
