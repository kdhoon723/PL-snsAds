# 논문용 Figure 모음

종류별 개별 **정사각형** figure (논문 작성 시 골라 넣기 위함). 모두 **벡터 SVG + 300dpi PNG**, 색맹·흑백 안전.

- **SVG**: 한글이 벡터 패스로 임베드(`svg.fonttype=path`) → 폰트 없는 PC에서도 안 깨짐. Word·한글(HWP)·PowerPoint·Inkscape 직접 삽입/편집 가능.
- **PNG**: 어디든 붙는 300dpi 래스터.
- 미리보기: `_contact_sheet.png` (전체 11종 한 장)
- 개별 파일: `square/<name>.{svg,png}`
- 2-패널 조합본(참고): `fig1_system_performance.{svg,png}`, `fig2_v2_training_performance.{svg,png}`

## 재생성
```bash
source .venv/bin/activate
python 모델학습/figures/make_all_figs.py        # 01~09 (추론 불필요)
python 모델학습/figures/make_pr_roc.py          # 10~11 (cycle41 추론, 확률은 test_probs.npz 캐시)
python 모델학습/figures/make_contact_sheet.py   # 미리보기
```

## 목록

| 파일 | 증명 대상 | 차트 | 논문 위치 | 추론 |
|---|---|---|---|---|
| `01_learning_curve` | **학습이 잘됐다** (수렴·early stop·과적합 억제) | twin-axis loss↓+valF1↑ | 방법/결과 | ✕ |
| `02_per_category_f1` | **성능** 카테고리별 test F1 + macro 0.826 | 가로 막대 | 결과 ⭐ | ✕ |
| `03_v1_vs_v2` | v1→v2 **개선** (약한 카테고리 ↑) | 그룹 막대 | 결과/논의 | ✕ |
| `04_weights_grade` | **가중치 정당성 + 근거 등급**(A/B/C) | 막대+해칭 | 방법 ⭐ | ✕ |
| `05_synergy_curve` | 시너지 보정 설계 (Inverted-U) | 꺾은선 | 방법 | ✕ |
| `06_scoring_ablation` | 점수식 **설계 근거** (신뢰도 제거 채택) | 막대(MAE) | 결과/논의 | ✕ |
| `07_data_composition` | **클래스 불균형** (희귀 카테고리) | 가로 막대 | 데이터 | ✕ |
| `08_score_decomposition` | 점수식 **직관화** (예시 1건) | 누적 기여도 | 방법 ⭐ | ✕ |
| `09_irr_kappa` | **라벨 품질** (κ + Landis-Koch) | 막대+밴드 | 데이터/한계 | ✕ |
| `10_pr_curves` | **성능 (엄밀)** PR + AP, 불균형 정석 | 곡선 7+micro | 결과 ⭐ | ✓ |
| `11_roc_curves` | 성능 ROC + AUC (참고용) | 곡선 7+micro | 부록 | ✓ |
| `12_three_tasks` | **멀티태스크** 3 head(카테고리·방향성·강도) | 그룹 막대 | 결과 ⭐ | ✕ |
| `13_confusion` | **혼동행렬** 카테고리별 TP/FP/FN/TN | 2×2 ×7 | 결과/부록 | ✓ |
| `14_calibration` | **신뢰도 보정 곡선** + ECE (제거 정당화) | reliability | 논의 | ✓ |
| `15_score_scatter` | **점수 회귀 검증** 예측 vs 정답 (r·R²·MAE) | 산점도 | 결과 ⭐⭐ | ✓(앙상블) |

> 12·13·14는 단일 cycle41, **15는 4-앙상블+채택공식**(헤드라인 MAE 5.80·r 0.838과 일치). 06 ablation도 앙상블 기준.

## 핵심 수치 (figure 출처)
- 단일 모델 cycle41: val best F1 **0.819**(ep10), test macro-F1 **0.826**, macro-AP **0.878**, macro-AUC 0.961
- 카테고리별 F1: 호혜 0.88 / 희소 0.86 / 권위 0.85 / 가격·증명 0.84 / 긴급 0.81 / 정체성 0.69
- **멀티태스크 3 head F1**: 카테고리 0.826 / 방향성 0.928 / **강도 0.646**(최약, 주관성)
- **점수 회귀(앙상블)**: r **0.838**, R² 0.702, MAE **5.80** (예측 vs 정답)
- **신뢰도 보정 ECE 0.030** (raw 확률 잘 보정 → 후처리 제거 정당화)
- 점수식 ablation MAE: −신뢰도 **5.80**(채택) < Full 6.83 < 시너지제거 7.31 < v1 7.87 < 강도제거 8.12
- 데이터 양성비(6,149): 증명 22% … 정체성·희소 ~10% (불균형)
- IRR: 방향성 κ=0.666(substantial), 강도 κ=0.487(moderate)

## 2페이지 논문 추천 조합
- figure 1장만: **15**(점수 회귀 = 최종 출력 검증) 또는 **02/10**(분류 성능)
- figure 2장: **15**(점수 정확도) + **02/10**(분류 성능) ← 최종출력+분류 둘 다 / 또는 **08**(식 직관화)+**15**(검증)
- 멀티태스크 강조: **12**(3 head) 포함
- 정직성 어필: **04**(가중치 등급) + **14**(보정 제거 정당화)

## 전체 분류 (증명 대상별)
- **학습 품질**: 01
- **분류 성능**: 02, 10, 11, 13 · **멀티태스크**: 12
- **최종 점수(회귀)**: 15 · **점수식 설계**: 05, 06, 08
- **방법·데이터·정직성**: 03, 04, 07, 09, 14
