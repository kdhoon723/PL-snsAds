"""Gradio 추론 UI — SNS 광고 소비심리 자극 측정

실행:
  source .venv/bin/activate
  python 모델학습/app.py
"""
from __future__ import annotations
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "모델학습" / "runs"
CAT = ["사회적_정체성", "희소성", "긴급성", "사회적_증명", "가격비교", "권위_신뢰", "호혜성"]
WEIGHTS = {"권위_신뢰":0.221,"사회적_증명":0.159,"사회적_정체성":0.156,"호혜성":0.151,
           "희소성":0.126,"긴급성":0.096,"가격비교":0.091}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AdsClassifier(nn.Module):
    def __init__(self, m, n, d=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(m)
        self.dropout = nn.Dropout(d)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, n)
    def forward(self, i, a):
        o = self.backbone(input_ids=i, attention_mask=a)
        return self.classifier(self.dropout(o.last_hidden_state[:, 0]))


def load_run(run_name):
    run = RUNS / run_name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AdsClassifier(cfg["model_name"], len(CAT), cfg["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(run / "best_model.pt", weights_only=True, map_location=DEVICE))
    model.eval()
    thresholds = [result["best_thresholds"][c] for c in CAT]
    return tok, model, cfg["max_length"], thresholds


# 모델 미리 로드 (Cycle 19 단일 + Cycle 23 앙상블 멤버 3개)
print(f"🔧 Device: {DEVICE}")
print("📥 Loading models...")
MODELS = {}
for run_name in ["cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]:
    MODELS[run_name] = load_run(run_name)
    print(f"   ✓ {run_name}")

# Cycle 23 (앙상블) threshold
ENSEMBLE_THR = [
    json.loads((RUNS / "cycle23_ensemble_gamma1" / "result.json").read_text(encoding="utf-8"))["thresholds"][c]
    for c in CAT
]
print("✅ All models loaded\n")


@torch.no_grad()
def _predict_one(text: str, run_name: str):
    tok, model, max_len, _ = MODELS[run_name]
    enc = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    return torch.sigmoid(logits).cpu().numpy()[0]


def analyze(text: str, model_choice: str):
    text = text.strip()
    if not text:
        return None, None, "텍스트를 입력하세요", "{}"

    if model_choice == "Cycle 19 (단일, 빠름)":
        probs = _predict_one(text, "cycle19_dropout0.1")
        thr = MODELS["cycle19_dropout0.1"][3]
    else:  # Cycle 23 앙상블
        ps = [_predict_one(text, n) for n in
              ["cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]]
        probs = np.mean(ps, axis=0)
        thr = ENSEMBLE_THR

    preds = (probs >= np.array(thr)).astype(int)
    intensity = float(sum(WEIGHTS[c] * probs[i] for i, c in enumerate(CAT)))

    # 카테고리별 바차트 데이터 (확률, 임계값 함께)
    bar_data = [
        {"카테고리": c, "확률": float(probs[i]), "임계값": float(thr[i]),
         "양성": "✓" if preds[i] else "", "가중치": WEIGHTS[c]}
        for i, c in enumerate(CAT)
    ]

    # 강도 점수 텍스트
    level = ("강함 🔥" if intensity >= 0.4
             else "중간 ⚡" if intensity >= 0.2
             else "약함 💧" if intensity >= 0.05
             else "거의없음 ⚪")
    pos_cats = [c for i, c in enumerate(CAT) if preds[i]]
    pos_str = ", ".join(pos_cats) if pos_cats else "없음"

    summary_md = f"""
### 🎯 강도 점수: **{intensity:.4f}** ({level})

- 양성 카테고리: **{pos_str}**
- 카테고리 수: {len(pos_cats)} / 7
- 모델: {model_choice}
- 강도 범위 참고: 0.0 ~ 0.84 (학습 데이터 평균 0.146)
"""

    # 상세 JSON
    detail = {
        "text": text,
        "model": model_choice,
        "intensity_score": round(intensity, 4),
        "positive_categories": pos_cats,
        "per_category": {
            c: {
                "probability": round(float(probs[i]), 4),
                "threshold": round(float(thr[i]), 3),
                "predicted": int(preds[i]),
                "academic_weight": WEIGHTS[c],
            } for i, c in enumerate(CAT)
        },
    }

    # 강도 점수 0~1 게이지용
    return bar_data, intensity, summary_md, json.dumps(detail, ensure_ascii=False, indent=2)


# 예시 광고
EXAMPLES = [
    ["오늘만 50% 할인! 무료배송 ✨", "Cycle 19 (단일, 빠름)"],
    ["갓생러들의 필수템! 직장인 100명이 선택", "Cycle 19 (단일, 빠름)"],
    ["박보영이 직접 추천한 우리 동네 맛집", "Cycle 19 (단일, 빠름)"],
    ["한정판 콜라보 굿즈, 선착순 300명", "Cycle 19 (단일, 빠름)"],
    ["서울대 치과병원 구강외과 전문의 임플란트", "Cycle 19 (단일, 빠름)"],
    ["신규 가입 시 5만원 쿠폰팩 + 첫 구매 무료배송", "Cycle 23 (앙상블, 정확)"],
    ["대한민국 NO.1 헬스&뷰티 스토어 올리브영", "Cycle 23 (앙상블, 정확)"],
    ["내일은 없는 마지막 기회! 70% SALE", "Cycle 23 (앙상블, 정확)"],
]

# UI
with gr.Blocks(title="SNS 광고 소비심리 자극 측정", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🛍️ SNS 광고 소비심리 자극 측정

KLUE-RoBERTa-large fine-tuned (Macro F1 **0.7871**, 4,914건 학습).
7개 카테고리: 권위·신뢰 / 사회적 증명 / 사회적 정체성 / 호혜성 / 희소성 / 긴급성 / 가격비교.
강도 점수는 학술 가중치 합산.
""")

    with gr.Row():
        with gr.Column(scale=2):
            text_in = gr.Textbox(
                label="광고 텍스트",
                placeholder="예: 오늘만 50% 할인! 무료배송 ✨",
                lines=4,
            )
            model_choice = gr.Radio(
                ["Cycle 19 (단일, 빠름)", "Cycle 23 (앙상블, 정확)"],
                value="Cycle 19 (단일, 빠름)",
                label="모델",
            )
            btn = gr.Button("🔍 분석하기", variant="primary")

        with gr.Column(scale=3):
            summary_md = gr.Markdown(label="요약")
            intensity_slider = gr.Slider(
                minimum=0, maximum=1, value=0, step=0.001,
                label="강도 점수 (0~1)", interactive=False,
            )

    bar_df = gr.Dataframe(
        headers=["카테고리", "확률", "임계값", "양성", "가중치"],
        datatype=["str", "number", "number", "str", "number"],
        label="카테고리별 결과",
        interactive=False,
    )

    with gr.Accordion("상세 JSON", open=False):
        detail_json = gr.Code(language="json")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_in, model_choice],
        label="예시 광고",
    )

    # queue=False → SSE 안 거치고 직접 HTTP POST 동기 응답
    # (Cloudflare 통한 SSE 스트림 클라이언트 호환성 회피)
    btn.click(
        analyze,
        inputs=[text_in, model_choice],
        outputs=[bar_df, intensity_slider, summary_md, detail_json],
        queue=False,
    )
    text_in.submit(
        analyze,
        inputs=[text_in, model_choice],
        outputs=[bar_df, intensity_slider, summary_md, detail_json],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
