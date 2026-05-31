"""FastAPI 추론 서버 — SNS 광고 소비심리 자극 측정

실행:
  source .venv/bin/activate
  python 모델학습/server.py

엔드포인트:
  GET  /                   단일 페이지 HTML
  POST /api/analyze        {text, model} → JSON
  POST /api/analyze-image  multipart 이미지 → OCR → 분석
  GET  /api/models         사용 가능 모델 목록
"""
from __future__ import annotations
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
import uvicorn

PROJECT = Path("PROJECT_ROOT")
RUNS = PROJECT / "모델학습" / "runs"
CAT = ["사회적_정체성", "희소성", "긴급성", "사회적_증명", "가격비교", "권위_신뢰", "호혜성"]
WEIGHTS = {"권위_신뢰":0.221,"사회적_증명":0.159,"사회적_정체성":0.156,"호혜성":0.151,
           "희소성":0.126,"긴급성":0.096,"가격비교":0.091}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# v2 추가
sys.path.insert(0, str(PROJECT / "모델학습"))
from train_v3_multitask import MultiTaskClassifier, POLARITY, INTENSITY
from scoring_v2 import (
    CategoryResult, calculate_ad_score, calibrate_confidence,
    POLARITY_SCORE, INTENSITY_SCORE,
)


class AdsClassifier(nn.Module):
    def __init__(self, m, n, d=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(m)
        self.dropout = nn.Dropout(d)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, n)
    def forward(self, i, a):
        o = self.backbone(input_ids=i, attention_mask=a)
        return self.classifier(self.dropout(o.last_hidden_state[:, 0]))


def load_run(run_name: str):
    run = RUNS / run_name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AdsClassifier(cfg["model_name"], len(CAT), cfg["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(run / "best_model.pt", weights_only=True, map_location=DEVICE))
    model.eval()
    thresholds = [result["best_thresholds"][c] for c in CAT]
    return tok, model, cfg["max_length"], thresholds


print(f"🔧 Device: {DEVICE}")
print("📥 Loading v1 classifier models (Cycle 19/13/18, F1 0.787)...")
MODELS = {}
for run_name in ["cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]:
    MODELS[run_name] = load_run(run_name)
    print(f"   ✓ {run_name}")
ENSEMBLE_THR = [
    json.loads((RUNS / "cycle23_ensemble_gamma1" / "result.json").read_text(encoding="utf-8"))["thresholds"][c]
    for c in CAT
]
print("✅ v1 classifiers loaded")

# v2 multitask 모델 (Cycle 50 4-앙상블, F1 0.8296)
V2_MODELS = []
V2_THR = None
def load_v2():
    global V2_MODELS, V2_THR
    if V2_MODELS:
        return
    print("📥 Loading v2 multitask models (Cycle 41/48/39/46 ensemble, F1 0.830)...")
    for rn in ["cycle41_taskweight_cat", "cycle48_seed7", "cycle39_taskweight_balanced", "cycle46_dropout0.2"]:
        rdir = RUNS / rn
        cfg = json.loads((rdir / "config.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        m = MultiTaskClassifier(cfg["model"], cfg["dropout"]).to(DEVICE)
        m.load_state_dict(torch.load(rdir / "best_model.pt", weights_only=True, map_location=DEVICE))
        m.eval()
        V2_MODELS.append((tok, m, cfg["max_length"]))
        print(f"   ✓ {rn}")
    # Cycle 50 결과에서 threshold
    ens_result = json.loads((RUNS / "cycle50_ensemble_41_48_39_46" / "result.json").read_text(encoding="utf-8"))
    V2_THR = [ens_result["thresholds"][c] for c in CAT]
    print("✅ v2 multitask ensemble loaded\n")

# startup에서 v2도 로딩 (메모리 충분)
load_v2()

# Qwen3-VL OCR (광고 이미지 → 텍스트)
OCR_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OCR_AVAILABLE = False
OCR_PROCESSOR = None
OCR_MODEL = None

def load_ocr():
    """OCR 모델 lazy load (이미지 첫 요청 시)"""
    global OCR_PROCESSOR, OCR_MODEL, OCR_AVAILABLE
    if OCR_AVAILABLE:
        return
    print(f"📥 Loading OCR ({OCR_MODEL_NAME})...")
    OCR_PROCESSOR = AutoProcessor.from_pretrained(OCR_MODEL_NAME)
    OCR_MODEL = AutoModelForImageTextToText.from_pretrained(
        OCR_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    OCR_MODEL.eval()
    OCR_AVAILABLE = True
    print("✅ OCR loaded\n")


OCR_PROMPT = """이 광고 이미지의 본문 카피 텍스트만 추출해주세요.
- 사용자가 광고에서 읽는 한국어 본문/제목/슬로건 모두 포함
- 해시태그(#태그) 포함
- 로고·브랜드명·URL·도메인은 본문에 자연스럽게 포함된 경우에만 유지
- 단순 상품번호·바코드·이미지 캡션 같은 메타데이터는 제외
- 줄바꿈 그대로 유지
- 설명/메타 없이 추출된 텍스트만 출력"""


@torch.no_grad()
def extract_text(image: Image.Image) -> str:
    """이미지 → 추출된 광고 텍스트"""
    if not OCR_AVAILABLE:
        load_ocr()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": OCR_PROMPT},
    ]}]
    text = OCR_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = OCR_PROCESSOR(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(DEVICE)
    generated = OCR_MODEL.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return OCR_PROCESSOR.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def _predict(text: str, run_name: str):
    tok, model, max_len, _ = MODELS[run_name]
    enc = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    return torch.sigmoid(logits).cpu().numpy()[0]


def analyze(text: str, model_key: str):
    if model_key == "single":
        probs = _predict(text, "cycle19_dropout0.1")
        thr = MODELS["cycle19_dropout0.1"][3]
    else:  # ensemble
        ps = [_predict(text, n) for n in ["cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]]
        probs = np.mean(ps, axis=0)
        thr = ENSEMBLE_THR
    preds = (probs >= np.array(thr)).astype(int)
    intensity = float(sum(WEIGHTS[c] * probs[i] for i, c in enumerate(CAT)))
    level = ("강함 🔥" if intensity >= 0.4 else "중간 ⚡" if intensity >= 0.2
             else "약함 💧" if intensity >= 0.05 else "거의없음 ⚪")
    return {
        "version": "v1",
        "text": text,
        "model": "Cycle 19 단일" if model_key == "single" else "Cycle 23 앙상블",
        "intensity_score": round(intensity, 4),
        "intensity_level": level,
        "positive_categories": [c for i, c in enumerate(CAT) if preds[i]],
        "per_category": [
            {
                "category": c,
                "probability": round(float(probs[i]), 4),
                "threshold": round(float(thr[i]), 3),
                "predicted": int(preds[i]),
                "weight": WEIGHTS[c],
            } for i, c in enumerate(CAT)
        ],
    }


@torch.no_grad()
def analyze_v2(text: str, use_calibration: bool = True, use_synergy: bool = True):
    """v2 Multitask 앙상블 (Cycle 50 4-모델) + PDF + 팀원 A 점수 공식.

    use_calibration: PDF baseline 신뢰도 보정 (기본 True = PDF 원본)
    use_synergy: Inverted-U 시너지 보정 (기본 True = PDF 원본)
    """
    if not V2_MODELS:
        load_v2()
    cat_probs_acc = None
    pol_probs_acc = None
    int_probs_acc = None
    for tok, m, max_len in V2_MODELS:
        enc = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        cat_l, pol_l, int_l = m(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        cp = torch.sigmoid(cat_l).cpu().numpy()[0]
        pp = F.softmax(pol_l, dim=-1).cpu().numpy()[0]
        ip = F.softmax(int_l, dim=-1).cpu().numpy()[0]
        cat_probs_acc = cp if cat_probs_acc is None else cat_probs_acc + cp
        pol_probs_acc = pp if pol_probs_acc is None else pol_probs_acc + pp
        int_probs_acc = ip if int_probs_acc is None else int_probs_acc + ip
    n_models = len(V2_MODELS)
    cat_probs = cat_probs_acc / n_models
    pol_probs = pol_probs_acc / n_models
    int_probs = int_probs_acc / n_models

    thr = np.array(V2_THR)
    cat_preds = (cat_probs >= thr).astype(int)
    pol_preds = pol_probs.argmax(-1)
    int_preds = int_probs.argmax(-1)

    # CategoryResult → 점수 공식 (옵션 반영)
    results = []
    for i, c in enumerate(CAT):
        results.append(CategoryResult(
            category=c,
            probability=float(cat_probs[i]),
            is_positive=bool(cat_preds[i]),
            polarity=POLARITY[pol_preds[i]] if cat_preds[i] else None,
            intensity=INTENSITY[int_preds[i]] if cat_preds[i] else None,
        ))
    sr = calculate_ad_score(results, use_calibration=use_calibration, use_synergy=use_synergy)

    return {
        "version": "v2",
        "text": text,
        "model": "v2 보정 점수 (PDF) — Cycle 50 multitask 4-앙상블 (F1 0.8296)",
        "final_score_100": sr["final_score_100"],
        "raw_score": sr["raw_score"],
        "n_positive_categories": sr["n_positive_categories"],
        "synergy_factor": sr["synergy_factor"],
        "options": sr["options"],
        "per_category": [
            {
                "category": c,
                "probability": round(float(cat_probs[i]), 4),
                "threshold": round(float(thr[i]), 3),
                "predicted": int(cat_preds[i]),
                "weight": WEIGHTS[c],
                "polarity": POLARITY[pol_preds[i]] if cat_preds[i] else None,
                "intensity": INTENSITY[int_preds[i]] if cat_preds[i] else None,
                "calibrated_confidence": next((p["calibrated_confidence"] for p in sr["per_category"] if p["category"] == c), 0.0),
                "score_contribution": next((p["score"] for p in sr["per_category"] if p["category"] == c), 0.0),
            } for i, c in enumerate(CAT)
        ],
    }


class AnalyzeReq(BaseModel):
    text: str
    model: Literal["single", "ensemble", "v2"] = "ensemble"
    use_calibration: bool = True  # v2 한정 (PDF 신뢰도 baseline)
    use_synergy: bool = True       # v2 한정 (Inverted-U)


app = FastAPI(title="SNS 광고 소비심리 자극 측정")

INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>광고 소비심리 자극 측정</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg" crossorigin="anonymous"></script>
<style>
  :root {
    --bg: #fafafa;
    --surface: #ffffff;
    --border: #e7e5e4;
    --text: #1c1917;
    --text-soft: #78716c;
    --text-mute: #a8a29e;
    --accent: #18181b;
    --accent-soft: #f4f4f5;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --identity: #8b5cf6;
    --scarcity: #ec4899;
    --urgency: #f97316;
    --proof: #06b6d4;
    --price: #10b981;
    --authority: #6366f1;
    --recip: #eab308;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0a0a0a;
      --surface: #18181b;
      --border: #27272a;
      --text: #fafafa;
      --text-soft: #a1a1aa;
      --text-mute: #71717a;
      --accent: #fafafa;
      --accent-soft: #27272a;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--text);
    font-family: "Pretendard Variable", Pretendard, -apple-system, sans-serif;
    line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .container { max-width: 760px; margin: 0 auto; padding: 48px 24px 80px; }

  /* Header */
  header { margin-bottom: 32px; }
  .brand { display: inline-flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--text-mute); margin-bottom: 16px;
    letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; }
  .brand-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
  h1 { font-size: 32px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; }
  .sub { color: var(--text-soft); font-size: 15px; }
  .sub strong { color: var(--text); font-weight: 600; }

  /* Input card */
  .card { background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px; margin-bottom: 16px; }
  textarea { width: 100%; min-height: 90px; padding: 0; font-size: 16px;
    font-family: inherit; border: 0; resize: vertical; background: transparent;
    color: var(--text); outline: none; }
  textarea::placeholder { color: var(--text-mute); }
  .input-row { display: flex; align-items: center; margin-top: 16px;
    padding-top: 16px; border-top: 1px solid var(--border); gap: 12px; }
  .input-row .shortcut { flex: 1; }
  .icon-btn { width: 38px; height: 38px; padding: 0; background: transparent;
    border: 1px solid var(--border); border-radius: 10px; color: var(--text-soft);
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: 0.15s; flex-shrink: 0; }
  .icon-btn:hover { color: var(--text); border-color: var(--text-soft); background: var(--accent-soft); }
  .toggle { display: inline-flex; align-items: center; gap: 8px; font-size: 13px;
    color: var(--text-soft); cursor: pointer; user-select: none; }
  .toggle input { appearance: none; width: 32px; height: 18px; border-radius: 10px;
    background: var(--border); position: relative; cursor: pointer; transition: 0.2s; }
  .toggle input:checked { background: var(--accent); }
  .toggle input::after { content: ''; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%; background: var(--surface);
    transition: 0.2s; }
  .toggle input:checked::after { transform: translateX(14px); }
  button.primary { background: var(--accent); color: var(--bg); border: 0;
    padding: 10px 20px; border-radius: 10px; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: 0.15s; font-family: inherit; }
  button.primary:hover { transform: translateY(-1px); }
  button.primary:disabled { opacity: 0.5; cursor: wait; transform: none; }
  .shortcut { color: var(--text-mute); font-size: 12px; margin-left: 8px; }

  /* Sub-options (v2 calibration / synergy) */
  .subopts { display: flex; flex-direction: column; gap: 8px;
    padding: 10px 16px; background: var(--accent-soft); border-radius: 12px;
    margin-bottom: 16px; font-size: 12px; color: var(--text-soft); }
  .subopts.hide { display: none; }
  .subopts-row { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .subopts-label { font-weight: 600; color: var(--text); }
  .subopts label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
    user-select: none; }
  .subopts label:hover { color: var(--text); }
  .subopts label small { color: var(--text-mute); }
  .subopts input[type="checkbox"] { accent-color: var(--accent); cursor: pointer; }
  .formula-bar { display: flex; gap: 10px; align-items: center; padding-top: 8px;
    border-top: 1px dashed var(--border); flex-wrap: wrap; }
  .formula-label { color: var(--text-mute); font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600; }
  .formula-bar code { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace;
    font-size: 12px; color: var(--text); background: var(--surface);
    padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border);
    white-space: nowrap; overflow-x: auto; max-width: 100%; }
  .formula-math { color: var(--text); font-size: 14px; padding: 4px 0;
    overflow-x: auto; max-width: 100%; }
  .formula-math .katex { font-size: 1.0em; }
  .formula-raw { margin-left: 6px; }
  .formula-raw summary { cursor: pointer; color: var(--text-mute); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.05em; }
  .formula-raw summary:hover { color: var(--text-soft); }
  .formula-raw[open] code { display: inline-block; margin-top: 4px; }

  /* Glossary (각 항 설명) */
  .glossary { margin-top: 4px; border-top: 1px dashed var(--border); padding-top: 8px; }
  .glossary > summary { cursor: pointer; color: var(--text-soft); font-size: 12px;
    font-weight: 600; user-select: none; list-style: none; display: flex;
    align-items: center; gap: 6px; }
  .glossary > summary::-webkit-details-marker { display: none; }
  .glossary > summary::before { content: '▸'; transition: transform 0.15s; color: var(--text-mute); }
  .glossary[open] > summary::before { transform: rotate(90deg); }
  .glossary > summary:hover { color: var(--text); }
  .glossary-body { padding: 12px 4px 4px; }
  .glossary-intro { font-size: 13px; color: var(--text-soft); line-height: 1.6;
    margin-bottom: 12px; }
  .glossary-intro b { color: var(--text); }
  .gloss-item { display: flex; gap: 12px; padding: 8px 0;
    border-bottom: 1px solid var(--border); align-items: flex-start; }
  .gloss-item:last-of-type { border-bottom: 0; }
  .gloss-sym { flex-shrink: 0; min-width: 64px; font-family: "JetBrains Mono", "SF Mono", Menlo, monospace;
    font-size: 14px; font-weight: 700; color: var(--accent);
    background: var(--accent-soft); padding: 3px 8px; border-radius: 6px; text-align: center; }
  .gloss-desc { font-size: 13px; color: var(--text); line-height: 1.55; }
  .gloss-desc b { font-weight: 600; }
  .gloss-val { display: block; margin-top: 3px; font-size: 12px; color: var(--text-soft); }
  .gloss-val small { color: var(--text-mute); }
  .glossary-note { margin-top: 12px; padding: 10px 12px; background: var(--accent-soft);
    border-radius: 8px; font-size: 12px; color: var(--text-soft); line-height: 1.6; }
  .glossary-note b { color: var(--text); font-weight: 600; }

  /* Examples */
  .examples { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 32px; }
  .chip { padding: 6px 12px; font-size: 12px; color: var(--text-soft);
    background: var(--accent-soft); border-radius: 100px; cursor: pointer;
    transition: 0.15s; border: 1px solid transparent; }
  .chip:hover { color: var(--text); border-color: var(--border); }

  /* Result */
  #result { display: none; }
  #result.show { display: block; animation: fade 0.3s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  /* Intensity hero */
  .hero { text-align: center; padding: 36px 24px; margin-bottom: 16px; }
  .score { font-size: 72px; font-weight: 700; letter-spacing: -0.04em;
    line-height: 1; margin-bottom: 8px; font-variant-numeric: tabular-nums; }
  .level { font-size: 14px; color: var(--text-soft); text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600; }
  .level.high { color: var(--red); }
  .level.mid { color: var(--amber); }
  .level.low { color: var(--green); }
  .level.none { color: var(--text-mute); }
  .gauge { margin-top: 20px; height: 6px; background: var(--accent-soft);
    border-radius: 100px; overflow: hidden; max-width: 360px; margin-left: auto; margin-right: auto; }
  .gauge-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
    border-radius: 100px; transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1); }

  /* Active categories */
  .actives { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 16px; }
  .tag { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
    background: var(--accent-soft); border-radius: 100px; font-size: 13px; font-weight: 500; }
  .tag-dot { width: 6px; height: 6px; border-radius: 50%; }

  /* Category cards */
  .cats { display: grid; gap: 8px; }
  .cat-row { display: grid; grid-template-columns: 120px 1fr 50px;
    align-items: center; gap: 16px; padding: 12px 16px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    transition: 0.15s; }
  .cat-row.pos { border-color: transparent; }
  .cat-row.pos[data-c="사회적_정체성"] { background: color-mix(in srgb, var(--identity) 8%, var(--surface)); }
  .cat-row.pos[data-c="희소성"] { background: color-mix(in srgb, var(--scarcity) 8%, var(--surface)); }
  .cat-row.pos[data-c="긴급성"] { background: color-mix(in srgb, var(--urgency) 8%, var(--surface)); }
  .cat-row.pos[data-c="사회적_증명"] { background: color-mix(in srgb, var(--proof) 8%, var(--surface)); }
  .cat-row.pos[data-c="가격비교"] { background: color-mix(in srgb, var(--price) 8%, var(--surface)); }
  .cat-row.pos[data-c="권위_신뢰"] { background: color-mix(in srgb, var(--authority) 8%, var(--surface)); }
  .cat-row.pos[data-c="호혜성"] { background: color-mix(in srgb, var(--recip) 8%, var(--surface)); }
  .cat-name { display: flex; align-items: center; gap: 8px; font-weight: 500; font-size: 14px; }
  .cat-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .cat-bar { background: var(--accent-soft); border-radius: 100px; height: 8px; position: relative;
    overflow: hidden; }
  .cat-bar-fill { height: 100%; border-radius: 100px;
    transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1); }
  .cat-bar-thr { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--text-mute);
    border-radius: 1px; opacity: 0.5; }
  .cat-prob { font-size: 13px; color: var(--text-soft); text-align: right;
    font-variant-numeric: tabular-nums; }
  .cat-row.pos .cat-prob { color: var(--text); font-weight: 600; }

  /* Meta */
  .meta { display: flex; justify-content: space-between; margin-top: 24px;
    padding-top: 16px; border-top: 1px solid var(--border); font-size: 12px;
    color: var(--text-mute); }
  details summary { cursor: pointer; color: var(--text-mute); font-size: 12px;
    user-select: none; margin-top: 16px; }
  details summary:hover { color: var(--text-soft); }
  details pre { background: var(--accent-soft); padding: 16px; border-radius: 8px;
    overflow: auto; font-size: 12px; margin-top: 8px; }

  /* Loading */
  .loading { text-align: center; padding: 48px 0; color: var(--text-soft); }
  .loading p { margin-top: 12px; font-size: 13px; }
  .spinner { display: inline-block; width: 24px; height: 24px;
    border: 2px solid var(--accent-soft); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: var(--red); padding: 16px; background: color-mix(in srgb, var(--red) 8%, var(--surface));
    border-radius: 12px; border: 1px solid color-mix(in srgb, var(--red) 20%, transparent); }

  .shortcut { color: var(--text-mute); font-size: 12px; }

  /* Image preview inside input card */
  #inputCard { position: relative; }
  .img-preview { position: relative; margin-bottom: 12px; padding: 8px;
    background: var(--accent-soft); border-radius: 10px; text-align: center; }
  #imgPreviewSrc { max-width: 100%; max-height: 240px; border-radius: 6px;
    display: block; margin: 0 auto; }
  .img-clear { position: absolute; top: 12px; right: 12px;
    width: 28px; height: 28px; border-radius: 50%; background: rgba(0,0,0,0.6);
    color: white; border: 0; cursor: pointer; font-size: 14px; line-height: 1;
    display: flex; align-items: center; justify-content: center; }
  .img-clear:hover { background: rgba(0,0,0,0.85); }

  /* Drop overlay (드래그할 때 textarea 위 덮어서 표시) */
  .drop-overlay { position: absolute; inset: 0; background: color-mix(in srgb, var(--accent) 6%, var(--surface));
    border: 2px dashed var(--accent); border-radius: 16px; display: none;
    align-items: center; justify-content: center; pointer-events: none;
    z-index: 10; }
  .drop-overlay.show { display: flex; }
  .drop-overlay-content { text-align: center; }
  .drop-icon-big { font-size: 64px; margin-bottom: 8px; }
  .drop-overlay-text { font-size: 16px; font-weight: 600; color: var(--text); }

  /* OCR result box */
  .ocr-box { background: var(--accent-soft); padding: 12px 16px; border-radius: 10px;
    margin-bottom: 16px; font-size: 14px; line-height: 1.6; }
  .ocr-label { font-size: 11px; color: var(--text-mute); text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600; margin-bottom: 6px; }
  .ocr-text { white-space: pre-wrap; color: var(--text); }

  @media (max-width: 600px) {
    .container { padding: 24px 16px 48px; }
    h1 { font-size: 24px; }
    .score { font-size: 56px; }
    .cat-row { grid-template-columns: 100px 1fr 44px; gap: 10px; padding: 10px 12px; }
    .cat-name { font-size: 13px; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand"><span class="brand-dot"></span> AD Persuasion Meter</div>
    <h1>광고 소비심리 자극 측정</h1>
    <p class="sub">SNS 광고의 7가지 설득 전략을 탐지하고 자극 강도를 정량화합니다.
      <strong>KLUE-RoBERTa-large</strong> · F1 0.787</p>
  </header>

  <div class="card" id="inputCard">
    <div class="img-preview" id="imgPreview" style="display:none">
      <img id="imgPreviewSrc">
      <button class="img-clear" id="imgClear" title="이미지 제거">✕</button>
    </div>
    <textarea id="text" placeholder="광고 텍스트를 입력하세요  ·  📎로 이미지 첨부도 가능"></textarea>
    <input type="file" id="fileInput" accept="image/*" hidden>
    <div class="input-row">
      <button class="icon-btn" id="attachBtn" title="이미지 첨부 (촬영/갤러리)" aria-label="이미지 첨부">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 17.93 8.8L9.41 17.34a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
        </svg>
      </button>
      <label class="toggle" title="v2: PDF 점수 공식 (학술 가중치 × 신뢰도 보정 × 방향성 × 강도 × 시너지). 카테고리 F1 0.830 (v1 0.787 대비 +0.043)">
        <input type="checkbox" id="v2toggle" checked>
        <span>보정 점수 (PDF)</span>
      </label>
      <span class="shortcut" id="inputHint" style="flex:1; text-align:right">⌘+Enter</span>
      <button class="primary" id="go">분석</button>
    </div>
    <div class="drop-overlay" id="dropOverlay">
      <div class="drop-overlay-content">
        <div class="drop-icon-big">🖼️</div>
        <div class="drop-overlay-text">이미지를 놓으세요</div>
      </div>
    </div>
  </div>

  <div class="subopts" id="v2subopts">
    <div class="subopts-row">
      <span class="subopts-label">v2 옵션:</span>
      <label title="PDF 신뢰도 baseline 보정. 카테고리별 baseline(권위 0.90 / 정체성 0.65 등)까지 prob를 끌어올림. ablation 결과 OFF가 MAE -1.03 개선">
        <input type="checkbox" id="useCalibration" checked> 신뢰도 보정 <small>(PDF baseline)</small>
      </label>
      <label title="Inverted-U 시너지: n=2 ×1.15 / n=3 ×1.25 / n≥4 ×0.90. PDF 원본은 단조증가 (1.00/1.15/1.30/1.45)였으나 팀원 A 의견 융합해 Inverted-U로 수정">
        <input type="checkbox" id="useSynergy" checked> 시너지 보정 <small>(Inverted-U, PDF 수정)</small>
      </label>
    </div>
    <div class="formula-bar" id="formulaBar">
      <span class="formula-label">현재 공식</span>
      <div id="formulaDisplay" class="formula-math"></div>
      <details class="formula-raw">
        <summary>LaTeX</summary>
        <code id="formulaTex"></code>
      </details>
    </div>
    <details class="glossary">
      <summary>📖 각 항이 무슨 뜻이야? (펼쳐서 보기)</summary>
      <div class="glossary-body">
        <p class="glossary-intro">광고 하나의 점수는 <b>탐지된 카테고리마다 점수를 매겨 더한 뒤, 시너지로 보정</b>해서 0~100점으로 만듭니다. 각 항의 뜻:</p>
        <div class="gloss-item">
          <span class="gloss-sym">w</span>
          <div class="gloss-desc">
            <b>카테고리 가중치</b> — 그 자극이 실제 구매에 미치는 영향력. 논문 메타분석 수치로 정함.
            <span class="gloss-val">권위 0.221 · 증명 0.159 · 정체성 0.156 · 호혜 0.151 · 희소 0.126 · 긴급 0.096 · 가격 0.091 <small>(합 1.0)</small></span>
          </div>
        </div>
        <div class="gloss-item">
          <span class="gloss-sym">ĉ</span>
          <div class="gloss-desc">
            <b>보정된 확신도 (0~1)</b> — 모델이 "이 광고에 이 자극이 있다"고 얼마나 확신하는지.
            <span class="gloss-val">신뢰도 보정 ON이면, 잘 안 잡히는 약한 카테고리(정체성 등)를 기준선까지 끌어올림. <small>OFF면 모델 원본 확률 그대로 사용 → 실험 결과 OFF가 더 정확했음</small></span>
          </div>
        </div>
        <div class="gloss-item">
          <span class="gloss-sym">p</span>
          <div class="gloss-desc">
            <b>방향성</b> — 광고가 어느 쪽으로 미는지.
            <span class="gloss-val">긍정(사라고 권유) +1.0 · 중립(정보 안내) +0.5 · 부정(하지 말라고 억제, 공익광고) −1.0</span>
          </div>
        </div>
        <div class="gloss-item">
          <span class="gloss-sym">t</span>
          <div class="gloss-desc">
            <b>강도</b> — 그 자극이 얼마나 센지.
            <span class="gloss-val">강(70% 할인·100명 한정 등 구체적·극단적) 1.5 · 보통 1.0 · 약(추상적 표현) 0.5</span>
          </div>
        </div>
        <div class="gloss-item">
          <span class="gloss-sym">synergy(n)</span>
          <div class="gloss-desc">
            <b>시너지 보정</b> — 여러 자극을 동시에 쓰면 효과가 달라짐.
            <span class="gloss-val">1개 ×1.0 · 2개 ×1.15 · 3개 ×1.25 (같이 쓰면 상승) · 4개↑ ×0.90 (너무 많으면 오히려 거부감 ↓)</span>
          </div>
        </div>
        <div class="gloss-item">
          <span class="gloss-sym">n</span>
          <div class="gloss-desc"><b>탐지된 양성 카테고리 수</b> — 이 광고에서 발견된 자극 종류의 개수.</div>
        </div>
        <div class="gloss-item">
          <span class="gloss-sym">×100</span>
          <div class="gloss-desc"><b>점수 환산</b> — 0~1 사이 값을 보기 쉽게 0~100점으로.</div>
        </div>
        <p class="glossary-note">⚠️ 이 점수 공식은 팀원이 만든 설계서(PDF) 기반입니다. 단 <b>시너지는 원본(1.0/1.15/1.30/1.45 계속 증가)과 다르게</b>, "자극이 너무 많으면 역효과"라는 의견을 반영해 3개에서 정점 찍고 내려가는 형태(1.0/1.15/1.25/0.90)로 바꿨습니다. 신뢰도 보정 ON/OFF, 시너지 ON/OFF는 위 체크박스로 직접 비교해볼 수 있습니다.</p>
      </div>
    </details>
  </div>

  <div class="examples">
    <span class="chip">오늘만 50% 할인! 무료배송 ✨</span>
    <span class="chip">갓생러들의 필수템! 직장인 100명이 선택</span>
    <span class="chip">박보영이 직접 추천한 우리 동네 맛집</span>
    <span class="chip">한정판 콜라보 굿즈, 선착순 300명</span>
    <span class="chip">서울대 치과병원 전문의 임플란트</span>
    <span class="chip">신규 가입 시 5만원 쿠폰팩 + 첫 구매 무료배송</span>
    <span class="chip">대한민국 NO.1 헬스&뷰티 스토어</span>
    <span class="chip">내일은 없는 마지막 기회! 70% SALE</span>
  </div>

  <div id="result"></div>
</div>

<script>
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const COLORS = {
  '사회적_정체성': 'var(--identity)',
  '희소성': 'var(--scarcity)',
  '긴급성': 'var(--urgency)',
  '사회적_증명': 'var(--proof)',
  '가격비교': 'var(--price)',
  '권위_신뢰': 'var(--authority)',
  '호혜성': 'var(--recip)',
};

$$('.chip').forEach(el => el.addEventListener('click', () => {
  clearImage();
  $('#text').value = el.textContent.trim();
  $('#text').focus();
  analyze();
}));

$('#text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); }
});
$('#go').addEventListener('click', submit);
$('#imgClear').addEventListener('click', clearImage);
$('#attachBtn').addEventListener('click', () => $('#fileInput').click());
$('#fileInput').addEventListener('change', e => {
  if (e.target.files[0]) setImage(e.target.files[0]);
  e.target.value = '';  // 같은 파일 재선택 가능
});

// 통합 submit: 이미지 있으면 OCR+분석, 없으면 텍스트만
let currentImage = null;
function submit() {
  if (currentImage) analyzeImage(currentImage);
  else analyze();
}

function clearImage() {
  currentImage = null;
  $('#imgPreview').style.display = 'none';
  $('#imgPreviewSrc').src = '';
}

function setImage(file) {
  if (!file || !file.type.startsWith('image/')) return false;
  if (file.size > 10 * 1024 * 1024) {
    alert('이미지가 너무 큽니다 (최대 10MB)'); return false;
  }
  currentImage = file;
  const reader = new FileReader();
  reader.onload = e => {
    $('#imgPreviewSrc').src = e.target.result;
    $('#imgPreview').style.display = '';
  };
  reader.readAsDataURL(file);
  // 자동 분석
  analyzeImage(file);
  return true;
}

// Drag & drop on textarea/card
const card = $('#inputCard'), overlay = $('#dropOverlay');
let dragCounter = 0;
window.addEventListener('dragenter', e => {
  if (e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files')) {
    dragCounter++; overlay.classList.add('show');
  }
});
window.addEventListener('dragleave', e => {
  dragCounter--; if (dragCounter <= 0) { dragCounter = 0; overlay.classList.remove('show'); }
});
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => {
  e.preventDefault(); dragCounter = 0; overlay.classList.remove('show');
  const file = e.dataTransfer.files[0];
  if (file) setImage(file);
});

// Paste image (Ctrl/Cmd+V)
window.addEventListener('paste', e => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) { e.preventDefault(); setImage(file); return; }
    }
  }
});

function getModelChoice() {
  return $('#v2toggle').checked ? 'v2' : 'ensemble';
}
function getV2Options() {
  return {
    use_calibration: $('#useCalibration').checked,
    use_synergy: $('#useSynergy').checked,
  };
}
// v2 토글에 따라 sub-options 표시/숨김
function syncV2Subopts() {
  $('#v2subopts').classList.toggle('hide', !$('#v2toggle').checked);
}
$('#v2toggle').addEventListener('change', syncV2Subopts);
syncV2Subopts();

// 옵션 변경 시 현재 공식 갱신 (KaTeX 렌더링)
function buildFormulaTex(useCal, useSyn) {
  const conf = useCal ? '\\hat{c}_i' : '\\text{prob}_i';
  const syn = useSyn ? '\\cdot \\text{synergy}(n) ' : '';
  return `\\text{score} = \\sum_{i \\in \\text{positive}} w_i \\cdot ${conf} \\cdot p_i \\cdot t_i ${syn}\\times 100`;
}
function updateFormula() {
  const useCal = $('#useCalibration').checked;
  const useSyn = $('#useSynergy').checked;
  const tex = buildFormulaTex(useCal, useSyn);
  const el = $('#formulaDisplay');
  $('#formulaTex').textContent = tex;
  if (window.katex) {
    try {
      katex.render(tex, el, { throwOnError: false, displayMode: false });
    } catch (e) {
      el.textContent = tex;
    }
  } else {
    el.textContent = tex;  // KaTeX 로딩 전 fallback
  }
}
$('#useCalibration').addEventListener('change', updateFormula);
$('#useSynergy').addEventListener('change', updateFormula);
// KaTeX는 defer 로딩 — DOMContentLoaded 후 렌더링
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', updateFormula);
} else {
  // 이미 로드됐으면 KaTeX 로딩 대기
  if (window.katex) updateFormula();
  else window.addEventListener('load', updateFormula);
}

async function analyzeImage(file) {
  const result = $('#result');
  $('#go').disabled = true; $('#go').textContent = 'OCR + 분석 중';
  result.classList.add('show');
  result.innerHTML = '<div class="loading"><div class="spinner"></div><p>Qwen3-VL OCR + 7 카테고리 분류 중...</p></div>';
  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('model', getModelChoice());
    const opts = getV2Options();
    fd.append('use_calibration', opts.use_calibration);
    fd.append('use_synergy', opts.use_synergy);
    const r = await fetch('/api/analyze-image', { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: 'HTTP ' + r.status}));
      throw new Error(err.detail || ('HTTP ' + r.status));
    }
    const data = await r.json();
    $('#text').value = data.ocr_text;
    render(data);
  } catch (e) {
    result.innerHTML = `<div class="error">${e.message}</div>`;
  } finally {
    $('#go').disabled = false; $('#go').textContent = '분석';
  }
}

async function analyze() {
  const text = $('#text').value.trim();
  const result = $('#result');
  if (!text) { result.classList.add('show'); result.innerHTML = '<div class="error">텍스트를 입력하세요</div>'; return; }
  const model = getModelChoice();
  const body = { text, model, ...getV2Options() };
  $('#go').disabled = true; $('#go').textContent = '분석 중';
  result.classList.add('show');
  result.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    const r = await fetch('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    render(await r.json());
  } catch (e) {
    result.innerHTML = `<div class="error">${e.message}</div>`;
  } finally {
    $('#go').disabled = false; $('#go').textContent = '분석';
  }
}

function render(d) {
  if (d.version === 'v2') return renderV2(d);
  return renderV1(d);
}

function renderV1(d) {
  const pct = Math.min(100, d.intensity_score / 0.84 * 100);
  const levelClass = d.intensity_score >= 0.4 ? 'high'
                  : d.intensity_score >= 0.2 ? 'mid'
                  : d.intensity_score >= 0.05 ? 'low' : 'none';
  const levelTxt = d.intensity_score >= 0.4 ? '강한 자극'
                 : d.intensity_score >= 0.2 ? '중간 자극'
                 : d.intensity_score >= 0.05 ? '약한 자극' : '거의 없음';

  const tags = d.positive_categories.map(c =>
    `<span class="tag"><span class="tag-dot" style="background:${COLORS[c]}"></span>${c.replace('_', ' ')}</span>`
  ).join('');

  // 가중치 큰 카테고리부터 정렬
  const sorted = [...d.per_category].sort((a, b) => b.weight - a.weight);
  const rows = sorted.map(c => {
    const w = Math.min(100, c.probability * 100);
    const tpct = Math.min(100, c.threshold * 100);
    const cls = c.predicted ? 'pos' : '';
    return `<div class="cat-row ${cls}" data-c="${c.category}">
      <div class="cat-name"><span class="cat-dot" style="background:${COLORS[c.category]}"></span>${c.category.replace('_', ' ')}</div>
      <div class="cat-bar">
        <div class="cat-bar-fill" style="width:${w}%; background:${COLORS[c.category]}"></div>
        <div class="cat-bar-thr" style="left:${tpct}%"></div>
      </div>
      <div class="cat-prob">${(c.probability * 100).toFixed(0)}%</div>
    </div>`;
  }).join('');

  const ocrBlock = d.ocr_text
    ? `<div class="ocr-box"><div class="ocr-label">📷 추출된 텍스트</div><div class="ocr-text">${escapeHtml(d.ocr_text)}</div></div>`
    : '';

  $('#result').innerHTML = `
    ${ocrBlock}
    <div class="card hero">
      <div class="level ${levelClass}">${levelTxt}</div>
      <div class="score">${d.intensity_score.toFixed(3)}</div>
      <div class="gauge"><div class="gauge-fill" style="width:${pct}%"></div></div>
      ${tags ? `<div class="actives">${tags}</div>` : ''}
    </div>
    <div class="cats">${rows}</div>
    <div class="meta">
      <span>막대=확률 · 회색 선=임계값</span>
      <span>가중치 순 정렬</span>
    </div>
    <details><summary>상세 JSON</summary><pre>${JSON.stringify(d, null, 2)}</pre></details>
  `;
}

function renderV2(d) {
  // 점수 0~100, 시너지 표시
  const score = d.final_score_100;
  const pct = Math.min(100, Math.max(0, score));
  const levelClass = score >= 50 ? 'high' : score >= 25 ? 'mid' : score >= 5 ? 'low' : 'none';
  const levelTxt = score >= 50 ? '강한 자극'
                 : score >= 25 ? '중간 자극'
                 : score >= 5 ? '약한 자극' : '거의 없음';

  const POL_EMOJI = { '긍정': '↑', '중립': '·', '부정': '↓' };
  const INT_EMOJI = { '강': '●●●', '보통': '●●', '약': '●' };

  const tags = d.per_category.filter(c => c.predicted).map(c =>
    `<span class="tag" title="${c.polarity || ''} ${c.intensity || ''}">
      <span class="tag-dot" style="background:${COLORS[c.category]}"></span>
      ${c.category.replace('_', ' ')}
    </span>`
  ).join('');

  // 가중치 큰 순 정렬
  const sorted = [...d.per_category].sort((a, b) => b.weight - a.weight);
  const rows = sorted.map(c => {
    const w = Math.min(100, c.probability * 100);
    const tpct = Math.min(100, c.threshold * 100);
    const cls = c.predicted ? 'pos' : '';
    const tags = c.predicted
      ? `<span style="font-size:11px;color:var(--text-soft);margin-left:6px">${POL_EMOJI[c.polarity]} ${INT_EMOJI[c.intensity]}</span>`
      : '';
    return `<div class="cat-row ${cls}" data-c="${c.category}">
      <div class="cat-name">
        <span class="cat-dot" style="background:${COLORS[c.category]}"></span>
        ${c.category.replace('_', ' ')}${tags}
      </div>
      <div class="cat-bar">
        <div class="cat-bar-fill" style="width:${w}%; background:${COLORS[c.category]}"></div>
        <div class="cat-bar-thr" style="left:${tpct}%"></div>
      </div>
      <div class="cat-prob">${(c.probability * 100).toFixed(0)}%</div>
    </div>`;
  }).join('');

  const ocrBlock = d.ocr_text
    ? `<div class="ocr-box"><div class="ocr-label">📷 추출된 텍스트</div><div class="ocr-text">${escapeHtml(d.ocr_text)}</div></div>`
    : '';

  const opts = d.options || {use_calibration: true, use_synergy: true};
  const optBadge = [
    opts.use_calibration ? '신뢰도 보정 ON' : '신뢰도 OFF',
    opts.use_synergy ? '시너지 ON' : '시너지 OFF',
  ].join(' · ');
  const usedFormulaTex = buildFormulaTex(opts.use_calibration, opts.use_synergy);
  const syn = d.n_positive_categories >= 1
    ? `n=${d.n_positive_categories} 카테고리 × 시너지 ×${d.synergy_factor} · ${optBadge}`
    : `양성 카테고리 없음 · ${optBadge}`;

  $('#result').innerHTML = `
    ${ocrBlock}
    <div class="card hero">
      <div class="level ${levelClass}">${levelTxt} · 보정 점수 (PDF)</div>
      <div class="score">${score.toFixed(1)}<span style="font-size:24px;color:var(--text-mute);font-weight:500"> / 100</span></div>
      <div class="gauge"><div class="gauge-fill" style="width:${pct}%"></div></div>
      <div style="font-size:12px;color:var(--text-mute);margin-top:12px">${syn}</div>
      ${tags ? `<div class="actives">${tags}</div>` : ''}
    </div>
    <div class="cats">${rows}</div>
    <div class="meta">
      <span>↑긍정 ·중립 ↓부정 / ●●●강 ●●보통 ●약</span>
      <span>Cycle 50 앙상블 (F1 0.830)</span>
    </div>
    <div class="formula-bar" style="margin-top:12px;background:var(--accent-soft);padding:10px 14px;border-radius:10px;border:0">
      <span class="formula-label">사용된 공식</span>
      <div id="usedFormulaMath" class="formula-math"></div>
      <details class="formula-raw">
        <summary>LaTeX</summary>
        <code>${usedFormulaTex}</code>
      </details>
    </div>
    <details><summary>상세 JSON</summary><pre>${JSON.stringify(d, null, 2)}</pre></details>
  `;
  // 사용된 공식 KaTeX 렌더
  if (window.katex) {
    try { katex.render(usedFormulaTex, $('#usedFormulaMath'), { throwOnError: false, displayMode: false }); }
    catch (e) { $('#usedFormulaMath').textContent = usedFormulaTex; }
  } else {
    $('#usedFormulaMath').textContent = usedFormulaTex;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/models")
def models():
    return {
        "available": ["single", "ensemble", "v2"],
        "single": "v1 단일: Cycle 19 (multi-label sigmoid, F1 0.778, 단순 강도점수 Σw·prob)",
        "ensemble": "v1 앙상블: Cycle 23 (Cycle 19+13+18, F1 0.787, 단순 강도점수 Σw·prob)",
        "v2": "v2 보정 점수 (PDF): Cycle 50 multitask 앙상블 (Cycle 41+48+39+46, F1 0.830) + 학술 보정 공식 Σ(w·ĉ·p·t)·synergy·100",
        "weights": WEIGHTS,
        "categories": CAT,
    }


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > 4000:
        raise HTTPException(400, "text too long (max 4000 chars)")
    if req.model == "v2":
        return analyze_v2(text, use_calibration=req.use_calibration, use_synergy=req.use_synergy)
    return analyze(text, req.model)


@app.post("/api/analyze-image")
async def api_analyze_image(
    file: UploadFile = File(...),
    model: str = Form("ensemble"),
    use_calibration: bool = Form(True),
    use_synergy: bool = Form(True),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "이미지 파일이 아닙니다")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "이미지가 너무 큽니다 (최대 10MB)")
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"이미지 로드 실패: {e}")
    # 너무 큰 이미지는 리사이즈 (Qwen3-VL 입력 효율)
    max_dim = 1280
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    text = extract_text(img)
    if not text.strip():
        raise HTTPException(422, "이미지에서 텍스트를 추출하지 못했습니다")
    if model == "v2":
        result = analyze_v2(text, use_calibration=use_calibration, use_synergy=use_synergy)
    else:
        result = analyze(text, model if model in ("single", "ensemble") else "ensemble")
    result["ocr_text"] = text
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
