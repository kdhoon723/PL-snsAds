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
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
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
print("📥 Loading classifier models...")
MODELS = {}
for run_name in ["cycle19_dropout0.1", "cycle13_claude_only", "cycle18_dropout0.5"]:
    MODELS[run_name] = load_run(run_name)
    print(f"   ✓ {run_name}")
ENSEMBLE_THR = [
    json.loads((RUNS / "cycle23_ensemble_gamma1" / "result.json").read_text(encoding="utf-8"))["thresholds"][c]
    for c in CAT
]
print("✅ Classifiers loaded\n")

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


class AnalyzeReq(BaseModel):
    text: str
    model: Literal["single", "ensemble"] = "single"


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
      <span class="shortcut" id="inputHint">⌘+Enter · 드롭/⌘+V로 이미지</span>
      <button class="primary" id="go">분석</button>
    </div>
    <div class="drop-overlay" id="dropOverlay">
      <div class="drop-overlay-content">
        <div class="drop-icon-big">🖼️</div>
        <div class="drop-overlay-text">이미지를 놓으세요</div>
      </div>
    </div>
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

async function analyzeImage(file) {
  const result = $('#result');
  $('#go').disabled = true; $('#go').textContent = 'OCR + 분석 중';
  result.classList.add('show');
  result.innerHTML = '<div class="loading"><div class="spinner"></div><p>Qwen3-VL OCR + 7 카테고리 분류 중...</p></div>';
  try {
    const fd = new FormData(); fd.append('file', file);
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
  const model = 'ensemble';
  $('#go').disabled = true; $('#go').textContent = '분석 중';
  result.classList.add('show');
  result.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    const r = await fetch('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model }),
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
        "available": ["single", "ensemble"],
        "single": "Cycle 19 (klue/roberta-large + focal γ=1.0 + dropout 0.1)",
        "ensemble": "Cycle 23 (Cycle 19 + 13 + 18 평균)",
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
    return analyze(text, req.model)


@app.post("/api/analyze-image")
async def api_analyze_image(file: UploadFile = File(...)):
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
    result = analyze(text, "ensemble")
    result["ocr_text"] = text
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
