# Cloudflare Pages 정적 데모

이 폴더는 실제 Python/FastAPI/PyTorch 추론 서버 없이 발표 내용을 확인하기 위한 정적 배포본입니다.

## 구성

- `index.html` — 프로젝트 설명 페이지
- `demo.html` — 기존 웹 UI와 유사한 정적 데모 페이지
- `assets/site.css` — 공통 스타일


## 현재 배포

- Cloudflare Pages project: `snsads`
- Production URL: <https://snsads.pages.dev>
- Custom domain: <https://pl.example.invalid>
- 연결일: 2026-07-07

`pl.example.invalid`의 기존 Cloudflare Tunnel CNAME은 Pages 대상인 `snsads.pages.dev`로 전환했습니다.

## 배포 설정 예시

Cloudflare Pages에서 이 저장소를 연결할 때:

- Build command: 비워둠
- Build output directory: `cloudflare-pages`

## 주의

`demo.html`은 실제 모델 API를 호출하지 않습니다. 준비된 예시 문구에 대한 사전 계산 결과만 표시합니다.
실제 임의 문구 분석이나 이미지 OCR은 `모델학습/server.py` 기반 서버가 필요합니다.
