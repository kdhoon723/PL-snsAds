FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/opt/hf \
    SNSADS_PROFILE=cpu-demo

RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    transformers==5.9.0 \
    fastapi==0.162.0 \
    uvicorn==0.54.0 \
    numpy==2.5.4 \
    scikit-learn==1.9.1 \
    pydantic==2.14.2 \
    pillow==12.2.0 \
    python-multipart==0.0.29

RUN python -c "from transformers import AutoConfig, AutoTokenizer; \
AutoTokenizer.from_pretrained('klue/roberta-large'); AutoConfig.from_pretrained('klue/roberta-large')" \
 && chmod -R a+rX /opt/hf

ENV HF_HUB_OFFLINE=1

WORKDIR /app/모델학습
COPY 모델학습/server.py 모델학습/train_v3_multitask.py 모델학습/scoring_v2.py ./
RUN mkdir runs

USER 1000:1000
EXPOSE 7860
CMD ["python", "server.py"]
