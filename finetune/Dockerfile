FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers==4.45.0" \
    "datasets==2.21.0" \
    "accelerate==0.34.0" \
    "evaluate==0.4.3" \
    "jiwer==3.0.4" \
    "soundfile==0.12.1" \
    "librosa==0.10.2"

ENV HF_HOME=/app/hf_cache

RUN python -c "from transformers import WhisperForConditionalGeneration, WhisperProcessor; import evaluate; WhisperProcessor.from_pretrained('openai/whisper-small'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-small'); evaluate.load('wer'); print('All cached.')"

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]