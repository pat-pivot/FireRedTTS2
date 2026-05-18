FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# System deps for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps. Match FireRedTTS-2 docker recipe + RunPod + HF downloader.
RUN pip install --no-cache-dir \
    soundfile==0.12.1 \
    torchao==0.10.0 \
    torchtune \
    "transformers>=4.45,<4.55" \
    einops \
    librosa \
    huggingface_hub \
    hf_transfer \
    runpod

# Copy repo, install fireredtts2 package
COPY . /app/
RUN pip install --no-cache-dir -e /app

# Pre-download model weights into image so cold start is fast.
# Skip if HF_HUB_TOKEN env is needed for gated models (FireRedTTS-2 is open).
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='FireRedTeam/FireRedTTS2', \
                      local_dir='/app/pretrained_models/FireRedTTS2', \
                      local_dir_use_symlinks=False)"

CMD ["python", "-u", "handler.py"]
