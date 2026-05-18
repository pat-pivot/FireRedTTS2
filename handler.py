"""RunPod Serverless handler for FireRedTTS-2.

Input shape (from RunPod job):
    {
        "input": {
            "text": "[S1] Hey. [S2] Hi.",        # required, dialogue format
            "temperature": 0.9,                   # optional
            "topk": 30,                           # optional
            "voice_mode": "random"                # "random" (default) or "clone"
        }
    }

Output:
    { "audio_base64": "<wav bytes, base64>", "sample_rate": 24000 }
"""

import base64
import io
import os
import re
from typing import List, Optional

import runpod
import soundfile as sf
import torch

from fireredtts2.fireredtts2 import FireRedTTS2

PRETRAINED_DIR = os.environ.get("PRETRAINED_DIR", "/app/pretrained_models/FireRedTTS2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[FireRedTTS2] Loading model from {PRETRAINED_DIR} on {DEVICE}...")
model = FireRedTTS2(
    pretrained_dir=PRETRAINED_DIR,
    gen_type="dialogue",
    device=DEVICE,
)
print("[FireRedTTS2] Model loaded.")


def split_dialogue(text: str) -> List[str]:
    """Split '[S1] hi [S2] hey' into ['[S1] hi', '[S2] hey']."""
    parts = re.findall(r"(\[S[0-9]\][^\[\]]*)", text)
    return [p.strip() for p in parts if p.strip()]


def synthesize_dialogue(text: str, temperature: float, topk: int) -> bytes:
    text_list = split_dialogue(text)
    if not text_list:
        raise ValueError(
            "No [S1]/[S2] turns parsed from text. Use format '[S1] line [S2] line'."
        )

    audio = model.generate_dialogue(
        text_list=text_list,
        prompt_wav_list=None,
        prompt_text_list=None,
        temperature=temperature,
        topk=topk,
    )
    audio_np = audio.squeeze(0).cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def handler(job):
    job_input = job.get("input", {})
    text: Optional[str] = job_input.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "Missing 'text' (non-empty string with [S1]/[S2] labels)."}

    temperature = float(job_input.get("temperature", 0.9))
    topk = int(job_input.get("topk", 30))

    try:
        wav_bytes = synthesize_dialogue(text.strip(), temperature=temperature, topk=topk)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    return {
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
        "sample_rate": 24000,
        "turn_count": len(split_dialogue(text)),
    }


runpod.serverless.start({"handler": handler})
