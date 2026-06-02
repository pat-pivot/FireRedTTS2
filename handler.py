"""RunPod Serverless handler for FireRedTTS-2.

Input shape (from RunPod job):
    {
        "input": {
            "text": "[S1]Hey. [S2]Hi.",         # required, dialogue format
            "temperature": 0.9,                  # optional
            "topk": 30,                          # optional
            "voice_mode": "random",              # "random" (default) or "clone"
            # Required when voice_mode="clone":
            "spk1_prompt_audio_url": "https://...",  # ref audio for [S1]
            "spk1_prompt_text":      "[S1] transcript of ref audio",
            # Optional second speaker for dialogue cloning:
            "spk2_prompt_audio_url": "https://...",
            "spk2_prompt_text":      "[S2] transcript of ref audio",
        }
    }

Output:
    { "audio_base64": "<wav bytes, base64>", "sample_rate": 24000 }
"""

import base64
import io
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import runpod
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from fireredtts2.fireredtts2 import FireRedTTS2

PRETRAINED_DIR = os.environ.get("PRETRAINED_DIR", "/app/pretrained_models/FireRedTTS2")
HF_REPO = os.environ.get("HF_REPO", "FireRedTeam/FireRedTTS2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download model weights on first worker boot — RunPod workers retain the
# local FS between requests, so the cost is amortized across the worker's life.
if not Path(PRETRAINED_DIR, "config.json").exists():
    print(f"[FireRedTTS2] Weights not found at {PRETRAINED_DIR}, downloading from {HF_REPO}...")
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=PRETRAINED_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"[FireRedTTS2] Download complete.")

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


def download_to_temp(url: str) -> str:
    """Download an audio URL to a temp .wav and return path."""
    suffix = ".mp3" if url.lower().split("?")[0].endswith(".mp3") else ".wav"
    fd, path = tempfile.mkstemp(prefix="firered_ref_", suffix=suffix)
    os.close(fd)
    print(f"[FireRedTTS2] Downloading ref audio: {url}")
    with urllib.request.urlopen(url) as resp, open(path, "wb") as out:
        out.write(resp.read())
    print(f"[FireRedTTS2] Saved ref audio to {path} ({os.path.getsize(path)} bytes)")
    return path


def ensure_speaker_prefix(text: str, speaker: str) -> str:
    """The model requires prompt_text to start with [S1]/[S2]. Add if missing."""
    text = text.strip()
    if re.match(r"^\[S[0-9]\]", text):
        return text
    return f"{speaker} {text}"


def build_prompts(job_input: dict) -> Tuple[Optional[list], Optional[list]]:
    """Return (prompt_wav_list, prompt_text_list) for clone mode, or (None, None)."""
    voice_mode = (job_input.get("voice_mode") or "random").lower()
    if voice_mode != "clone":
        return None, None

    spk1_url = job_input.get("spk1_prompt_audio_url") or job_input.get("spk1_prompt_audio")
    spk1_txt = job_input.get("spk1_prompt_text") or ""
    if not spk1_url or not spk1_txt.strip():
        raise ValueError(
            "voice_mode='clone' requires spk1_prompt_audio_url + spk1_prompt_text"
        )

    spk1_wav_path = download_to_temp(spk1_url)
    spk1_text = ensure_speaker_prefix(spk1_txt, "[S1]")
    prompt_wav_list = [spk1_wav_path]
    prompt_text_list = [spk1_text]

    spk2_url = job_input.get("spk2_prompt_audio_url") or job_input.get("spk2_prompt_audio")
    spk2_txt = job_input.get("spk2_prompt_text") or ""
    if spk2_url and spk2_txt.strip():
        spk2_wav_path = download_to_temp(spk2_url)
        spk2_text = ensure_speaker_prefix(spk2_txt, "[S2]")
        prompt_wav_list.append(spk2_wav_path)
        prompt_text_list.append(spk2_text)

    print(f"[FireRedTTS2] clone mode with {len(prompt_wav_list)} ref speaker(s)")
    return prompt_wav_list, prompt_text_list


def synthesize_dialogue(
    text: str,
    temperature: float,
    topk: int,
    prompt_wav_list: Optional[list],
    prompt_text_list: Optional[list],
) -> bytes:
    text_list = split_dialogue(text)
    if not text_list:
        raise ValueError(
            "No [S1]/[S2] turns parsed from text. Use format '[S1] line [S2] line'."
        )

    audio = model.generate_dialogue(
        text_list=text_list,
        prompt_wav_list=prompt_wav_list,
        prompt_text_list=prompt_text_list,
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
        prompt_wav_list, prompt_text_list = build_prompts(job_input)
        wav_bytes = synthesize_dialogue(
            text.strip(),
            temperature=temperature,
            topk=topk,
            prompt_wav_list=prompt_wav_list,
            prompt_text_list=prompt_text_list,
        )
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    return {
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
        "sample_rate": 24000,
        "turn_count": len(split_dialogue(text)),
        "voice_mode": (job_input.get("voice_mode") or "random").lower(),
    }


runpod.serverless.start({"handler": handler})
