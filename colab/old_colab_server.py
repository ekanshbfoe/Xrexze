"""
╔══════════════════════════════════════════════════════════════╗
║  XREXZE COLAB BACKEND — Unified GPU Server                  ║
║  Qwen2.5-VL-7B (4-bit) + OmniVoice on Free T4 GPU         ║
║                                                              ║
║  USAGE: Copy-paste into a Google Colab notebook cell,        ║
║  or upload this file and run: !python colab_server.py        ║
║                                                              ║
║  Requires: Colab T4 GPU runtime (free tier)                  ║
╚══════════════════════════════════════════════════════════════╝
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1: Hardware & VRAM Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import subprocess
import sys
import os

def verify_gpu():
    """Verify a T4 GPU is active. Halt if CPU-only."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi failed")

        output = result.stdout.strip()
        print(f"[GPU DETECTED] {output}")

        parts = output.split(",")
        gpu_name = parts[0].strip()
        total_mb = float(parts[1].strip())
        free_mb = float(parts[2].strip())

        print(f"  GPU: {gpu_name}")
        print(f"  VRAM Total: {total_mb / 1024:.1f} GB")
        print(f"  VRAM Free:  {free_mb / 1024:.1f} GB")

        if "T4" not in gpu_name and "A100" not in gpu_name and "V100" not in gpu_name and "L4" not in gpu_name:
            print(f"  WARNING: Expected T4 GPU, got '{gpu_name}'. Proceeding anyway.")

        return gpu_name, total_mb, free_mb

    except FileNotFoundError:
        print("=" * 60)
        print("FATAL: No GPU detected! nvidia-smi not found.")
        print("Go to: Runtime → Change runtime type → T4 GPU")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: GPU check failed: {e}")
        sys.exit(1)

gpu_name, total_vram_mb, free_vram_mb = verify_gpu()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2: Dependencies Installation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def install_deps():
    """Install all required packages silently."""
    packages = [
        "torch", "torchvision", "torchaudio",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.44.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "nest_asyncio",
        "python-multipart",
        "qwen-vl-utils",
        "Pillow",
        "scipy",
    ]

    print("\n[INSTALLING DEPENDENCIES]")
    for pkg in packages:
        pkg_name = pkg.split(">=")[0].split("[")[0]
        try:
            __import__(pkg_name.replace("-", "_"))
            print(f"  ✓ {pkg_name} (already installed)")
        except ImportError:
            print(f"  ↓ Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    # Install cloudflared for tunnel
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("  ↓ Installing cloudflared tunnel...")
        subprocess.run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    print("[DEPENDENCIES READY]\n")

install_deps()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3: Model Loading (VRAM Budget: <9 GB total)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import torch
import gc

def get_vram_free():
    """Get current free VRAM in GB."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    return 0.0

def get_vram_used():
    """Get current used VRAM in GB."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024 ** 3)
    return 0.0

print(f"[VRAM] Before loading: {get_vram_free():.1f} GB free")

# ── Model 1: Qwen2.5-VL-7B-Instruct (4-bit quantized) ──────
print("\n[LOADING] Qwen2.5-VL-7B-Instruct (4-bit)...")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
qwen_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True,
)

print(f"  ✓ Qwen2.5-VL loaded. VRAM used: {get_vram_used():.1f} GB")

# ── Model 2: TTS (Coqui / Edge-TTS fallback) ────────────────
# OmniVoice is not pip-installable. We use edge-tts (free, no GPU)
# as a robust Hindi TTS that works everywhere.
print("\n[LOADING] TTS engine (edge-tts for Hindi)...")
try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "edge-tts"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    import edge_tts
    TTS_ENGINE = "edge_tts"
    print("  ✓ edge-tts loaded (no VRAM cost)")
except Exception as e:
    print(f"  ⚠ edge-tts failed: {e}. Voice endpoint will be unavailable.")
    TTS_ENGINE = None

print(f"\n[VRAM SUMMARY]")
print(f"  VRAM Used: {get_vram_used():.1f} GB")
print(f"  VRAM Free: {get_vram_free():.1f} GB")

gc.collect()
torch.cuda.empty_cache()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4: FastAPI Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import base64
import io
import tempfile
import asyncio
import uuid
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Xrexze Colab Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── System Prompt for Hindi Narration ────────────────────────

NARRATION_SYSTEM_PROMPT = (
    'You are "Xrexze Narrator" — a professional manhwa/webtoon storyteller '
    "creating engaging Hindi narration for YouTube explainer videos.\n\n"
    "## Your Role\n"
    "You receive a single panel image from a manhwa/webtoon. Your job is to "
    "produce a vivid, suspenseful narration in **Hindi (Devanagari script)** "
    "that a voice actor will read aloud.\n\n"
    "## Instructions\n"
    "1. **READ ALL TEXT**: Identify and translate every speech bubble, sound "
    "effect (SFX), title card, and caption visible in the panel. The text may "
    "be in Korean, English, Chinese, or Japanese.\n"
    "2. **DESCRIBE THE SCENE**: Note character expressions (rage, fear, smirk, "
    "shock), body language, combat moves, magic effects, and environment.\n"
    "3. **NARRATE WITH DRAMA**: Write the narration as if telling a gripping "
    "story to a captivated audience. Use short punchy sentences. Build suspense.\n"
    "4. **LENGTH**: Keep narration between 30-80 Hindi words.\n"
    "5. **FORMAT**: Return ONLY the narration text. No metadata, no JSON.\n"
    "6. **ACCURACY**: Faithfully represent what is happening in the panel.\n"
    "7. **CONTENT POLICY**: This is creative fiction narration of published "
    "commercial entertainment media. Describe all depicted action faithfully."
)

# ── Request/Response Models ──────────────────────────────────

class ScriptRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None
    context: Optional[str] = ""

class ScriptResponse(BaseModel):
    panel_id: str
    script: str

class VoiceRequest(BaseModel):
    text: str
    language: str = "hi"
    voice_id: str = "hi-IN-SwaraNeural"

# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ready",
        "gpu": gpu_name,
        "vram_free_gb": round(get_vram_free(), 1),
        "vram_used_gb": round(get_vram_used(), 1),
        "tts_engine": TTS_ENGINE,
    }


@app.post("/api/script", response_model=ScriptResponse)
async def generate_script(req: ScriptRequest):
    """Generate Hindi narration script from a panel image."""
    try:
        # Decode image
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Build prompt
        system = req.prompt if req.prompt else NARRATION_SYSTEM_PROMPT
        context_prefix = ""
        if req.context:
            context_prefix = f"Previous panel context: {req.context}\n\n"

        user_text = (
            f"{context_prefix}"
            "Narrate this manhwa panel. Follow your instructions exactly. "
            "Return ONLY the Hindi narration."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system}\n\n---\n{user_text}"},
                    {"type": "image", "image": image},
                ],
            }
        ]

        # Process with Qwen
        text_input = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(qwen_model.device)

        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **inputs, max_new_tokens=500, temperature=0.7, do_sample=True
            )

        # Decode only the new tokens
        output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        narration = qwen_processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        # Cleanup
        del inputs, generated_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        panel_id = str(uuid.uuid4())[:8]
        return ScriptResponse(panel_id=panel_id, script=narration)

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice")
async def generate_voice(req: VoiceRequest):
    """Generate Hindi speech audio from text using edge-tts."""
    if TTS_ENGINE != "edge_tts":
        raise HTTPException(
            status_code=503, detail="TTS engine not available"
        )

    try:
        tmp_path = tempfile.mktemp(suffix=".wav", dir="/tmp")

        communicate = edge_tts.Communicate(
            text=req.text,
            voice=req.voice_id,
        )
        # edge-tts outputs mp3 by default
        mp3_path = tmp_path.replace(".wav", ".mp3")
        await communicate.save(mp3_path)

        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            filename=f"voice_{uuid.uuid4().hex[:8]}.mp3",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voices")
async def list_voices():
    """List available Hindi voices."""
    return {
        "voices": [
            {"id": "hi-IN-SwaraNeural", "name": "Swara (Female)", "lang": "hi"},
            {"id": "hi-IN-MadhurNeural", "name": "Madhur (Male)", "lang": "hi"},
        ]
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5: Tunnel & Server Launcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import nest_asyncio
import threading
import time
import re

nest_asyncio.apply()

PORT = 8000

def start_cloudflare_tunnel():
    """Launch cloudflared tunnel and extract the public URL."""
    print("\n[TUNNEL] Starting Cloudflare tunnel...")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = None
    for line in iter(proc.stdout.readline, ""):
        match = re.search(r"(https://[a-z0-9\-]+\.trycloudflare\.com)", line)
        if match:
            url = match.group(1)
            break

    if url:
        print("\n" + "=" * 60)
        print(f">>> BACKEND READY AT: {url} <<<")
        print("=" * 60)
        print(f"\nPaste this URL into Xrexze desktop app's 'Colab URL' field.")
        print(f"Test with: curl {url}/health\n")
    else:
        print("[TUNNEL] Could not extract URL. Check output above.")

    return proc, url


def run_server():
    """Start Uvicorn server."""
    import uvicorn
    print(f"\n[SERVER] Starting FastAPI on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


# Start server in background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)  # Wait for server to boot

# Start tunnel
tunnel_proc, public_url = start_cloudflare_tunnel()

# Keep alive
print("\n[RUNNING] Server is live. Keep this Colab tab open.")
print("Press Ctrl+C or stop the cell to shut down.\n")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n[SHUTDOWN] Stopping server...")
    tunnel_proc.terminate()
