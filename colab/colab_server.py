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
        "huggingface_hub",
        "soundfile",
        "omnivoice",
        "sentence-transformers",
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
# CELL 3: Global State & FastAPI Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import torch
import gc
import base64
import io
import tempfile
import asyncio
import uuid
import threading
import time
import re
import shutil
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_script = logging.getLogger("SCRIPT_GEN")
log_voice = logging.getLogger("VOICE_GEN")
log_memory = logging.getLogger("MEMORY")
log_loader = logging.getLogger("LOADER")
log_vram = logging.getLogger("VRAM")

# Global State
SYSTEM_STATE = {
    "status": "loading",
    "progress": "Initializing FastAPI server and Tunnel..."
}

# Globals for models
qwen_model = None
qwen_processor = None
TTS_ENGINE = None
omnivoice_model = None
embed_model = None

def get_vram_free():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    return 0.0

def get_vram_used():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024 ** 3)
    return 0.0


app = FastAPI(title="Xrexze Colab Backend", version="2.0.0")

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
    "commercial entertainment media. Describe all depicted action faithfully.\n\n"
    "## Entity Extraction\n"
    "After the narration, output exactly one JSON block:\n"
    '<<<JSON>>>{"characters": [{"name": "...", "status": "...", "location": "...", "skills": ["..."]}]}<<<END>>>'
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

class EmbedRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str

# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": SYSTEM_STATE["status"],
        "progress": SYSTEM_STATE["progress"],
        "gpu": gpu_name,
        "vram_free_gb": round(get_vram_free(), 1),
        "vram_used_gb": round(get_vram_used(), 1),
        "tts_engine": TTS_ENGINE if TTS_ENGINE else "None",
    }


@app.post("/api/script", response_model=ScriptResponse)
async def generate_script(req: ScriptRequest):
    """Generate Hindi narration script from a panel image."""
    if SYSTEM_STATE["status"] != "ready":
        raise HTTPException(
            status_code=503, 
            detail=f"Model is still loading. Current progress: {SYSTEM_STATE['progress']}"
        )
    log_script.info(f"Processing request... VRAM Free: {get_vram_free():.1f}GB")
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
    """Generate speech audio from text using OmniVoice (GPU)."""
    if SYSTEM_STATE["status"] != "ready":
        raise HTTPException(
            status_code=503, 
            detail=f"Model is still loading. Current progress: {SYSTEM_STATE['progress']}"
        )
        
    if TTS_ENGINE != "omnivoice" or omnivoice_model is None:
        raise HTTPException(
            status_code=503, detail="TTS engine not available"
        )
    log_voice.info(f"Processing request... VRAM Free: {get_vram_free():.1f}GB")
    try:
        import soundfile as sf
        
        # Generate audio tensor
        audio = omnivoice_model.generate(
            text=req.text
        )
        
        tmp_path = tempfile.mktemp(suffix=".wav", dir="/tmp")
        # Save audio tensor to .wav file at 24kHz
        sf.write(tmp_path, audio[0], 24000)

        return FileResponse(
            tmp_path,
            media_type="audio/wav",
            filename=f"voice_{uuid.uuid4().hex[:8]}.wav",
        )

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/embed")
async def embed_text(req: EmbedRequest):
    if embed_model is None:
        raise HTTPException(503, "Embedding model not loaded yet")
    log_memory.info(f"/api/embed called. Encoding {len(req.text)} chars. VRAM Free: {get_vram_free():.1f}GB")
    embedding = embed_model.encode(req.text, normalize_embeddings=True).tolist()
    return {"embedding": embedding}

SUMMARIZE_PROMPT = "You are a lore archivist. Compress the following story chapters into one dense paragraph preserving all key facts, character states, and plot developments. Output ONLY the summary paragraph."

@app.post("/api/summarize")
async def summarize_text(req: SummarizeRequest):
    if qwen_model is None:
        raise HTTPException(503, "Qwen model not loaded yet")
    log_memory.info(f"/api/summarize called. Input: {len(req.text)} chars. VRAM Free: {get_vram_free():.1f}GB")
    messages = [{"role": "user", "content": [{"type": "text", "text": f"{SUMMARIZE_PROMPT}\n\n{req.text}"}]}]
    text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text_input], padding=True, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=300, temperature=0.3)
    output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    summary = qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    del inputs, generated_ids, output_ids
    gc.collect(); torch.cuda.empty_cache()
    return {"summary": summary}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4: Tunnel & Server Launcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import nest_asyncio

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


# 1. Start server in background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)  # Wait for server to boot

# 2. Start tunnel
tunnel_proc, public_url = start_cloudflare_tunnel()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5: Background Model Loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def background_model_loader():
    global qwen_model, qwen_processor, TTS_ENGINE, omnivoice_model
    
    # Wait a bit for the tunnel to stabilize and user to copy URL
    time.sleep(5)
    
    try:
        # 1. Mount Google Drive
        SYSTEM_STATE["progress"] = "Waiting for Google Drive authorization (Check Colab tab)..."
        print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print(f"[LOADER] Failed to mount drive: {e}. Will use ephemeral local storage.")
            SYSTEM_STATE["progress"] = f"Warning: Drive mount failed. Using ephemeral storage."
            time.sleep(3)
        
        # 2. Check Cache
        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        DRIVE_CACHE_DIR = Path("/content/drive/MyDrive/Xrexze_load/AI_Models/Qwen2.5-VL-7B-Instruct")
        LOCAL_TMP_DIR = Path("/content/tmp_model")
        
        model_path = MODEL_ID
        
        if DRIVE_CACHE_DIR.exists() and (DRIVE_CACHE_DIR / "config.json").exists():
            SYSTEM_STATE["progress"] = "Loading model from Google Drive cache to GPU..."
            print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
            model_path = str(DRIVE_CACHE_DIR)
        else:
            SYSTEM_STATE["progress"] = "Downloading 15GB model from Hugging Face (first-time only)..."
            print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
            
            try:
                from huggingface_hub import snapshot_download
                
                # Download to ephemeral temp dir first
                snapshot_download(
                    repo_id=MODEL_ID,
                    local_dir=str(LOCAL_TMP_DIR),
                    local_dir_use_symlinks=False
                )
                
                # If drive is mounted, copy it over safely
                if Path("/content/drive").exists():
                    SYSTEM_STATE["progress"] = "Saving model to Google Drive for future fast boots..."
                    print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
                    
                    DRIVE_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(str(LOCAL_TMP_DIR), str(DRIVE_CACHE_DIR))
                    model_path = str(DRIVE_CACHE_DIR)
                    print(f"[LOADER] Successfully cached model to Drive: {model_path}")
                else:
                    model_path = str(LOCAL_TMP_DIR)
                    
            except Exception as e:
                print(f"[LOADER] Download/Cache failed: {e}. Falling back to default loader.")
                model_path = MODEL_ID

        # 3. Load Model
        SYSTEM_STATE["progress"] = f"Loading {model_path} into VRAM (4-bit)..."
        print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        qwen_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        print(f"  ✓ Qwen2.5-VL loaded. VRAM used: {get_vram_used():.1f} GB")
        
        # 4. Load TTS (OmniVoice)
        OMNI_MODEL_ID = "k2-fsa/OmniVoice"
        DRIVE_CACHE_DIR_OMNI = Path("/content/drive/MyDrive/Xrexze_load/AI_Models/OmniVoice")
        LOCAL_TMP_DIR_OMNI = Path("/content/tmp_omnivoice")
        
        model_path_omni = OMNI_MODEL_ID
        
        if DRIVE_CACHE_DIR_OMNI.exists():
            SYSTEM_STATE["progress"] = "Loading OmniVoice from Google Drive cache to GPU..."
            print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
            model_path_omni = str(DRIVE_CACHE_DIR_OMNI)
        else:
            SYSTEM_STATE["progress"] = "Downloading 3.3GB OmniVoice model to Drive..."
            print(f"\n[LOADER] {SYSTEM_STATE['progress']}")
            
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=OMNI_MODEL_ID,
                    local_dir=str(LOCAL_TMP_DIR_OMNI),
                    local_dir_use_symlinks=False
                )
                
                if Path("/content/drive").exists():
                    DRIVE_CACHE_DIR_OMNI.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(str(LOCAL_TMP_DIR_OMNI), str(DRIVE_CACHE_DIR_OMNI))
                    model_path_omni = str(DRIVE_CACHE_DIR_OMNI)
                    print(f"[LOADER] Successfully cached OmniVoice to Drive: {model_path_omni}")
                else:
                    model_path_omni = str(LOCAL_TMP_DIR_OMNI)
            except Exception as e:
                print(f"[LOADER] OmniVoice Download/Cache failed: {e}. Falling back to default loader.")
                model_path_omni = OMNI_MODEL_ID

        try:
            from omnivoice import OmniVoice
            
            omnivoice_model = OmniVoice.from_pretrained(
                model_path_omni, 
                device_map="cuda:0",
                dtype=torch.float16
            )
            
            TTS_ENGINE = "omnivoice"
            log_loader.info("  ✓ OmniVoice loaded")
        except Exception as e:
            log_loader.error(f"  ⚠ OmniVoice failed: {e}. Voice endpoint will be unavailable.")
            TTS_ENGINE = None

        # 5. Load Embedding Model (BAAI/bge-m3)
        BGE_MODEL_ID = "BAAI/bge-m3"
        DRIVE_CACHE_DIR_BGE = Path("/content/drive/MyDrive/Xrexze_load/AI_Models/bge-m3")
        LOCAL_TMP_DIR_BGE = Path("/content/tmp_bge_m3")
        
        model_path_bge = BGE_MODEL_ID
        if DRIVE_CACHE_DIR_BGE.exists():
            log_loader.info("Loading BGE-M3 from Google Drive cache...")
            model_path_bge = str(DRIVE_CACHE_DIR_BGE)
        else:
            log_loader.info("Downloading 2.2GB BGE-M3 model...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=BGE_MODEL_ID, local_dir=str(LOCAL_TMP_DIR_BGE), local_dir_use_symlinks=False)
            if Path("/content/drive").exists():
                DRIVE_CACHE_DIR_BGE.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(LOCAL_TMP_DIR_BGE), str(DRIVE_CACHE_DIR_BGE))
                model_path_bge = str(DRIVE_CACHE_DIR_BGE)
            else:
                model_path_bge = str(LOCAL_TMP_DIR_BGE)
        
        from sentence_transformers import SentenceTransformer
        global embed_model
        embed_model = SentenceTransformer(model_path_bge, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        log_loader.info(f"  ✓ BGE-M3 loaded.")

        log_vram.info(f"\n[VRAM SUMMARY]")
        log_vram.info(f"  VRAM Used: {get_vram_used():.1f} GB")
        log_vram.info(f"  VRAM Free: {get_vram_free():.1f} GB")

        gc.collect()
        torch.cuda.empty_cache()
        
        # 5. Ready
        SYSTEM_STATE["status"] = "ready"
        SYSTEM_STATE["progress"] = "Online"
        print("\n[LOADER] Model initialization complete! System is ready.")

    except Exception as e:
        SYSTEM_STATE["status"] = "error"
        SYSTEM_STATE["progress"] = f"Failed to load models: {str(e)}"
        print(f"\n[LOADER] FATAL ERROR: {e}")

# 3. Start background model loader
loader_thread = threading.Thread(target=background_model_loader, daemon=True)
loader_thread.start()

# Keep alive
print("\n[RUNNING] Server is live. Keep this Colab tab open.")
print("Press Ctrl+C or stop the cell to shut down.\n")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n[SHUTDOWN] Stopping server...")
    tunnel_proc.terminate()
