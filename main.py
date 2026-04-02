"""
Text-to-Image Fast Inference Backend
Two-pass FLUX Schnell via OpenRouter API:
  Pass 1: 64x64 preview  (~200ms network)
  Pass 2: 512x512 final  (~1-2s network)
No local GPU required.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OPENROUTER_URL = "https://openrouter.ai/api/v1/images/generations"
FLUX_MODEL = "black-forest-labs/flux-schnell"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MOCK_MODE:
        logger.info("MOCK_MODE=1 — returning fake colored images")
    elif not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set — set it in .env or environment")
    else:
        logger.info(f"Using OpenRouter FLUX model: {FLUX_MODEL}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


async def _call_openrouter(prompt: str, width: int, height: int, steps: int) -> Image.Image:
    """Call OpenRouter image generation API and return a PIL Image."""
    payload = {
        "model": FLUX_MODEL,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "n": 1,
        "response_format": "b64_json",
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "Text-to-Image Fast Inference",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        b64 = data["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


async def _run_inference(prompt: str, height: int, width: int, steps: int) -> Image.Image:
    if MOCK_MODE or not OPENROUTER_API_KEY:
        import random
        r, g, b = random.randint(80, 200), random.randint(80, 200), random.randint(80, 200)
        img = Image.new("RGB", (width, height), color=(r, g, b))
        await asyncio.sleep(0.15 if steps <= 2 else 1.0)
        return img

    return await _call_openrouter(prompt, width, height, steps)


def _encode_webp(img: Image.Image, quality: int = 80) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


async def _stream(prompt: str, request: Request) -> AsyncGenerator[str, None]:
    timings: dict = {}

    try:
        # ── Pass 1: blurry preview (64×64, 2 steps) ──────────────────────────
        t0 = time.time()
        preview_img = await _run_inference(prompt, 64, 64, 2)
        timings["preview_inference_ms"] = int((time.time() - t0) * 1000)

        if await request.is_disconnected():
            logger.info("Client disconnected before preview send")
            return

        t1 = time.time()
        preview_b64 = _encode_webp(preview_img, quality=60)
        timings["preview_encode_ms"] = int((time.time() - t1) * 1000)

        yield f"data: {json.dumps({'type': 'preview', 'img': preview_b64, 'timings': {**timings}})}\n\n"

        # ── Pass 2: final image (512×512, 8 steps) ───────────────────────────
        t2 = time.time()
        final_img = await _run_inference(prompt, 512, 512, 8)
        timings["final_inference_ms"] = int((time.time() - t2) * 1000)

        if await request.is_disconnected():
            logger.info("Client disconnected before final send")
            return

        t3 = time.time()
        final_b64 = _encode_webp(final_img, quality=85)
        timings["final_encode_ms"] = int((time.time() - t3) * 1000)
        timings["total_ms"] = int((time.time() - t0) * 1000)

        logger.info(f"Timings: {timings}")
        yield f"data: {json.dumps({'type': 'final', 'img': final_b64, 'timings': timings})}\n\n"
        yield 'data: {"type":"done"}\n\n'

    except asyncio.CancelledError:
        logger.info("Inference cancelled (client aborted)")
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.get("/generate")
async def generate(prompt: str, request: Request) -> StreamingResponse:
    return StreamingResponse(
        _stream(prompt, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "mock_mode": MOCK_MODE,
    }


# Serve frontend in production
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
