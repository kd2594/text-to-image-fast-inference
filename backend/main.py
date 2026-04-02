"""
Text-to-Image Fast Inference Backend
Two-pass FLUX Schnell: 64x64 preview (~200ms) then 512x512 final (~1.5s)
Model stays warm in GPU memory between requests.
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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"

pipe = None
_inference_lock = asyncio.Lock()


def _load_model():
    import torch
    from diffusers import FluxPipeline

    logger.info("Loading FLUX.1-schnell...")
    t0 = time.time()
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        logger.warning("No GPU found — CPU inference will be slow")
    logger.info(f"Model ready in {time.time() - t0:.1f}s")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    if MOCK_MODE:
        logger.info("MOCK_MODE=1 — skipping model load, returning fake images")
    else:
        try:
            loop = asyncio.get_event_loop()
            pipe = await loop.run_in_executor(None, _load_model)
        except Exception as e:
            logger.error(f"Model load failed: {e}. Set MOCK_MODE=1 to run without GPU.")
    yield
    if pipe is not None:
        del pipe


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _run_inference(prompt: str, height: int, width: int, steps: int) -> Image.Image:
    if MOCK_MODE or pipe is None:
        # Return a colored gradient as a fake image for development
        import random
        r, g, b = random.randint(80, 200), random.randint(80, 200), random.randint(80, 200)
        img = Image.new("RGB", (width, height), color=(r, g, b))
        time.sleep(0.15 if steps <= 2 else 1.0)  # simulate latency
        return img

    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,  # Schnell is distilled — no CFG needed
    )
    return result.images[0]


def _encode_webp(img: Image.Image, quality: int = 80) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


async def _stream(prompt: str, request: Request) -> AsyncGenerator[str, None]:
    timings: dict = {}
    loop = asyncio.get_event_loop()

    try:
        # ── Pass 1: blurry preview (64×64, 2 steps) ──────────────────────────
        t0 = time.time()
        preview_img = await loop.run_in_executor(
            None, _run_inference, prompt, 64, 64, 2
        )
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
        final_img = await loop.run_in_executor(
            None, _run_inference, prompt, 512, 512, 8
        )
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
