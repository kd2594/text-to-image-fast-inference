Inspired by https://modal.com/solutions/image-and-video 

cd "text-to-image-fast-inference"

**Steps**
1. Get your OpenRouter API key
Go to https://openrouter.ai/keys → create a key → make sure you have credits (FLUX Schnell is ~$0.003/image)
2. Put the key in .env

# text-to-image-fast-inference/backend/.env
```OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx```

3. Reinstall (deps changed) and run
```
cd "/Users/krishnadamarla/Library/CloudStorage/GoogleDrive-krishna.damarla@flex.ai/My Drive/GitHub Code/text-to-image-fast-inference/backend"
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
Open http://localhost:8080 → type a prompt → you'll see a real FLUX-generated image.
```

---

Commit 1

# UI dev (no GPU needed)
make install && make run-mock

# With real GPU
- make run
- Open http://localhost:8000 / 8080 — the backend serves the frontend automatically.

<img width="838" height="712" alt="image" src="https://github.com/user-attachments/assets/b5a833b2-2463-4ed9-a060-59434588fa34" />

