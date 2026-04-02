cd "text-to-image-fast-inference"

# UI dev (no GPU needed)
make install && make run-mock

# With real GPU
make run
Open http://localhost:8000 — the backend serves the frontend automatically.

<img width="838" height="712" alt="image" src="https://github.com/user-attachments/assets/b5a833b2-2463-4ed9-a060-59434588fa34" />

