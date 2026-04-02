cd "text-to-image-fast-inference"

# UI dev (no GPU needed)
make install && make run-mock

# With real GPU
make run
Open http://localhost:8000 — the backend serves the frontend automatically.
