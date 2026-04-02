.PHONY: install run run-mock

install:
	cd backend && uv sync

# Run with real FLUX model (requires GPU)
run:
	cd backend && uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run in mock mode (no GPU needed — great for UI dev)
run-mock:
	cd backend && MOCK_MODE=1 uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
