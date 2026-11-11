"""
backend/main.py - Main entrypoint for FastAPI application.
Exposes `app` for Vercel and ASGI runners.
"""
import sys
from pathlib import Path

# Ensure project root and backend dir are in sys.path
_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.server import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
