"""Re-export so ``uvicorn app:app`` continues to work."""

from src.serving.app import app

__all__ = ["app"]
