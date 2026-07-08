from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def mount_frontend(app: FastAPI) -> None:
    dist_dir = frontend_dist_dir()
    if not dist_dir.exists():
        return
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")
