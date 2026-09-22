"""The FastAPI backend: it serves the API and the web page."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

VERSION = "0.1.0"

# The frontend folder sits next to the backend folder.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Emergency Detection API", version=VERSION)


@app.get("/api/health")
def health():
    """Lets the frontend (and monitoring tools) check that the backend is running."""
    return {"status": "ok", "version": VERSION}


# This line must come AFTER all the API routes. It serves the frontend files
# (index.html, style.css, app.js) for every URL that is not an API route.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")