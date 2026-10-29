from pathlib import Path
from typing import Any
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from models.clip_model import load_clip_model
from services.search_service import load_index, search_loaded, search_text_loaded

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "index" / "faiss.index"
PATHS_PATH = ROOT / "index" / "image_paths.txt"
FRONTEND = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="VisionSearch AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_index_data = None

class TextQuery(BaseModel):
    query: str
    top_k: int = 10

def get_model():
    global _model
    if _model is None:
        _model = load_clip_model()
    return _model

def get_index():
    global _index_data
    if _index_data is None:
        if not INDEX_PATH.exists() or not PATHS_PATH.exists():
            raise HTTPException(
                status_code=400,
                detail="Image index is missing. Build it using your existing app first, then restart this API."
            )
        _index_data = load_index(INDEX_PATH, PATHS_PATH)
    return _index_data

def safe_result_path(value: Any) -> Path | None:
    # Existing search service commonly returns (path, score) pairs; support dicts too.
    if isinstance(value, dict):
        value = value.get("path") or value.get("image_path") or value.get("filepath") or value.get("file")
    elif isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not isinstance(value, (str, Path)):
        return None
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    try:
        p = p.resolve()
        p.relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None

def normalize_results(results):
    output = []
    for item in results:
        p = safe_result_path(item)
        if p:
            try:
                rel = p.relative_to(ROOT.resolve()).as_posix()
                score = None
                if isinstance(item, dict):
                    score = item.get("score") or item.get("similarity")
                elif isinstance(item, (tuple, list)) and len(item) > 1:
                    score = item[1]
                output.append({
                    "name": p.name,
                    "category": p.parent.name,
                    "image_url": "/image/" + rel,
                    "score": float(score) if score is not None else None,
                    "category_match": bool(
                        isinstance(item, dict) and item.get("category_match")
                    ),
                })
            except (TypeError, ValueError):
                output.append({"name": p.name, "image_url": "/image/" + rel, "score": None})
    return output

@app.get("/api/health")
def health():
    return {"status": "ok", "index_ready": INDEX_PATH.exists() and PATHS_PATH.exists()}

@app.post("/api/search/text")
def text_search(body: TextQuery):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a search description.")
    index, paths = get_index()
    model = get_model()
    embedding = model.encode_text(query)
    results = search_text_loaded(
        query,
        index,
        paths,
        embedding,
        top_k=max(1, min(body.top_k, 50)),
    )
    return {"query": query, "results": normalize_results(results)}

@app.post("/api/search/image")
async def image_search(file: UploadFile = File(...), top_k: int = 10):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file could not be opened as an image.")
    index, paths = get_index()
    model = get_model()
    embedding = model.encode_image(image)
    results = search_loaded(index, paths, embedding, top_k=max(1, min(top_k, 50)))
    return {"results": normalize_results(results)}

@app.get("/image/{image_path:path}")
def serve_image(image_path: str):
    p = (ROOT / image_path).resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid image path.")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(p)

app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
