# VisionSearch AI

A Streamlit + FastAPI image search application supporting text-to-image, image-to-image, and region-based (object-level) retrieval.

## Project layout

- `app.py`: Streamlit landing page
- `pages/`: Search and evaluation views
- `services/`: CLIP embedding, indexing, search, ranking, explanation, and history logic
- `models/`: Model loading and inference helpers
- `visionsearch_web_ui/`: FastAPI server + HTML/JS/CSS frontend
- `visionsearch_web_ui/region/`: **New** — region-based retrieval pipeline
- `dataset/images/`: Images grouped by category
- `data/`: Metadata and persisted search history
- `index/`: Generated FAISS index files

## Run locally (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Web UI (FastAPI)

```bash
python -m pip install -r requirements.txt
python -m uvicorn visionsearch_web_ui.api:app --reload
```

Open `http://127.0.0.1:8000`.

## Sharing search results

The result cards in both frontends include image download and sharing
controls. For Streamlit's native file sharing and local image links, run the
FastAPI server in a second terminal as shown above. Localhost links work only
on the machine running the server.

External image links are disabled by default. Enable them only after the
deployment owner has approved the complete indexed collection, including
uploads, for public access. For a deployed FastAPI host, set these variables
before starting it. If Streamlit runs as a separate process, set the same
public-sharing variables there as well:

```powershell
$env:VISIONSEARCH_PUBLIC_SHARING_ENABLED = "true"
$env:VISIONSEARCH_PUBLIC_BASE_URL = "https://your-public-api-host"
```

For Streamlit, `VISIONSEARCH_API_BASE_URL` selects the API used to fetch image
files and defaults to `http://127.0.0.1:8000`. Configure it to the real reachable
API host when the Streamlit app is not local. Never use a localhost URL as a
public share link. This project does not authenticate users; do not expose
private indexed images through a publicly reachable deployment.

The default command-line build indexes all supported images under
`dataset/caltech-101-selected/`. This includes any additional images you put in
that folder. It replaces the existing
`index/image_embeddings.npy`, `index/image_paths.txt`, and `index/faiss.index`
artifacts.

You can build the default `dataset/caltech-101-selected/` index from the command line:

```bash
python embeddings/image_embeddings.py
```

Run a text query against the resulting index with:

```bash
python test_search.py
```

The generated files are `index/image_embeddings.npy`, `index/image_paths.txt`, and `index/faiss.index`.

## Web UI image uploads

The FastAPI web UI supports individual images, multiple image selections, and
folder selection (including nested folders in browsers that support
`webkitdirectory`). Images can also be dragged into the upload area. Uploads
accept JPEG, PNG, and WebP image content; the server verifies and decodes each
image rather than trusting its filename or browser MIME type.

Override upload limits before launch with
`VISIONSEARCH_MAX_UPLOAD_FILE_MB`, `VISIONSEARCH_MAX_UPLOAD_REQUEST_MB`
(integer megabytes), and `VISIONSEARCH_MAX_UPLOAD_FILES_PER_REQUEST`.

Run the automated tests from the project root:

```bash
.venv\Scripts\pytest tests -v
```

---

## Region-Based (Object-Level) Image Retrieval

### How it works

Standard CLIP embeds the **entire image** as one 512-dimensional vector. If a
query asks for "a small dog in the background", the relevant pixels may
contribute only a small fraction of the global embedding and get drowned out by
the rest of the scene.

Region-based retrieval solves this by:

1. Running **YOLOv8n** on each indexed image to detect objects; in `auto`
   mode, images with no valid detections are stored as labeled grid tiles.
2. Padding each detected bbox by 5 % and cropping it from the original image.
3. Embedding the **whole image** *plus* every crop with CLIP (all L2-normalised).
4. Storing each vector in a **separate FAISS index** (`index/region_faiss.index`)
   with a **SQLite metadata database** (`index/region_metadata.db`) that maps
   every FAISS row to parent image path + bbox coordinates + class label.
5. At search time, grouping all region hits by parent image and returning the
   **best-scoring region per image** together with its bounding box.

The selected split mode also constrains region-search candidates: `yolo`
searches YOLO crops, `grid` searches grid tiles, and `auto` searches both.
Selected thumbnails submit their original bounding box so the backend embeds
the exact crop shown in the UI rather than rerunning detection for its ID.

Category-aware ranking is optional and disabled by default, preserving the
original CLIP ranking. To experiment with soft category compatibility, set
`VISIONSEARCH_REGION_CATEGORY_AWARE_ENABLED=1`. Query categories are resolved
only when an unambiguous object mention matches labels present in the YOLO
metadata; unsupported or ambiguous queries remain semantic-only. The
`VISIONSEARCH_REGION_CATEGORY_BONUS`,
`VISIONSEARCH_REGION_CATEGORY_PENALTY`, and
`VISIONSEARCH_REGION_CATEGORY_MIN_CONFIDENCE` settings control the adjustment
and minimum YOLO confidence. Category labels never filter candidates.

### Setup (one time)

```bash
pip install -r requirements.txt
# YOLOv8 weights download automatically on first indexing run (needs internet once)
```

### Build the region index

```bash
# YOLO + grid fallback (recommended)
python -m visionsearch_web_ui.region.index_manager --image-dir dataset/caltech-101-selected

# Grid-only (no YOLO, faster, less precise)
python -m visionsearch_web_ui.region.index_manager --image-dir dataset/caltech-101-selected --split-mode grid

# Force full rebuild from scratch
python -m visionsearch_web_ui.region.index_manager --image-dir dataset/caltech-101-selected --force
```

Expected output (CPU, ~1 006 images, ~12 min):
```
Building region index: 100%|████| 1006/1006 [12:30<00:00]
Region index built.
  Indexed: 1006 images
  Skipped: 0
  Total FAISS vectors: ~7000-9000
```

### API endpoints (Swagger at `/docs`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/region/index/build` | Build / rebuild region index |
| `POST` | `/api/region/index/add` | Add images incrementally |
| `GET`  | `/api/region/index/stats` | FAISS + SQLite statistics |
| `POST` | `/api/region/search/text` | Text-to-region search |
| `POST` | `/api/region/search/image/detect` | Detect crops in uploaded image |
| `POST` | `/api/region/search/image` | Image / crop / bbox search |
| `GET`  | `/api/region/crop` | Serve a crop JPEG on-the-fly |

### Run the evaluation

```bash
python evaluate_regions.py \
    --dataset-dir dataset/caltech-101-selected \
    --top-k 10 \
    --output-dir results
```

Outputs: `results/region_eval_metrics.csv` and `results/region_eval_chart.png`.

### Run the tests

```bash
# All tests
.venv\Scripts\pytest tests -v

# Region-specific tests only
.venv\Scripts\pytest tests/test_region_splitter.py tests/test_metadata_db.py tests/test_region_search.py -v
```

### Configuration reference

All knobs live in `visionsearch_web_ui/region/config.py` and can be overridden
with environment variables prefixed `VISIONSEARCH_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Ultralytics model weights |
| `YOLO_CONF_THRESHOLD` | `0.35` | Min detection confidence |
| `YOLO_MAX_DETECTIONS` | `8` | Max crops per image |
| `YOLO_MIN_CROP_PX` | `64` | Min crop width and height |
| `YOLO_PAD_FRACTION` | `0.05` | Bbox padding fraction |
| `GRID_ROWS` / `GRID_COLS` | `3` / `3` | Grid fallback dimensions |
| `GRID_OVERLAP` | `0.20` | Tile overlap fraction |
| `EMBED_BATCH_SIZE` | `8` | CLIP batch size |
| `DEFAULT_SEARCH_SCOPE` | `both` | `whole` / `regions` / `both` |
| `WHOLE_IMAGE_WEIGHT` | `0.5` | Weight for whole-image score in "both" mode |
| `CACHE_CROPS` | `0` | Set to `1` to save crop images to disk |

---

## Viva Notes

### Why does region-based retrieval improve small-object queries?

When an object occupies < 10 % of image pixels, its contribution to the
global CLIP embedding is proportionally small. By embedding the isolated crop,
CLIP attends fully to the object's texture, shape, and colour, producing a
vector that aligns much better with queries like "a small butterfly".

### Limitations

| Limitation | Detail |
|------------|--------|
| **Index growth** | Each image generates up to 9 extra vectors. 1 000 images → ~8 000 vectors. |
| **Detector errors** | If YOLO misses an object, the grid fallback helps but is less precise. |
| **Duplicate regions** | Two overlapping YOLO boxes produce similar embeddings, wasting top-k slots. |
| **Indexing latency** | 8–10× slower than whole-image indexing on CPU. |

### 5 Likely Examiner Questions

**Q1. How does FAISS find the nearest region embedding?**
> `IndexFlatIP` computes exact inner product between the query and every stored
> vector. Because all vectors are L2-normalised, inner product = cosine
> similarity. For > 1 M vectors, switch to `IndexIVFPQ` for approximate search.

**Q2. Why SQLite and not a second numpy array?**
> A numpy array stores only numbers. We need structured metadata per vector:
> parent path, bbox coordinates, class label, crop type. SQLite is zero-
> configuration, transactional, and inspectable with any SQLite browser.

**Q3. What happens when YOLO detects nothing?**
> The system falls back to a 3 × 3 overlapping tile grid with 20 % overlap so
> objects near tile boundaries are not cut off.

**Q4. How does the "both" scope combine scores?**
> `combined = 0.5 × whole_score + 0.5 × best_region_score`. The weight is
> configurable via `VISIONSEARCH_WHOLE_IMAGE_WEIGHT`.

**Q5. Is the region index backward-compatible?**
> Yes. The region index lives in separate files (`index/region_faiss.index`,
> `index/region_metadata.db`). The existing Streamlit app and
> `/api/search/text` / `/api/search/image` endpoints are not modified at all.
