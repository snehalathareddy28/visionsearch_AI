# VisionSearch AI

A Streamlit image search application supporting text-to-image, image-to-image, hybrid, and conversational search.

## Project layout

- `app.py`: Streamlit landing page
- `pages/`: Search and evaluation views
- `services/`: CLIP embedding, indexing, search, ranking, explanation, and history logic
- `models/`: Model loading and inference helpers
- `dataset/images/`: Images grouped by category
- `data/`: Metadata and persisted search history
- `indexes/`: Generated FAISS index files

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The default command-line build indexes all supported images under
`dataset/caltech-101-selected/`. This includes any additional images you put in
that folder, such as dog images. It replaces the existing
`index/image_embeddings.npy`, `index/image_paths.txt`, and `index/faiss.index`
artifacts.

In the app, select **Caltech-101 Selected** to index that folder, or select
**My Images** to index images under `dataset/images/`.
Text search also uses matching image-folder names as category labels, so an
explicit category such as “dog” is not outranked by unrelated images merely
because the query also describes an action or scene. Results show the category
alongside the image filename.

To use Caltech-101, place its `101_ObjectCategories.tar.gz` archive at
`caltech-101/101_ObjectCategories.tar.gz`. Select **Caltech-101** in Dataset
Management and build the index; the app extracts supported images into
`dataset/caltech-101/` before indexing them.

You can also build the default `dataset/caltech-101-selected/` index from the command line:

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

Install the web UI dependencies and launch it from the project root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r visionsearch_web_ui/requirements-ui.txt
python -m uvicorn visionsearch_web_ui.api:app --reload
```

Open `http://127.0.0.1:8000` and choose **Upload Images**. The default limits
are 10 MB per image, 25 MB total image data, and 20 images per request. The
website splits folder selections into sequential batches using the API's
advertised limits, reports added/duplicate/failed counts, and lets you retry
failed files. Override the limits before launch with
`VISIONSEARCH_MAX_UPLOAD_FILE_MB`, `VISIONSEARCH_MAX_UPLOAD_REQUEST_MB`
(integer megabytes), and `VISIONSEARCH_MAX_UPLOAD_FILES_PER_REQUEST`.

Uploaded images are stored under `uploads/`, with sanitized folder-relative
paths retained for folder uploads. The API adds normalized CLIP embeddings to
the existing FAISS collection and atomically updates
`index/faiss.index`, `index/image_paths.txt`, and `index/image_embeddings.npy`.
Each artifact is atomically replaced under a transaction journal; an interrupted
update is rolled back when the API next loads the index. Existing vectors remain
in the collection; identical image content is skipped using SHA-256 checksums,
including images already present in the legacy index.
Uploads are available to text and image searches immediately and remain
available after restart. The UI requires the existing index artifacts to be
present and consistent before it can add images.

Run the automated tests, including the upload API tests, from the project root:

```bash
python -m unittest discover -s tests -v
```
