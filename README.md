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

Place source images under `dataset/images/` and select **My Images** in the
Dataset Management section to build the CLIP/FAISS artifacts.

To use Caltech-101, place its `101_ObjectCategories.tar.gz` archive at
`caltech-101/101_ObjectCategories.tar.gz`. Select **Caltech-101** in Dataset
Management and build the index; the app extracts supported images into
`dataset/caltech-101/` before indexing them.

You can also build the default `dataset/images/` index from the command line:

```bash
python embeddings/image_embeddings.py
```

Run a text query against the resulting index with:

```bash
python test_search.py
```

The generated files are `index/image_embeddings.npy`, `index/image_paths.txt`, and `index/faiss.index`.
