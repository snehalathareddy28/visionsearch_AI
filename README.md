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
