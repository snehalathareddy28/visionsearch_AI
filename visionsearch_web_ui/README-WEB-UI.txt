VISIONSEARCH AI — CUSTOM WEB UI SETUP
=====================================

This is a starter frontend and FastAPI adapter intended to sit alongside your
existing VisionSearch AI project. It does not replace app.py or your backend files.

1. Copy api.py, frontend/, and requirements-ui.txt into the ROOT of your existing
   project (the same folder that contains app.py, models/, services/, index/, etc.).

2. Open PowerShell in the project root:
   cd C:\visionsearch-ai

3. Activate your existing environment:
   .\venv\Scripts\Activate.ps1

4. Install the small set of web API dependencies:
   python -m pip install -r requirements-ui.txt

5. Make sure the existing image index exists:
   index\faiss.index
   index\image_paths.txt
   index\image_embeddings.npy

   If it does not exist yet, you can temporarily run your existing Streamlit app
   only to build the index, or use your existing indexing script. The custom web UI
   itself does not use Streamlit.

6. Start the custom UI and API:
   python -m uvicorn api:app --reload

7. Open in Chrome:
   http://127.0.0.1:8000

IMPORTANT:
- Keep your existing app.py, models/, services/, utils/, dataset/, and index/ folders.
- If your search service returns results in a different format than (image_path, score),
  the API's normalize_results() in api.py may need a small adjustment.
- If you see an import error, copy the complete terminal traceback so the imports can
  be aligned with your actual project.
