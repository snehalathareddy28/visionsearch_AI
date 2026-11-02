VISIONSEARCH AI — FASTAPI WEB UI
===============================

Run these commands from the project root (the folder containing app.py,
models/, services/, index/, and visionsearch_web_ui/):

1. Activate the existing virtual environment:
   .\venv\Scripts\Activate.ps1

2. Install the project and web API dependencies:
   python -m pip install -r requirements.txt
   python -m pip install -r visionsearch_web_ui\requirements-ui.txt

3. Ensure the existing index files are present:
   index\faiss.index
   index\image_paths.txt
   index\image_embeddings.npy

4. Start the website and API:
   python -m uvicorn visionsearch_web_ui.api:app --reload

5. Open http://127.0.0.1:8000

Use the Text Search and Image Search tabs as before. Choose Upload Images to
add JPEG, PNG, or WebP images individually, as a multiple-file selection, or
from a folder (including nested folders in browsers that support
webkitdirectory). The upload drop area also accepts image drag-and-drop.

Search result cards provide Share, Download image, and native device sharing.
Native sharing sends the image file when the browser supports file sharing and
otherwise shares a link/text when available. WhatsApp, Telegram, Email, and
Copy image link require a public share URL. Local copied links point to the
local server and are not accessible from another person's device.

Public image links are disabled by default. Only enable them after the
deployment owner has approved the entire indexed image collection, including
uploads, for public access. In PowerShell, configure these variables before
starting the API:

   $env:VISIONSEARCH_PUBLIC_SHARING_ENABLED = "true"
   $env:VISIONSEARCH_PUBLIC_BASE_URL = "https://your-public-host"

Set these variables in the Streamlit process too if it is run separately.
The browser UI uses its current public origin. The Streamlit gallery uses
VISIONSEARCH_PUBLIC_BASE_URL for social links and
VISIONSEARCH_API_BASE_URL for fetching local image files; the latter defaults
to http://127.0.0.1:8000. Do not configure a public base URL unless it is a
real reachable deployment origin. The share endpoint only resolves opaque IDs
for supported images in the active index and does not expose filesystem paths.
Do not host private indexed images on an unauthenticated public deployment:
the existing search application itself has no user authentication.

Uploads are stored under uploads/ and appended to the existing CLIP/FAISS
collection. The default limits are 10 MB per image, 25 MB total image data,
and 20 files per request. The browser sends folders sequentially in batches
that respect these limits. Configure them with VISIONSEARCH_MAX_UPLOAD_FILE_MB,
VISIONSEARCH_MAX_UPLOAD_REQUEST_MB (integer megabytes), and
VISIONSEARCH_MAX_UPLOAD_FILES_PER_REQUEST.

The upload summary reports successful, duplicate, and failed files separately.
SHA-256 content checks skip duplicates already present in the existing index,
even when the indexed filename differs. Failed files can be retried without
re-uploading successful or duplicate files.
