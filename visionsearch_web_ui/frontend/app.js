const $ = (id) => document.getElementById(id);
let mode = "text";
const message = $("message");
const results = $("results");
let selectedUploads = [];
let previewUrls = [];
let uploading = false;
let uploadLimits = {
  maxFiles: 20,
  maxFileBytes: 10 * 1024 * 1024,
  maxRequestBytes: 25 * 1024 * 1024
};
const uploadOutcomes = new Map();
const previewLimit = 12;

function setMode(nextMode) {
  mode = nextMode;
  $("crumb-mode").textContent = nextMode === "text" ? "Text Search" : nextMode === "image" ? "Image Search" : "Upload Images";
  $("text-form").classList.toggle("hidden", nextMode !== "text");
  $("image-form").classList.toggle("hidden", nextMode !== "image");
  $("upload-form").classList.toggle("hidden", nextMode !== "upload");
  document.querySelectorAll("[data-mode]").forEach(button => {
    button.classList.toggle("active", button.classList.contains("nav-item") && button.dataset.mode === nextMode);
    button.classList.toggle("selected", button.classList.contains("mode-tab") && button.dataset.mode === nextMode);
  });
}
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.querySelectorAll(".example").forEach(button => button.addEventListener("click", () => { $("query").value = button.textContent; $("query").focus(); }));
$("image-file").addEventListener("change", () => {
  const file = $("image-file").files[0];
  $("file-name").textContent = file ? file.name : "No image selected";
});

function addUploads(fileList) {
  const known = new Set(selectedUploads);
  for (const file of fileList) {
    if (!known.has(file)) {
      selectedUploads.push(file);
      known.add(file);
    }
  }
  renderUploadSelection();
}

function renderUploadSelection() {
  for (const url of previewUrls) URL.revokeObjectURL(url);
  previewUrls = [];
  $("upload-count").textContent = `${selectedUploads.length} image${selectedUploads.length === 1 ? "" : "s"} selected`;
  $("upload-submit").disabled = uploading || selectedUploads.length === 0;
  const failedCount = selectedUploads.filter(file => uploadOutcomes.get(file)?.status === "failed").length;
  $("retry-uploads").classList.toggle("hidden", failedCount === 0);
  $("retry-uploads").disabled = uploading || failedCount === 0;
  $("upload-previews").innerHTML = "";

  for (const file of selectedUploads.slice(0, previewLimit)) {
    const preview = document.createElement("div");
    preview.className = "upload-preview";
    if (file.type.startsWith("image/")) {
      const image = document.createElement("img");
      const url = URL.createObjectURL(file);
      previewUrls.push(url);
      image.src = url;
      image.alt = "";
      preview.appendChild(image);
    }
    const details = document.createElement("div");
    details.className = "upload-preview-details";
    const name = document.createElement("strong");
    name.textContent = file.webkitRelativePath || file.name;
    const size = document.createElement("small");
    size.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    details.append(name, size);
    preview.appendChild(details);
    $("upload-previews").appendChild(preview);
  }
  if (selectedUploads.length > previewLimit) {
    const remaining = document.createElement("small");
    remaining.className = "upload-preview-more";
    remaining.textContent = `Showing ${previewLimit} of ${selectedUploads.length} selected images.`;
    $("upload-previews").appendChild(remaining);
  }
}

function setUploadStatus(text, error = false) {
  const status = $("upload-status");
  status.textContent = text;
  status.classList.toggle("error", error);
  status.classList.remove("hidden");
}

function setUploading(value) {
  uploading = value;
  $("upload-files").disabled = value;
  $("upload-folder").disabled = value;
  $("clear-uploads").disabled = value;
  $("upload-submit").textContent = value ? "Uploading and indexing…" : "Upload and index images ↗";
  $("upload-submit").disabled = value || selectedUploads.length === 0;
  $("retry-uploads").disabled = value || !selectedUploads.some(file => uploadOutcomes.get(file)?.status === "failed");
}

for (const input of [$("upload-files"), $("upload-folder")]) {
  input.addEventListener("change", () => {
    addUploads(input.files);
    input.value = "";
    $("upload-status").classList.add("hidden");
  });
}

const dropzone = $("upload-dropzone");
dropzone.addEventListener("click", event => {
  if (!event.target.closest("label") && !uploading) $("upload-files").click();
});
dropzone.addEventListener("keydown", event => {
  if ((event.key === "Enter" || event.key === " ") && !uploading) {
    event.preventDefault();
    $("upload-files").click();
  }
});
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", event => {
  if (!uploading) addUploads(event.dataTransfer.files);
});
$("clear-uploads").addEventListener("click", () => {
  if (uploading) return;
  selectedUploads = [];
  uploadOutcomes.clear();
  $("upload-status").classList.add("hidden");
  $("upload-progress").classList.add("hidden");
  renderUploadSelection();
});

function showMessage(text, error = false) {
  message.textContent = text;
  message.style.color = error ? "#ff9baf" : "";
  message.classList.remove("hidden");
}
function renderResults(items) {
  results.innerHTML = "";
  if (!items || !items.length) {
    showMessage("No image results were returned. Check that your image index contains supported images.");
    return;
  }
  message.classList.add("hidden");
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "image-card";
    const img = document.createElement("img");
    img.src = item.image_url;
    img.alt = item.name || "Search result";
    img.loading = "lazy";
    img.onerror = () => { img.style.display = "none"; };
    const meta = document.createElement("div");
    meta.className = "image-meta";
    const name = document.createElement("strong");
    name.textContent = item.category ? `${item.category} / ${item.name}` : item.name || "Image result";
    meta.appendChild(name);
    if (item.category_match) {
      const categoryMatch = document.createElement("span");
      categoryMatch.className = "score";
      categoryMatch.textContent = "Matched category";
      meta.appendChild(categoryMatch);
    }
    if (typeof item.score === "number" && Number.isFinite(item.score)) {
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = `Similarity: ${item.score.toFixed(3)}`;
      meta.appendChild(score);
    }
    card.append(img, meta);
    results.appendChild(card);
  }
}
async function runSearch(url, options, title) {
  showMessage("Searching your image library with CLIP + FAISS…");
  results.innerHTML = "";
  $("results-title").textContent = title;
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Search request failed.");
    renderResults(data.results);
  } catch (error) {
    showMessage(`${error.message} If this is an index error, build the image index first in your existing app, then restart the API.`, true);
  }
}
$("text-form").addEventListener("submit", event => {
  event.preventDefault();
  const query = $("query").value.trim();
  if (!query) { showMessage("Please enter a description first.", true); return; }
  runSearch("/api/search/text", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query, top_k: Number($("top-k").value)})
  }, `Results for “${query}”`);
});
$("image-form").addEventListener("submit", event => {
  event.preventDefault();
  const file = $("image-file").files[0];
  if (!file) { showMessage("Please choose an image first.", true); return; }
  const form = new FormData();
  form.append("file", file);
  runSearch(`/api/search/image?top_k=${$("top-k").value}`, {method:"POST", body:form}, "Similar images");
});

function splitUploadBatches(files) {
  const batches = [];
  const locallyRejected = [];
  let batch = [];
  let batchBytes = 0;
  const pushBatch = () => {
    if (batch.length) batches.push(batch);
    batch = [];
    batchBytes = 0;
  };
  for (const file of files) {
    if (file.size > uploadLimits.maxFileBytes) {
      locallyRejected.push({
        file,
        error: `File exceeds the ${(uploadLimits.maxFileBytes / (1024 * 1024)).toFixed(0)} MB per-file limit.`
      });
      continue;
    }
    if (file.size > uploadLimits.maxRequestBytes) {
      locallyRejected.push({
        file,
        error: `File exceeds the ${(uploadLimits.maxRequestBytes / (1024 * 1024)).toFixed(0)} MB batch limit.`
      });
      continue;
    }
    if (batch.length >= uploadLimits.maxFiles || batchBytes + file.size > uploadLimits.maxRequestBytes) {
      pushBatch();
    }
    batch.push(file);
    batchBytes += file.size;
  }
  pushBatch();
  return {batches, locallyRejected};
}

function requestUploadBatch(files) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const file of files) {
      form.append("files", file, file.name);
      form.append("relative_paths", file.webkitRelativePath || file.name);
    }
  const request = new XMLHttpRequest();
  request.open("POST", "/api/upload");
  request.addEventListener("load", () => {
    let data;
    try {
      data = JSON.parse(request.responseText);
    } catch {
      const error = new Error("The server returned an unreadable upload response.");
      error.status = request.status;
      reject(error);
      return;
    }
    if (request.status < 200 || request.status >= 300) {
      const error = new Error(data.detail || "The upload request failed.");
      error.status = request.status;
      reject(error);
      return;
    }
    resolve(data);
  });
  request.addEventListener("error", () => reject(new Error("The upload could not reach the server. Check your connection and try again.")));
  request.addEventListener("abort", () => reject(new Error("The upload was cancelled.")));
  request.send(form);
  });
}

function updateUploadProgress(total) {
  const processed = selectedUploads.filter(file => uploadOutcomes.has(file)).length;
  const percent = total ? Math.round((processed / total) * 100) : 0;
  $("upload-progress").setAttribute("aria-valuenow", String(percent));
  $("upload-progress-fill").style.width = `${percent}%`;
  return processed;
}

function renderUploadSummary(total) {
  const counts = {indexed: 0, duplicate: 0, failed: 0};
  for (const file of selectedUploads) {
    const outcome = uploadOutcomes.get(file);
    if (outcome && counts[outcome.status] !== undefined) counts[outcome.status]++;
  }
  const processed = updateUploadProgress(total);
  const summary = `Processed ${processed} of ${total} images · ${counts.indexed} successful · ${counts.duplicate} skipped · ${counts.duplicate} duplicates · ${counts.failed} failed.`;
  const failures = selectedUploads
    .map(file => uploadOutcomes.get(file))
    .filter(outcome => outcome?.status === "failed");
  const errorExamples = failures.slice(0, 3).map(item => `${item.filename}: ${item.error}`);
  if (failures.length > errorExamples.length) {
    errorExamples.push(`and ${failures.length - errorExamples.length} more failed files`);
  }
  setUploadStatus([summary, ...errorExamples].join(" "), counts.failed > 0);
  $("retry-uploads").classList.toggle("hidden", counts.failed === 0);
  $("retry-uploads").disabled = uploading || counts.failed === 0;
  if (counts.indexed > 0) {
    $("engine-status").textContent = "Connected · index ready";
    document.querySelector(".pulse").style.background = "#65d7a0";
  }
}

async function uploadSelected(files) {
  if (uploading || !files.length) return;
  setUploading(true);
  $("upload-progress").classList.remove("hidden");
  $("upload-progress").setAttribute("aria-valuenow", "0");
  $("upload-progress-fill").style.width = "0%";
  setUploadStatus("Preparing upload batches…");
  try {
    const response = await fetch("/api/upload/limits");
    const limits = await response.json();
    if (!response.ok) throw new Error(limits.detail || "Upload limits could not be loaded.");
    uploadLimits = {
      maxFiles: Number(limits.max_files),
      maxFileBytes: Number(limits.max_file_bytes),
      maxRequestBytes: Number(limits.max_request_bytes)
    };
    if (!Number.isInteger(uploadLimits.maxFiles) || uploadLimits.maxFiles < 1 ||
        !Number.isFinite(uploadLimits.maxFileBytes) || uploadLimits.maxFileBytes < 1 ||
        !Number.isFinite(uploadLimits.maxRequestBytes) || uploadLimits.maxRequestBytes < 1) {
      throw new Error("The server returned invalid upload limits.");
    }

    for (const file of files) uploadOutcomes.delete(file);
    const {batches, locallyRejected} = splitUploadBatches(files);
    for (const rejected of locallyRejected) {
      uploadOutcomes.set(rejected.file, {
        filename: rejected.file.name,
        status: "failed",
        error: rejected.error
      });
    }
    let completedBatches = 0;
    const batchQueue = batches.slice();
    while (batchQueue.length) {
      const batch = batchQueue.shift();
      const batchNumber = completedBatches + 1;
      const batchCount = completedBatches + batchQueue.length + 1;
      setUploadStatus(`Processed ${updateUploadProgress(selectedUploads.length)} of ${selectedUploads.length} images · uploading batch ${batchNumber} of ${batchCount}…`);
      let data;
      try {
        data = await requestUploadBatch(batch);
      } catch (error) {
        if (error.status === 413 && batch.length > 1) {
          const midpoint = Math.ceil(batch.length / 2);
          batchQueue.unshift(batch.slice(midpoint));
          batchQueue.unshift(batch.slice(0, midpoint));
          continue;
        }
        for (const file of batch) {
          uploadOutcomes.set(file, {filename: file.name, status: "failed", error: error.message});
        }
        completedBatches++;
        if (!error.status || error.status >= 500) {
          for (const remainingBatch of batchQueue) {
            for (const file of remainingBatch) {
              uploadOutcomes.set(file, {
                filename: file.name,
                status: "failed",
                error: "Not sent because the server or connection failed."
              });
            }
          }
          break;
        }
        continue;
      }

      if (!Array.isArray(data.file_results) || data.file_results.length !== batch.length) {
        for (const file of batch) {
          uploadOutcomes.set(file, {
            filename: file.name,
            status: "failed",
            error: "The server response did not include a result for this file."
          });
        }
        completedBatches++;
        continue;
      }
      batch.forEach((file, index) => uploadOutcomes.set(file, data.file_results[index]));
      completedBatches++;
      if (data.indexed_count > 0) {
        $("engine-status").textContent = "Connected · index ready";
        document.querySelector(".pulse").style.background = "#65d7a0";
      }
      updateUploadProgress(selectedUploads.length);
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    renderUploadSummary(selectedUploads.length);
  } catch (error) {
    setUploadStatus(error.message || "The upload could not be prepared.", true);
  } finally {
    setUploading(false);
    renderUploadSelection();
    if (files.length && files.every(file => uploadOutcomes.has(file))) {
      renderUploadSummary(selectedUploads.length);
    }
  }
}

$("upload-form").addEventListener("submit", event => {
  event.preventDefault();
  uploadSelected(selectedUploads);
});
$("retry-uploads").addEventListener("click", () => {
  const failed = selectedUploads.filter(file => uploadOutcomes.get(file)?.status === "failed");
  uploadSelected(failed);
});

fetch("/api/health").then(r => r.json()).then(data => {
  $("engine-status").textContent = data.index_ready ? "Connected · index ready" : "Connected · index needed";
  document.querySelector(".pulse").style.background = data.index_ready ? "#65d7a0" : "#e9b85f";
}).catch(() => {
  $("engine-status").textContent = "API connection unavailable";
});
