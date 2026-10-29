const $ = (id) => document.getElementById(id);
let mode = "text";
const message = $("message");
const results = $("results");

function setMode(nextMode) {
  mode = nextMode;
  $("crumb-mode").textContent = nextMode === "text" ? "Text Search" : "Image Search";
  $("text-form").classList.toggle("hidden", nextMode !== "text");
  $("image-form").classList.toggle("hidden", nextMode !== "image");
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
fetch("/api/health").then(r => r.json()).then(data => {
  $("engine-status").textContent = data.index_ready ? "Connected · index ready" : "Connected · index needed";
  document.querySelector(".pulse").style.background = data.index_ready ? "#65d7a0" : "#e9b85f";
}).catch(() => {
  $("engine-status").textContent = "API connection unavailable";
});
