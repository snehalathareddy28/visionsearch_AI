/**
 * region_search.js — Region Search mode for VisionSearch AI.
 *
 * Adds to the existing app.js mode system:
 *  - "region" mode tab / nav item handling
 *  - Segmented controls for search_scope and split_mode
 *  - Image upload → /api/region/search/image/detect → crop thumbnails
 *  - Canvas draw-your-own-box interaction
 *  - Region text search → /api/region/search/text
 *  - Image/region/bbox search → /api/region/search/image
 *  - renderRegionResults: results with bbox overlay on cards
 */

(function () {
  "use strict";

  // ---- State ----
  let regionScope = "both";
  let regionSplitMode = "yolo";
  let selectedRegionId = null;  // numeric crop id, or null for whole/bbox
  let drawnBbox = null;         // {x1,y1,x2,y2} in image pixel space
  let detectedRegions = [];     // from /detect endpoint
  let canvasImageElement = null;
  let isDrawing = false;
  let drawStart = { x: 0, y: 0 };
  let uploadedImageFile = null;
  let detectionRequestId = 0;
  let detectedSplitMode = null;
  let regionSelectionNeedsRedetection = false;
  let searchRequestSequence = 0;
  let activeSearchRequestId = null;

  // ---- Helpers ----
  const $id = (id) => document.getElementById(id);

  function getTopK() {
    return Number($id("top-k")?.value ?? 10);
  }

  function createSearchRequestId() {
    searchRequestSequence += 1;
    return `${Date.now().toString(36)}-${searchRequestSequence.toString(36)}`;
  }

  function isCurrentSearch(requestId) {
    return activeSearchRequestId === requestId;
  }

  // ---- Extend existing setMode to handle "region" ----
  const _originalSetMode = window.setMode;
  window.setMode = function (nextMode) {
    if (typeof _originalSetMode === "function") _originalSetMode(nextMode);
    const regionForm = $id("region-form");
    if (regionForm) regionForm.classList.toggle("hidden", nextMode !== "region");
    // Update crumb label if original didn't
    if (nextMode === "region") {
      const crumb = $id("crumb-mode");
      if (crumb) crumb.textContent = "Region Search";
    }
  };

  // ---- Segmented controls ----
  document.querySelectorAll("#scope-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#scope-seg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      regionScope = btn.dataset.value;
    });
  });

  document.querySelectorAll("#split-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#split-seg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const nextMode = btn.dataset.value;
      if (nextMode !== regionSplitMode && detectedRegions.length) {
        detectedRegions = [];
        selectedRegionId = null;
        detectedSplitMode = null;
        regionSelectionNeedsRedetection = true;
        renderCropThumbnails([]);
        redrawCanvas();
        showRegionMessage("Detection mode changed. Detect regions again before searching with a selected region.");
      }
      regionSplitMode = nextMode;
    });
  });

  // ---- Text search ----
  $id("region-text-search-btn")?.addEventListener("click", async () => {
    const query = $id("region-query")?.value.trim();
    if (!query) {
      showRegionMessage("Please enter a search query.", true);
      return;
    }
    const requestId = createSearchRequestId();
    activeSearchRequestId = requestId;
    showRegionMessage("Searching with CLIP + region index…");
    clearResults();
    $id("results-title").textContent = `Region results for "${query}"`;

    try {
      const resp = await fetch("/api/region/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: getTopK(),
          search_scope: regionScope,
          whole_weight: 0.5,
          split_mode: regionSplitMode,
          request_id: requestId,
        }),
      });
      const data = await resp.json();
      if (!isCurrentSearch(requestId)) return;
      if (!resp.ok) throw new Error(data.detail || "Region text search failed.");
      if (resp.headers.get("X-Request-ID") !== requestId) {
        throw new Error("Search response ID did not match the current request. Please retry.");
      }
      renderRegionResults(data);
    } catch (err) {
      if (isCurrentSearch(requestId)) showRegionMessage(err.message, true);
    }
  });

  // ---- Image upload & detect ----
  $id("region-image-file")?.addEventListener("change", () => {
    const file = $id("region-image-file").files[0];
    if (!file) return;
    detectionRequestId += 1;
    activeSearchRequestId = null;
    uploadedImageFile = file;
    detectedRegions = [];
    selectedRegionId = null;
    detectedSplitMode = null;
    regionSelectionNeedsRedetection = false;
    drawnBbox = null;
    $id("region-crops-row").innerHTML = "";
    $id("region-detect-btn").style.display = "inline-flex";
    // Show preview on canvas
    const reader = new FileReader();
    reader.onload = (e) => loadImageOnCanvas(e.target.result);
    reader.readAsDataURL(file);
  });

  $id("region-detect-btn")?.addEventListener("click", async () => {
    if (!uploadedImageFile) return;
    const requestId = ++detectionRequestId;
    const requestedMode = regionSplitMode;
    detectedRegions = [];
    selectedRegionId = null;
    drawnBbox = null;
    renderCropThumbnails([]);
    redrawCanvas();
    showRegionMessage(
      requestedMode === "yolo" ? "Detecting objects with YOLOv8…" : "Splitting image into grid tiles…"
    );
    const form = new FormData();
    form.append("file", uploadedImageFile);
    form.append("split_mode", requestedMode);
    const detectButton = $id("region-detect-btn");
    detectButton.disabled = true;

    try {
      const resp = await fetch("/api/region/search/image/detect", {
        method: "POST",
        body: form,
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Detection failed.");
      if (requestId !== detectionRequestId) return;
      if (data.requested_mode !== requestedMode || !Array.isArray(data.regions)) {
        throw new Error("The detection response did not match the selected method. Please try again.");
      }
      detectedRegions = data.regions || [];
      detectedSplitMode = requestedMode;
      regionSelectionNeedsRedetection = false;
      renderCropThumbnails(detectedRegions);
      drawDetectedBoxesOnCanvas(detectedRegions);
      if (data.fallback) {
        showRegionMessage(`${data.message} Grid tiles are shown, not YOLO detections.`);
      } else if (requestedMode === "yolo" && detectedRegions.length === 0) {
        showRegionMessage("YOLO found no valid objects. No grid tiles were substituted.");
      } else if (data.used_mode === "grid") {
        showRegionMessage(`Grid mode returned ${detectedRegions.length} tiles.`);
      } else {
        hideRegionMessage();
      }
    } catch (err) {
      if (requestId === detectionRequestId) {
        detectedRegions = [];
        renderCropThumbnails([]);
        redrawCanvas();
        showRegionMessage(err.message, true);
      }
    } finally {
      if (requestId === detectionRequestId) detectButton.disabled = false;
    }
  });

  // ---- Search with crop / bbox / whole ----
  $id("region-search-crop-btn")?.addEventListener("click", async () => {
    if (!uploadedImageFile) {
      showRegionMessage("Please upload an image first.", true);
      return;
    }
    if (regionSelectionNeedsRedetection || (selectedRegionId !== null && detectedSplitMode !== regionSplitMode)) {
      showRegionMessage("Detection mode changed. Detect regions again before searching with a selected region.", true);
      return;
    }
    const requestId = createSearchRequestId();
    activeSearchRequestId = requestId;
    showRegionMessage("Searching region index…");
    clearResults();
    $id("results-title").textContent = "Region search results";

    const form = new FormData();
    form.append("file", uploadedImageFile);
    form.append("top_k", getTopK());
    form.append("search_scope", regionScope);
    form.append("whole_weight", "0.5");
    form.append("split_mode", regionSplitMode);
    form.append("request_id", requestId);

    if (drawnBbox) {
      form.append("bbox_x1", drawnBbox.x1);
      form.append("bbox_y1", drawnBbox.y1);
      form.append("bbox_x2", drawnBbox.x2);
      form.append("bbox_y2", drawnBbox.y2);
      form.append("query_label", "custom_bbox");
      if (regionSplitMode !== "auto") form.append("query_crop_type", regionSplitMode);
    } else if (selectedRegionId !== null) {
      const selected = detectedRegions.find((region) => Number(region.region_id) === Number(selectedRegionId));
      if (!selected) {
        showRegionMessage("The selected region is no longer available. Detect regions again.", true);
        return;
      }
      form.append("region_id", selectedRegionId);
      form.append("bbox_x1", selected.bbox[0]);
      form.append("bbox_y1", selected.bbox[1]);
      form.append("bbox_x2", selected.bbox[2]);
      form.append("bbox_y2", selected.bbox[3]);
      form.append("query_label", selected.class_label);
      form.append("query_crop_type", selected.crop_type);
    }
    // else: whole image (no bbox/region_id fields)

    try {
      const resp = await fetch("/api/region/search/image", {
        method: "POST",
        body: form,
      });
      const data = await resp.json();
      if (!isCurrentSearch(requestId)) return;
      if (!resp.ok) throw new Error(data.detail || "Region image search failed.");
      if (resp.headers.get("X-Request-ID") !== requestId) {
        throw new Error("Search response ID did not match the current request. Please retry.");
      }
      renderRegionResults(data);
    } catch (err) {
      if (isCurrentSearch(requestId)) showRegionMessage(err.message, true);
    }
  });

  // ---- Clear drawn box ----
  $id("region-clear-draw")?.addEventListener("click", () => {
    drawnBbox = null;
    selectedRegionId = null;
    document.querySelectorAll(".crop-thumb").forEach((t) => t.classList.remove("selected"));
    if (canvasImageElement) drawDetectedBoxesOnCanvas(detectedRegions);
  });

  // ---- Canvas interaction ----
  function loadImageOnCanvas(src) {
    const canvas = $id("region-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvasImageElement = img;
      const maxW = Math.min(600, canvas.parentElement.offsetWidth || 600);
      const scale = maxW / img.naturalWidth;
      canvas.width = Math.round(img.naturalWidth * scale);
      canvas.height = Math.round(img.naturalHeight * scale);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      $id("region-canvas-area").classList.remove("hidden");
    };
    img.src = src;
  }

  function getCanvasScale() {
    if (!canvasImageElement) return 1;
    const canvas = $id("region-canvas");
    return canvasImageElement.naturalWidth / canvas.width;
  }

  function canvasPos(e) {
    const canvas = $id("region-canvas");
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  const canvas = $id("region-canvas");
  if (canvas) {
    canvas.addEventListener("mousedown", (e) => {
      isDrawing = true;
      drawStart = canvasPos(e);
      drawnBbox = null;
      selectedRegionId = null;
      document.querySelectorAll(".crop-thumb").forEach((t) => t.classList.remove("selected"));
    });

    canvas.addEventListener("mousemove", (e) => {
      if (!isDrawing || !canvasImageElement) return;
      const pos = canvasPos(e);
      redrawCanvas();
      const ctx = canvas.getContext("2d");
      ctx.strokeStyle = "#7eebd9";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(
        drawStart.x, drawStart.y,
        pos.x - drawStart.x, pos.y - drawStart.y
      );
      ctx.setLineDash([]);
    });

    canvas.addEventListener("mouseup", (e) => {
      if (!isDrawing) return;
      isDrawing = false;
      const pos = canvasPos(e);
      const scale = getCanvasScale();
      drawnBbox = {
        x1: Math.min(drawStart.x, pos.x) * scale,
        y1: Math.min(drawStart.y, pos.y) * scale,
        x2: Math.max(drawStart.x, pos.x) * scale,
        y2: Math.max(drawStart.y, pos.y) * scale,
      };
      redrawCanvas();
      drawBboxOnCanvas(drawStart.x, drawStart.y, pos.x - drawStart.x, pos.y - drawStart.y, "#7eebd9", "custom");
    });
  }

  function redrawCanvas() {
    const canvas = $id("region-canvas");
    if (!canvas || !canvasImageElement) return;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(canvasImageElement, 0, 0, canvas.width, canvas.height);
  }

  function drawDetectedBoxesOnCanvas(regions) {
    const canvas = $id("region-canvas");
    if (!canvas || !canvasImageElement) return;
    redrawCanvas();
    const scale = 1 / getCanvasScale();
    const colours = ["#7eebd9", "#ffd97d", "#ff8fab", "#a0cfff", "#c6f87e"];
    regions.forEach((r, i) => {
      const colour = colours[i % colours.length];
      const [x1, y1, x2, y2] = r.bbox.map((v) => v * scale);
      drawBboxOnCanvas(x1, y1, x2 - x1, y2 - y1, colour, regionDisplayLabel(r));
    });
  }

  function regionDisplayLabel(region) {
    if (region.crop_type === "yolo") return region.class_label || "Object";
    return `Grid tile ${Number(region.region_id) + 1}`;
  }

  function drawBboxOnCanvas(x, y, w, h, colour, label) {
    const canvas = $id("region-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    if (label) {
      ctx.fillStyle = colour;
      ctx.fillRect(x, y - 18, ctx.measureText(label).width + 8, 18);
      ctx.fillStyle = "#111";
      ctx.font = "bold 11px monospace";
      ctx.fillText(label, x + 4, y - 4);
    }
  }

  // ---- Crop thumbnails ----
  function renderCropThumbnails(regions) {
    const row = $id("region-crops-row");
    if (!row) return;
    row.innerHTML = "";
    regions.forEach((r) => {
      const thumb = document.createElement("div");
      thumb.className = "crop-thumb";
      thumb.dataset.regionId = r.region_id;

      const img = document.createElement("img");
      img.src = `data:image/jpeg;base64,${r.thumbnail_b64}`;
      img.alt = regionDisplayLabel(r);

      const lbl = document.createElement("div");
      lbl.className = "crop-label";
      const label = regionDisplayLabel(r);
      lbl.textContent = label + (r.crop_type === "yolo" ? ` ${(r.confidence * 100).toFixed(0)}%` : "");

      thumb.append(img, lbl);
      thumb.addEventListener("click", () => {
        document.querySelectorAll(".crop-thumb").forEach((t) => t.classList.remove("selected"));
        thumb.classList.add("selected");
        selectedRegionId = r.region_id;
        drawnBbox = null;
        // Highlight this box on canvas
        redrawCanvas();
        drawDetectedBoxesOnCanvas(regions);
        const scale = 1 / getCanvasScale();
        const [x1, y1, x2, y2] = r.bbox.map((v) => v * scale);
        const ctx = $id("region-canvas").getContext("2d");
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      });
      row.appendChild(thumb);
    });
  }

  // ---- Result rendering with bbox overlay ----
  function renderRegionResults(items) {
    const results = $id("results");
    const message = $id("message");
    results.innerHTML = "";
    if (!Array.isArray(items) || !items.length) {
      message.textContent = "No matching regions returned for this search mode. Check the index or try another mode.";
      message.style.color = "";
      message.classList.remove("hidden");
      return;
    }
    const validItems = items.filter(
      (item) => item && typeof item === "object" && typeof item.image_url === "string"
    );
    if (!validItems.length) {
      message.textContent = "Region results were returned in an unsupported format. Refresh the page and try again.";
      message.style.color = "#ff9baf";
      message.classList.remove("hidden");
      return;
    }
    message.classList.add("hidden");

    validItems.forEach((item) => {
      const card = document.createElement("article");
      card.className = "image-card";
      card.style.position = "relative";

      const img = document.createElement("img");
      img.src = item.image_url;
      img.alt = item.class_label || "Region result";
      img.loading = "lazy";
      img.style.display = "block";
      img.style.width = "100%";

      const meta = document.createElement("div");
      meta.className = "image-meta";

      const nameLine = document.createElement("strong");
      nameLine.textContent =
        typeof item.name === "string" && item.name ? item.name : "Region result";
      meta.appendChild(nameLine);

      const scoreLine = document.createElement("span");
      scoreLine.className = "score";
      scoreLine.textContent = Number.isFinite(item.score)
        ? `Similarity: ${item.score.toFixed(3)}`
        : "Similarity score unavailable";
      meta.appendChild(scoreLine);

      if (item.class_label && item.class_label !== "whole") {
        const labelLine = document.createElement("span");
        labelLine.className = "score";
        const confPct = Number.isFinite(item.confidence) && item.confidence > 0
          ? ` ${(item.confidence * 100).toFixed(0)}%`
          : "";
        labelLine.textContent = `${item.class_label}${confPct} [${item.match_type}]`;
        meta.appendChild(labelLine);
      }

      if (window.VisionSearchShare) {
        meta.appendChild(window.VisionSearchShare.createShareControl(item));
      }

      card.append(img, meta);

      // Draw bbox overlay after image loads
      if (item.matched_bbox) {
        const [bx1, by1, bx2, by2] = item.matched_bbox;
        img.onload = () => {
          const iw = img.naturalWidth;
          const ih = img.naturalHeight;
          const dw = img.offsetWidth || img.width;
          const dh = img.offsetHeight || img.height;
          const scaleX = dw / iw;
          const scaleY = dh / ih;

          const overlay = document.createElement("div");
          overlay.className = "bbox-overlay";
          overlay.style.left = `${bx1 * scaleX}px`;
          overlay.style.top = `${by1 * scaleY}px`;
          overlay.style.width = `${(bx2 - bx1) * scaleX}px`;
          overlay.style.height = `${(by2 - by1) * scaleY}px`;

          if (item.class_label && item.class_label !== "whole") {
            const labelEl = document.createElement("div");
            labelEl.className = "bbox-label";
            labelEl.textContent = item.class_label;
            overlay.appendChild(labelEl);
          }
          card.appendChild(overlay);
        };
      }

      results.appendChild(card);
    });
  }

  // ---- Message helpers ----
  function showRegionMessage(text, isError = false) {
    const msg = $id("message");
    if (!msg) return;
    msg.textContent = text;
    msg.style.color = isError ? "#ff9baf" : "";
    msg.classList.remove("hidden");
  }

  function hideRegionMessage() {
    const msg = $id("message");
    if (msg) msg.classList.add("hidden");
  }

  function clearResults() {
    const r = $id("results");
    if (r) r.innerHTML = "";
    hideRegionMessage();
  }

})();
