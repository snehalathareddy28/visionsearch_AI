(function () {
  "use strict";

  const defaultMessage = "Found this image using VisionSearch AI!";

  function safeFilename(value) {
    const filename = String(value || "visionsearch-image.jpg")
      .split(/[\\/]/)
      .pop()
      .replace(/[^A-Za-z0-9._-]+/g, "_")
      .replace(/^\.+/, "");
    return filename || "visionsearch-image.jpg";
  }

  function createAction(label, className, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.addEventListener("click", handler);
    return button;
  }

  function createShareControl(item) {
    const details = document.createElement("details");
    details.className = "share-control";

    const summary = document.createElement("summary");
    summary.setAttribute("aria-label", "Share or download image");
    summary.innerHTML = '<span aria-hidden="true">↗</span> Share';

    const menu = document.createElement("div");
    menu.className = "share-menu";
    const feedback = document.createElement("div");
    feedback.className = "share-feedback";
    feedback.setAttribute("role", "status");
    feedback.setAttribute("aria-live", "polite");

    const imageUrl = new URL(item.image_url, window.location.href).href;
    const publicUrl = item.share_url
      ? new URL(item.share_url, window.location.href).href
      : null;
    const isLocalHost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
    const copyUrl = publicUrl || (isLocalHost ? imageUrl : null);
    const filename = safeFilename(item.name);
    const description = [defaultMessage];
    if (item.class_label && item.class_label !== "whole") {
      description.push(`Matched object: ${item.class_label}.`);
    }
    if (typeof item.score === "number" && Number.isFinite(item.score)) {
      description.push(`Similarity score: ${item.score.toFixed(3)}.`);
    }
    const message = description.join(" ");

    function report(text, isError) {
      feedback.textContent = text;
      feedback.classList.toggle("error", Boolean(isError));
    }

    function openShareUrl(url) {
      if (!publicUrl) {
        report("Public image links are disabled for this deployment.", true);
        return;
      }
      window.open(url(publicUrl), "_blank", "noopener,noreferrer");
      report("Share option opened.");
    }

    const whatsapp = createAction("WhatsApp", "share-menu-action", () =>
      openShareUrl((url) => `https://wa.me/?text=${encodeURIComponent(`${message} ${url}`)}`)
    );
    const telegram = createAction("Telegram", "share-menu-action", () =>
      openShareUrl((url) =>
        `https://t.me/share/url?url=${encodeURIComponent(url)}&text=${encodeURIComponent(message)}`
      )
    );
    const email = createAction("Email", "share-menu-action", () =>
      openShareUrl((url) =>
        `mailto:?subject=${encodeURIComponent("Image from VisionSearch AI")}` +
        `&body=${encodeURIComponent(`${message}\n${url}`)}`
      )
    );
    const copyLink = createAction("Copy image link", "share-menu-action", async () => {
      if (!copyUrl) {
        report("Public image links are disabled for this deployment.", true);
        return;
      }
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(copyUrl);
        } else {
          const input = document.createElement("textarea");
          input.value = copyUrl;
          input.style.position = "fixed";
          input.style.opacity = "0";
          document.body.appendChild(input);
          input.select();
          const copied = document.execCommand("copy");
          input.remove();
          if (!copied) throw new Error("Clipboard access is unavailable.");
        }
        report(isLocalHost && !publicUrl ? "Local-only image link copied." : "Image link copied.");
      } catch {
        report("Could not copy the image link. Check browser clipboard permissions.", true);
      }
    });
    const download = document.createElement("a");
    download.className = "share-menu-action";
    download.href = imageUrl;
    download.download = filename;
    download.textContent = "Download image";

    const nativeShare = createAction("Share image…", "share-menu-action", async () => {
      if (!navigator.share) {
        report("Device sharing is not supported in this browser.", true);
        return;
      }
      try {
        const response = await fetch(imageUrl);
        if (!response.ok) throw new Error("The image could not be loaded.");
        const blob = await response.blob();
        if (typeof File !== "undefined") {
          const file = new File([blob], filename, {
            type: blob.type || "application/octet-stream",
          });
          const files = [file];
          if (navigator.canShare && navigator.canShare({ files })) {
            try {
              await navigator.share({ files, title: filename, text: message });
              report("Image shared.");
              return;
            } catch (error) {
              if (error.name === "AbortError") return;
              if (!copyUrl) throw error;
            }
          }
        }
        await navigator.share({
          title: filename,
          text: message,
          ...(publicUrl ? { url: publicUrl } : {}),
        });
        report(
          publicUrl
            ? "Share sheet opened with the image link."
            : "Share sheet opened without a link; this local image URL is not public."
        );
      } catch (error) {
        if (error.name === "AbortError") return;
        report(error.message || "The image could not be shared.", true);
      }
    });

    for (const action of [whatsapp, telegram, email]) action.disabled = !publicUrl;
    copyLink.disabled = !copyUrl;
    if (!publicUrl) {
      const notice = document.createElement("small");
      notice.className = "share-link-notice";
      notice.textContent = isLocalHost
        ? "Copied links are local-only. Enable public sharing after approving the deployed image collection."
        : "Public links are off. Enable them only for an approved public image collection.";
      menu.appendChild(notice);
    }
    menu.append(whatsapp, telegram, email, copyLink, download, nativeShare, feedback);
    details.append(summary, menu);
    return details;
  }

  window.VisionSearchShare = { createShareControl };
})();
