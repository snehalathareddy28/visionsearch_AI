(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const csrfCookieName = "visionsearch_secure_csrf";
  let csrfToken = readCookie(csrfCookieName);
  let currentUser = null;
  let authGeneration = 0;
  let selectedCandidate = null;
  let candidates = [];
  let searchPreviewUrl = null;
  let resultPreviewUrl = null;
  let candidateSelectionSequence = 0;
  let searchBusy = false;

  function readCookie(name) {
    const prefix = `${name}=`;
    const entry = document.cookie.split("; ").find((cookie) => cookie.startsWith(prefix));
    if (!entry) return null;
    try {
      return decodeURIComponent(entry.slice(prefix.length));
    } catch {
      return null;
    }
  }

  function setMessage(id, text, kind = "info") {
    const element = $(id);
    element.textContent = text;
    element.classList.remove("hidden", "error", "success");
    if (kind === "error" || kind === "success") element.classList.add(kind);
  }

  function clearMessage(id) {
    const element = $(id);
    element.textContent = "";
    element.classList.add("hidden");
    element.classList.remove("error", "success");
  }

  function setAuthenticatedUser(user) {
    authGeneration += 1;
    currentUser = user;
    const authenticated = Boolean(user);
    const isAdmin = authenticated && user.role === "admin";
    $("secure-login-form").classList.toggle("hidden", authenticated);
    $("secure-session-panel").classList.toggle("hidden", !authenticated);
    $("silver-user-panel").classList.toggle("hidden", !authenticated);
    $("silver-admin-panel").classList.toggle("hidden", !isAdmin);
    if (authenticated) {
      $("secure-session-label").textContent = `Signed in as ${user.username}`;
      $("secure-session-role").textContent = user.role === "admin" ? "Administrator" : "Authorized user";
      $("secure-login-form").reset();
      clearMessage("silver-auth-message");
    } else {
      $("secure-session-label").textContent = "";
      $("secure-session-role").textContent = "";
      $("silver-records").replaceChildren();
      $("silver-users").replaceChildren();
      clearRetrievedInformation();
      selectedCandidate = null;
      candidates = [];
      candidateSelectionSequence += 1;
      $("silver-candidates").replaceChildren();
      $("silver-selected-photo").classList.add("hidden");
      $("silver-registration-form").reset();
      $("silver-user-create-form").reset();
      clearMessage("silver-user-management-message");
    }
  }

  function clearRetrievedInformation() {
    $("silver-search-result").classList.add("hidden");
    $("silver-result-title").textContent = "";
    $("silver-result-similarity").textContent = "";
    $("silver-result-information").textContent = "";
    $("silver-result-preview").removeAttribute("src");
    if (resultPreviewUrl) URL.revokeObjectURL(resultPreviewUrl);
    resultPreviewUrl = null;
  }

  function expireSession(message) {
    setAuthenticatedUser(null);
    setMessage("silver-auth-message", message || "Your session has expired. Sign in again.", "error");
  }

  async function responseData(response) {
    let data = {};
    try {
      data = await response.json();
    } catch {
      if (response.status !== 204) throw new Error("The server returned an unreadable response.");
    }
    return data;
  }

  async function request(url, options = {}) {
    const {mutating = false, ...requestOptions} = options;
    const headers = new Headers(requestOptions.headers || {});
    if (mutating) {
      if (!csrfToken) throw new Error("Your security token is unavailable. Sign in again.");
      headers.set("X-CSRF-Token", csrfToken);
    }
    const response = await fetch(url, {
      ...requestOptions,
      headers,
      credentials: "same-origin"
    });
    const data = await responseData(response);
    if (response.status === 401 && currentUser) {
      expireSession("Your session expired. Sign in again to continue.");
    }
    if (!response.ok) {
      throw new Error(response.status === 401
        ? (data.detail || "Sign in is required.")
        : (data.detail || "The request could not be completed."));
    }
    return data;
  }

  async function issueCsrfToken() {
    const data = await request("/api/secure-info/auth/csrf");
    if (typeof data.csrf_token !== "string" || !data.csrf_token) {
      throw new Error("A security token could not be initialized.");
    }
    csrfToken = data.csrf_token;
  }

  async function initializeSession() {
    try {
      const hadSessionCsrf = Boolean(csrfToken);
      if (!csrfToken) await issueCsrfToken();
      const response = await fetch("/api/secure-info/auth/me", {
        credentials: "same-origin"
      });
      const data = await responseData(response);
      if (response.ok && hadSessionCsrf && data.username && data.role) {
        setAuthenticatedUser(data);
        if (data.role === "admin") {
          await Promise.all([refreshRegistrations(), refreshManagedUsers(true)]);
        }
      } else {
        setAuthenticatedUser(null);
        if (response.status === 401 && hadSessionCsrf) {
          setMessage("silver-auth-message", "Your session has expired. Sign in again.", "error");
        } else if (response.ok && !hadSessionCsrf) {
          setMessage("silver-auth-message", "Your saved session needs a fresh sign-in.", "error");
        }
      }
    } catch {
      setAuthenticatedUser(null);
      setMessage("silver-auth-message", "Authentication is temporarily unavailable. Try again.", "error");
    }
  }

  $("secure-login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("secure-login-submit");
    const username = $("secure-username").value.trim();
    const password = $("secure-password").value;
    if (!username || !password) {
      setMessage("silver-auth-message", "Enter your username and password.", "error");
      return;
    }
    button.disabled = true;
    button.textContent = "Signing in…";
    clearMessage("silver-auth-message");
    try {
      if (!csrfToken) await issueCsrfToken();
      const data = await request("/api/secure-info/auth/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username, password}),
        mutating: true
      });
      if (!data.user || typeof data.csrf_token !== "string") {
        throw new Error("The sign-in response was incomplete.");
      }
      csrfToken = data.csrf_token;
      const verified = await request("/api/secure-info/auth/me");
      setAuthenticatedUser(verified);
      if (verified.role === "admin") {
        await Promise.all([refreshRegistrations(), refreshManagedUsers(true)]);
      }
    } catch (error) {
      setMessage("silver-auth-message", error.message || "Sign-in failed.", "error");
    } finally {
      $("secure-password").value = "";
      button.disabled = false;
      button.innerHTML = "Sign in <span>↗</span>";
    }
  });

  $("secure-logout").addEventListener("click", async () => {
    $("secure-logout").disabled = true;
    try {
      await request("/api/secure-info/auth/logout", {method: "POST", mutating: true});
      csrfToken = null;
      setAuthenticatedUser(null);
      await issueCsrfToken();
      setMessage("silver-auth-message", "You have signed out.", "success");
    } catch (error) {
      setMessage("silver-auth-message", error.message || "Sign-out failed.", "error");
    } finally {
      $("secure-logout").disabled = false;
    }
  });

  $("silver-search-file").addEventListener("change", () => {
    const file = $("silver-search-file").files[0];
    $("silver-search-file-label").textContent = file ? file.name : "Choose a college photograph";
    $("silver-search-submit").disabled = !file || searchBusy;
    clearRetrievedInformation();
    clearMessage("silver-search-message");
    if (searchPreviewUrl) URL.revokeObjectURL(searchPreviewUrl);
    searchPreviewUrl = file ? URL.createObjectURL(file) : null;
    const preview = $("silver-search-preview");
    preview.classList.toggle("hidden", !searchPreviewUrl);
    if (searchPreviewUrl) preview.src = searchPreviewUrl;
  });

  function setSearchBusy(value) {
    searchBusy = value;
    $("silver-search-file").disabled = value;
    $("silver-search-submit").disabled = value || !$("silver-search-file").files.length;
    $("silver-search-submit").textContent = value ? "Matching photograph…" : "Search Silver Jubilee records ↗";
  }

  $("silver-search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = $("silver-search-file").files[0];
    if (!file || !currentUser) {
      setMessage("silver-search-message", "Sign in and choose a photograph first.", "error");
      return;
    }
    const requestAuthGeneration = authGeneration;
    clearRetrievedInformation();
    clearMessage("silver-search-message");
    setSearchBusy(true);
    setMessage("silver-search-message", "Searching the registered image collection…");
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await request("/api/silver-jubilee/search", {
        method: "POST",
        body: form,
        mutating: true
      });
      if (requestAuthGeneration !== authGeneration || !currentUser) return;
      if (!data.matched) {
        const messages = {
          no_confident_match: "No registered photograph was similar enough. Try a clearer or different photograph.",
          no_registered_information: "A similar image was found, but it has no Silver Jubilee information registered.",
          ambiguous_match: "Several photographs matched too closely to identify one safely. Try another photograph."
        };
        setMessage("silver-search-message", messages[data.reason] || "No registered photograph was matched.");
        return;
      }
      if (typeof data.title !== "string" || typeof data.information !== "string") {
        throw new Error("The retrieved information response was incomplete.");
      }
      if (resultPreviewUrl) URL.revokeObjectURL(resultPreviewUrl);
      resultPreviewUrl = URL.createObjectURL(file);
      $("silver-result-preview").src = resultPreviewUrl;
      $("silver-result-title").textContent = data.title;
      $("silver-result-similarity").textContent =
        typeof data.similarity === "number" && Number.isFinite(data.similarity)
          ? `Similarity: ${data.similarity.toFixed(3)}`
          : "Registered match";
      $("silver-result-information").textContent = data.information;
      $("silver-search-result").classList.remove("hidden");
      setMessage("silver-search-message", "Authorized registered information retrieved.", "success");
    } catch (error) {
      if (requestAuthGeneration !== authGeneration || !currentUser) return;
      setMessage("silver-search-message", error.message || "The photograph could not be searched.", "error");
    } finally {
      setSearchBusy(false);
    }
  });

  function safeImageUrl(value) {
    try {
      const imageUrl = new URL(value, window.location.origin);
      if (imageUrl.origin !== window.location.origin || !imageUrl.pathname.startsWith("/image/")) return null;
      return imageUrl;
    } catch {
      return null;
    }
  }

  function referenceFromImageUrl(value) {
    const imageUrl = safeImageUrl(value);
    if (!imageUrl) throw new Error("The selected search result is not a valid indexed image.");
    let reference;
    try {
      reference = decodeURIComponent(imageUrl.pathname.slice("/image/".length));
    } catch {
      throw new Error("The selected image reference is invalid.");
    }
    if (
      !reference ||
      reference.startsWith("/") ||
      reference.includes("\\") ||
      reference.split("/").some((part) => !part || part === "." || part === "..")
    ) {
      throw new Error("The selected image reference is invalid.");
    }
    return reference;
  }

  async function hashIndexedImage(imageUrl) {
    if (!window.crypto || !window.crypto.subtle) {
      throw new Error("Secure image verification is unavailable in this browser context.");
    }
    const response = await fetch(imageUrl, {credentials: "same-origin"});
    if (!response.ok) throw new Error("The selected indexed photograph could not be verified.");
    const bytes = await response.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  $("silver-candidate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!currentUser || currentUser.role !== "admin") return;
    const query = $("silver-candidate-query").value.trim();
    if (!query) {
      setMessage("silver-candidate-message", "Enter a search description.", "error");
      return;
    }
    $("silver-candidate-submit").disabled = true;
    candidateSelectionSequence += 1;
    $("silver-candidates").replaceChildren();
    selectedCandidate = null;
    $("silver-selected-photo").classList.add("hidden");
    $("silver-registration-submit").disabled = true;
    setMessage("silver-candidate-message", "Finding indexed photographs…");
    try {
      const data = await request("/api/search/text", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query, top_k: 20})
      });
      candidates = Array.isArray(data.results)
        ? data.results.filter((item) => item && typeof item.name === "string" && safeImageUrl(item.image_url))
        : [];
      renderCandidates();
      setMessage(
        "silver-candidate-message",
        candidates.length ? `Choose one of ${candidates.length} indexed search result${candidates.length === 1 ? "" : "s"}.` : "No indexed photographs matched that search."
      );
    } catch (error) {
      setMessage("silver-candidate-message", error.message || "The indexed photographs could not be searched.", "error");
    } finally {
      $("silver-candidate-submit").disabled = false;
    }
  });

  function renderCandidates() {
    const container = $("silver-candidates");
    container.replaceChildren();
    candidates.forEach((candidate, index) => {
      const imageUrl = safeImageUrl(candidate.image_url);
      if (!imageUrl) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "silver-candidate";
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `Select indexed photograph ${candidate.name}`);
      const image = document.createElement("img");
      image.src = imageUrl.href;
      image.alt = "";
      image.loading = "lazy";
      const name = document.createElement("strong");
      name.textContent = candidate.name;
      const category = document.createElement("small");
      category.textContent = typeof candidate.category === "string" ? candidate.category : "Indexed image";
      button.append(image, name, category);
      button.addEventListener("click", () => selectCandidate(index, button));
      container.appendChild(button);
    });
  }

  async function selectCandidate(index, button) {
    const selectionSequence = ++candidateSelectionSequence;
    const candidate = candidates[index];
    const imageUrl = candidate && safeImageUrl(candidate.image_url);
    if (!imageUrl) return;
    for (const item of $("silver-candidates").querySelectorAll(".silver-candidate")) {
      item.setAttribute("aria-pressed", "false");
    }
    button.setAttribute("aria-pressed", "true");
    selectedCandidate = null;
    $("silver-registration-submit").disabled = true;
    const selected = $("silver-selected-photo");
    selected.replaceChildren();
    selected.classList.remove("hidden");
    const preview = document.createElement("img");
    preview.src = imageUrl.href;
    preview.alt = "";
    const label = document.createElement("span");
    label.textContent = "Verifying the selected indexed photograph…";
    selected.append(preview, label);
    try {
      const image_reference = referenceFromImageUrl(imageUrl.href);
      const image_hash = await hashIndexedImage(imageUrl.href);
      if (selectionSequence !== candidateSelectionSequence) return;
      selectedCandidate = {image_reference, image_hash};
      label.textContent = `Selected: ${candidate.name}. The server will re-check this photograph before registration.`;
      $("silver-registration-submit").disabled = false;
      setMessage("silver-candidate-message", "Photograph reference and SHA-256 prepared.");
    } catch (error) {
      if (selectionSequence !== candidateSelectionSequence) return;
      label.textContent = "This photograph could not be verified. Select another result.";
      setMessage("silver-candidate-message", error.message || "Photograph verification failed.", "error");
    }
  }

  function setAdminBusy(value) {
    $("silver-candidate-submit").disabled = value;
    $("silver-registration-submit").disabled = value || !selectedCandidate;
    $("silver-refresh-records").disabled = value;
    $("silver-title").disabled = value;
    $("silver-information").disabled = value;
  }

  function setUserManagementBusy(value) {
    $("silver-user-username").disabled = value;
    $("silver-user-password").disabled = value;
    $("silver-user-create-submit").disabled = value;
    $("silver-refresh-users").disabled = value;
  }

  $("silver-user-create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!currentUser || currentUser.role !== "admin") {
      setMessage("silver-user-management-message", "Administrator access is required.", "error");
      return;
    }
    const username = $("silver-user-username").value.trim();
    const password = $("silver-user-password").value;
    setUserManagementBusy(true);
    setMessage("silver-user-management-message", "Creating user account…");
    try {
      await request("/api/secure-info/users", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username, password}),
        mutating: true
      });
      $("silver-user-create-form").reset();
      if (await refreshManagedUsers()) {
        setMessage("silver-user-management-message", `User “${username}” created.`, "success");
      }
    } catch (error) {
      setMessage("silver-user-management-message", error.message || "The user account could not be created.", "error");
    } finally {
      setUserManagementBusy(false);
    }
  });

  function safeManagedUserId(user) {
    const id = Number(user && user.id);
    return Number.isSafeInteger(id) && id > 0 ? id : null;
  }

  function formatAccountDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Creation date unavailable" : date.toLocaleString();
  }

  async function refreshManagedUsers(showLoading = false) {
    if (!currentUser || currentUser.role !== "admin") return false;
    if (showLoading) setMessage("silver-user-management-message", "Loading user accounts…");
    try {
      const users = await request("/api/secure-info/users");
      renderManagedUsers(Array.isArray(users) ? users : []);
      if (showLoading) clearMessage("silver-user-management-message");
      return true;
    } catch (error) {
      setMessage("silver-user-management-message", error.message || "User accounts could not be loaded.", "error");
      return false;
    }
  }

  function renderManagedUsers(users) {
    const container = $("silver-users");
    container.replaceChildren();
    const regularUsers = users.filter((user) =>
      safeManagedUserId(user)
      && typeof user.username === "string"
      && user.role === "user"
      && typeof user.is_active === "boolean"
    );
    if (!regularUsers.length) {
      const empty = document.createElement("div");
      empty.className = "silver-empty";
      empty.textContent = "No regular user accounts have been created.";
      container.appendChild(empty);
      return;
    }
    for (const user of regularUsers) {
      const id = safeManagedUserId(user);
      const card = document.createElement("article");
      card.className = "silver-user";
      const heading = document.createElement("div");
      heading.className = "silver-user-heading";
      const details = document.createElement("div");
      details.className = "silver-user-details";
      const username = document.createElement("strong");
      username.textContent = user.username;
      const metadata = document.createElement("small");
      metadata.textContent = `Role: ${user.role} · Created: ${formatAccountDate(user.created_at)}`;
      const status = document.createElement("span");
      status.className = `silver-user-status${user.is_active ? "" : " inactive"}`;
      status.textContent = user.is_active ? "ACTIVE" : "INACTIVE";
      details.append(username, metadata, status);

      const actions = document.createElement("div");
      actions.className = "silver-user-actions";
      const statusButton = document.createElement("button");
      statusButton.type = "button";
      statusButton.className = "secondary-button";
      statusButton.textContent = user.is_active ? "Deactivate" : "Activate";
      statusButton.addEventListener("click", () => changeManagedUserStatus(id, user));
      const resetButton = document.createElement("button");
      resetButton.type = "button";
      resetButton.className = "secondary-button";
      resetButton.textContent = "Reset password";
      resetButton.addEventListener("click", () => toggleManagedUserPasswordForm(card, user, id));
      actions.append(statusButton, resetButton);
      heading.append(details, actions);
      card.appendChild(heading);
      container.appendChild(card);
    }
  }

  async function changeManagedUserStatus(id, user) {
    const activate = !user.is_active;
    if (!activate && !window.confirm(`Deactivate “${user.username}”? Their active sessions will be revoked.`)) return;
    setMessage("silver-user-management-message", "Updating account status…");
    try {
      await request(`/api/secure-info/users/${id}/status`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({is_active: activate}),
        mutating: true
      });
      if (await refreshManagedUsers()) {
        setMessage(
          "silver-user-management-message",
          activate ? `“${user.username}” is active.` : `“${user.username}” has been deactivated.`,
          "success"
        );
      }
    } catch (error) {
      setMessage("silver-user-management-message", error.message || "Account status could not be updated.", "error");
    }
  }

  function toggleManagedUserPasswordForm(card, user, id) {
    const existing = card.querySelector(".silver-user-reset");
    if (existing) {
      existing.remove();
      return;
    }
    const form = document.createElement("form");
    form.className = "silver-user-reset";
    const field = document.createElement("div");
    field.className = "silver-field";
    const label = document.createElement("label");
    label.textContent = `Temporary password for ${user.username}`;
    const password = document.createElement("input");
    password.type = "password";
    password.minLength = 12;
    password.maxLength = 1024;
    password.autocomplete = "new-password";
    password.required = true;
    field.append(label, password);
    const save = document.createElement("button");
    save.type = "submit";
    save.className = "primary";
    save.textContent = "Set password";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "secondary-button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => form.remove());
    form.append(field, save, cancel);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!currentUser || currentUser.role !== "admin") {
        setMessage("silver-user-management-message", "Administrator access is required.", "error");
        return;
      }
      save.disabled = true;
      cancel.disabled = true;
      setMessage("silver-user-management-message", "Resetting password…");
      try {
        await request(`/api/secure-info/users/${id}/password`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({password: password.value}),
          mutating: true
        });
        form.reset();
        form.remove();
        setMessage("silver-user-management-message", `Password reset for “${user.username}”.`, "success");
      } catch (error) {
        setMessage("silver-user-management-message", error.message || "The password could not be reset.", "error");
        save.disabled = false;
        cancel.disabled = false;
      }
    });
    card.appendChild(form);
    password.focus();
  }

  $("silver-refresh-users").addEventListener("click", () => refreshManagedUsers(true));

  $("silver-registration-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!currentUser || currentUser.role !== "admin") {
      setMessage("silver-registration-message", "Administrator access is required.", "error");
      return;
    }
    if (!selectedCandidate) {
      setMessage("silver-registration-message", "Choose and verify an indexed photograph first.", "error");
      return;
    }
    setAdminBusy(true);
    clearMessage("silver-registration-message");
    try {
      await request("/api/secure-info/registrations", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          ...selectedCandidate,
          title: $("silver-title").value.trim(),
          information: $("silver-information").value.trim()
        }),
        mutating: true
      });
      $("silver-registration-form").reset();
      selectedCandidate = null;
      $("silver-selected-photo").classList.add("hidden");
      $("silver-registration-submit").disabled = true;
      setMessage("silver-registration-message", "Silver Jubilee information registered.", "success");
      await refreshRegistrations();
    } catch (error) {
      setMessage("silver-registration-message", error.message || "Registration failed.", "error");
    } finally {
      setAdminBusy(false);
    }
  });

  function safeRecordId(record) {
    const id = Number(record && record.id);
    return Number.isSafeInteger(id) && id > 0 ? id : null;
  }

  function recordImageName(record) {
    const path = typeof record.image_path === "string" ? record.image_path : "";
    const name = path.split(/[\\/]/).filter(Boolean).pop();
    return name || "Registered photograph";
  }

  async function refreshRegistrations() {
    if (!currentUser || currentUser.role !== "admin") return;
    try {
      const data = await request("/api/secure-info/registrations?limit=100&offset=0");
      renderRegistrations(Array.isArray(data) ? data : []);
    } catch (error) {
      setMessage("silver-registration-message", error.message || "Registration list could not be loaded.", "error");
    }
  }

  function renderRegistrations(records) {
    const container = $("silver-records");
    container.replaceChildren();
    if (!records.length) {
      const empty = document.createElement("div");
      empty.className = "silver-empty";
      empty.textContent = "No Silver Jubilee records have been registered.";
      container.appendChild(empty);
      return;
    }
    for (const record of records) {
      const id = safeRecordId(record);
      if (!id || typeof record.title !== "string") continue;
      const card = document.createElement("article");
      card.className = "silver-record";
      const heading = document.createElement("div");
      heading.className = "silver-record-heading";
      const details = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = record.title;
      const imageName = document.createElement("small");
      imageName.textContent = recordImageName(record);
      details.append(title, imageName);
      const actions = document.createElement("div");
      actions.className = "silver-record-actions";
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "secondary-button";
      edit.textContent = "Edit";
      edit.addEventListener("click", () => toggleRecordEditor(card, record, id));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "secondary-button";
      remove.textContent = "Delete";
      remove.addEventListener("click", () => deleteRecord(id, record.title));
      actions.append(edit, remove);
      heading.append(details, actions);
      card.appendChild(heading);
      container.appendChild(card);
    }
  }

  function toggleRecordEditor(card, record, id) {
    const existing = card.querySelector(".silver-record-edit");
    if (existing) {
      existing.remove();
      return;
    }
    const form = document.createElement("form");
    form.className = "silver-record-edit";
    const titleLabel = document.createElement("label");
    titleLabel.textContent = "Information title";
    const titleInput = document.createElement("input");
    titleInput.maxLength = 200;
    titleInput.required = true;
    titleInput.value = record.title;
    const informationLabel = document.createElement("label");
    informationLabel.textContent = "Silver Jubilee information";
    const information = document.createElement("textarea");
    information.maxLength = 100000;
    information.required = true;
    information.placeholder = "Enter replacement information; existing plaintext is not returned by the API.";
    const save = document.createElement("button");
    save.type = "submit";
    save.className = "primary";
    save.textContent = "Save update";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "secondary-button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => form.remove());
    form.append(titleLabel, titleInput, informationLabel, information, save, cancel);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      save.disabled = true;
      try {
        await request(`/api/secure-info/registrations/${id}`, {
          method: "PUT",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({title: titleInput.value.trim(), information: information.value.trim()}),
          mutating: true
        });
        setMessage("silver-registration-message", "Registration updated.", "success");
        await refreshRegistrations();
      } catch (error) {
        setMessage("silver-registration-message", error.message || "Registration could not be updated.", "error");
        save.disabled = false;
      }
    });
    card.appendChild(form);
    information.focus();
  }

  async function deleteRecord(id, title) {
    if (!window.confirm(`Delete the Silver Jubilee registration “${title}”? This cannot be undone.`)) return;
    try {
      await request(`/api/secure-info/registrations/${id}`, {method: "DELETE", mutating: true});
      setMessage("silver-registration-message", "Registration deleted.", "success");
      await refreshRegistrations();
    } catch (error) {
      setMessage("silver-registration-message", error.message || "Registration could not be deleted.", "error");
    }
  }

  $("silver-refresh-records").addEventListener("click", refreshRegistrations);
  initializeSession();
})();
