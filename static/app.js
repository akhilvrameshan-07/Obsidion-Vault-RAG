/**
 * app.js — Vault RAG front-end logic
 *
 * Features:
 * 1. POST /ask — answer + sources displayed; answer text set via textContent (safe)
 * 2. In-browser chat history (JS array, cleared on page refresh)
 * 3. Sync Vault button — calls POST /sync, shows inline summary or error
 * 4. Independent loading states: ask button and sync button are separate
 */

const ASK_ENDPOINT  = "/ask";
const SYNC_ENDPOINT = "/sync";

// ---------- DOM refs ----------
const input        = document.getElementById("question-input");
const submitBtn    = document.getElementById("submit-btn");
const respArea     = document.getElementById("response-area");
const loadingEl    = document.getElementById("loading");
const errorBox     = document.getElementById("error-box");
const answerBlock  = document.getElementById("answer-block");
const answerText   = document.getElementById("answer-text");
const sourcesBlock = document.getElementById("sources-block");
const sourcesList  = document.getElementById("sources-list");
const chatHistory  = document.getElementById("chat-history");
const syncBtn      = document.getElementById("sync-btn");
const syncStatus   = document.getElementById("sync-status");

// ---------- In-memory chat history ----------
/** @type {Array<{question: string, answer: string, sources: string[]}>} */
const history = [];

/** Append a completed exchange to the visible history panel. */
function pushHistory(question, answer, sources) {
  history.push({ question, answer, sources });

  const entry = document.createElement("div");
  entry.className = "history-entry";

  // Question row
  const qEl = document.createElement("div");
  qEl.className = "history-q";
  qEl.textContent = question;           // safe: textContent only
  entry.appendChild(qEl);

  // Answer
  const aEl = document.createElement("div");
  aEl.className = "history-a";
  aEl.textContent = answer;             // safe: textContent only
  entry.appendChild(aEl);

  // Sources (chips)
  if (sources && sources.length > 0) {
    const chipsRow = document.createElement("div");
    chipsRow.className = "history-sources";
    sources.forEach(title => {
      const chip = document.createElement("span");
      chip.className = "history-source-chip";
      chip.textContent = title;         // safe: textContent only
      chipsRow.appendChild(chip);
    });
    entry.appendChild(chipsRow);
  }

  // Insert before the input area (append to top of history div — we
  // use flex-direction:column so entries appear in chronological order)
  chatHistory.appendChild(entry);

  // Scroll the new entry into view smoothly
  entry.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------- Ask helpers ----------

function setTextSafe(el, text) {
  el.textContent = text;
}

function showLoading() {
  respArea.classList.add("visible");
  loadingEl.classList.add("visible");
  errorBox.classList.remove("visible");
  answerBlock.classList.remove("visible");
  sourcesBlock.classList.remove("visible");
}

function showError(message) {
  loadingEl.classList.remove("visible");
  errorBox.classList.add("visible");
  setTextSafe(errorBox, message);
  answerBlock.classList.remove("visible");
  sourcesBlock.classList.remove("visible");
}

function showResult(answer, sources) {
  loadingEl.classList.remove("visible");
  errorBox.classList.remove("visible");

  setTextSafe(answerText, answer);
  answerBlock.classList.add("visible");

  // Render sources as a clean vertical list
  sourcesList.innerHTML = "";           // safe: we only append our own elements
  if (sources && sources.length > 0) {
    sources.forEach(title => {
      const li = document.createElement("li");
      setTextSafe(li, title);           // safe: textContent only
      sourcesList.appendChild(li);
    });
    sourcesBlock.classList.add("visible");
  } else {
    sourcesBlock.classList.remove("visible");
  }
}

// ---------- Ask handler ----------

async function handleSubmit() {
  const question = input.value.trim();
  if (!question) {
    input.focus();
    return;
  }

  submitBtn.disabled = true;
  showLoading();

  try {
    const response = await fetch(ASK_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let detail = `Server error ${response.status}`;
      try {
        const data = await response.json();
        if (data.detail) detail = data.detail;
      } catch (_) { /* ignore */ }
      showError(`⚠️ ${detail}`);
      return;
    }

    const data = await response.json();
    showResult(data.answer, data.sources);

    // Move this exchange into the history panel and clear the live area
    pushHistory(question, data.answer, data.sources);
    input.value = "";
    // Hide the live response panel — the answer is now in history
    respArea.classList.remove("visible");
    answerBlock.classList.remove("visible");
    sourcesBlock.classList.remove("visible");

  } catch (err) {
    if (err instanceof TypeError) {
      showError("⚠️ Cannot reach the backend. Is uvicorn running on http://localhost:8000?");
    } else {
      showError(`⚠️ Unexpected error: ${err.message}`);
    }
  } finally {
    submitBtn.disabled = false;
    input.focus();
  }
}

// ---------- Sync handler ----------

async function handleSync() {
  syncBtn.disabled = true;
  syncStatus.textContent = "Syncing…";
  syncStatus.className = "";

  try {
    const response = await fetch(SYNC_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (response.status === 409) {
      syncStatus.textContent = "⚠️ Sync already in progress — please wait.";
      syncStatus.className = "error";
      return;
    }

    if (!response.ok) {
      let detail = `Error ${response.status}`;
      try {
        const d = await response.json();
        if (d.detail) detail = d.detail;
      } catch (_) { /* ignore */ }
      syncStatus.textContent = `⚠️ ${detail}`;
      syncStatus.className = "error";
      return;
    }

    const data = await response.json();
    const { added, changed, deleted, unchanged } = data;
    syncStatus.textContent =
      `Synced: ${added} added, ${changed} changed, ${deleted} deleted, ${unchanged} unchanged.`;
    syncStatus.className = "success";

  } catch (err) {
    syncStatus.textContent = "⚠️ Cannot reach the backend for sync.";
    syncStatus.className = "error";
  } finally {
    syncBtn.disabled = false;
  }
}

// ---------- Event listeners ----------

submitBtn.addEventListener("click", handleSubmit);
syncBtn.addEventListener("click", handleSync);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleSubmit();
});
