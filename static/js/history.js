// history.js — powers the /history page

const PAGE_SIZE = 15;
let currentFilter = "";
let currentOffset = 0;
let totalCount = 0;

function formatDate(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16) + " UTC";
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("is-active"));
      chip.classList.add("is-active");
      currentFilter = chip.dataset.filter;
      currentOffset = 0;
      document.getElementById("historyList").innerHTML = "";
      loadHistory();
    });
  });

  document.getElementById("loadMoreBtn").addEventListener("click", loadHistory);
  loadHistory();
});

async function loadHistory() {
  const params = new URLSearchParams({ limit: PAGE_SIZE, offset: currentOffset });
  if (currentFilter) params.append("prediction", currentFilter);

  const res = await fetch(`/api/history?${params.toString()}`);
  const data = await res.json();
  if (!data.ok) return;

  totalCount = data.total;
  const list = document.getElementById("historyList");
  const emptyEl = document.getElementById("historyEmpty");

  if (currentOffset === 0 && data.results.length === 0) {
    emptyEl.hidden = false;
    document.getElementById("loadMoreBtn").hidden = true;
    return;
  }
  emptyEl.hidden = true;

  data.results.forEach((row) => {
    const item = document.createElement("div");
    item.className = "history-row";
    const predClass = "badge-" + row.prediction.toLowerCase();
    item.innerHTML = `
      <div class="history-row-main">
        <span class="history-badge ${predClass}">${row.prediction}</span>
        <div>
          <div class="history-row-title">${row.sample_label || "Unlabeled sample"} <span class="history-source">· ${row.source}</span></div>
          <div class="history-row-meta">${formatDate(row.created_at)} &middot; Confidence ${row.confidence.toFixed(1)}% &middot; Top factor: ${row.top_feature}</div>
        </div>
      </div>
      <div class="history-row-params">
        pH ${row.ph.toFixed(2)} &middot; Cl ${row.chlorine.toFixed(2)} mg/L &middot; Hard ${row.hardness.toFixed(0)} ppm &middot; NO3 ${row.nitrate.toFixed(2)} mg/L
      </div>
      <a class="btn btn-secondary" href="/api/report/${row.id}">Download PDF</a>
    `;
    list.appendChild(item);
  });

  currentOffset += data.results.length;
  document.getElementById("loadMoreBtn").hidden = currentOffset >= totalCount;
}
