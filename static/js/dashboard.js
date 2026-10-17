// dashboard.js — powers the main HydroScan AI dashboard page

const FEATURE_META = {
  ph:        { label: "pH",        unit: "" },
  chlorine:  { label: "Chlorine",  unit: "mg/L" },
  hardness:  { label: "Hardness",  unit: "ppm" },
  nitrate:   { label: "Nitrate",   unit: "mg/L" },
};
const STATUS_COLOR = { Safe: "#1FAE6B", Caution: "#E0A419", Unsafe: "#E0453F" };

let lastAnalysisId = null;
let lastStripImagePath = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initManualForm();
  initStripUpload();
  initResultActions();
});

// ============================= Tabs ============================= //
function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => { t.classList.remove("is-active"); t.setAttribute("aria-selected", "false"); });
      tab.classList.add("is-active");
      tab.setAttribute("aria-selected", "true");
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));
      document.querySelector(`.tab-panel[data-panel="${tab.dataset.tab}"]`).classList.add("is-active");
    });
  });
}

// ============================= Manual entry ============================= //
function initManualForm() {
  const form = document.getElementById("predictForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const params = {
      ph: document.getElementById("f_ph").value,
      chlorine: document.getElementById("f_chlorine").value,
      hardness: document.getElementById("f_hardness").value,
      nitrate: document.getElementById("f_nitrate").value,
      sample_label: document.getElementById("f_label").value,
      source: "manual",
    };
    await runPrediction(params, document.getElementById("analyzeBtn"), document.getElementById("formErrors"));
  });
}

// ============================= Strip scan ============================= //
function initStripUpload() {
  const drop = document.getElementById("stripDrop");
  const fileInput = document.getElementById("stripFile");
  const chooseBtn = document.getElementById("stripChooseBtn");

  chooseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleStripFile(fileInput.files[0]);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.style.borderColor = "var(--teal)"; })
  );
  ["dragleave", "drop"].forEach((evt) =>
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.style.borderColor = ""; })
  );
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleStripFile(file);
  });

  document.getElementById("analyzeStripBtn").addEventListener("click", async () => {
    const params = {
      ph: document.getElementById("s_ph").value,
      chlorine: document.getElementById("s_chlorine").value,
      hardness: document.getElementById("s_hardness").value,
      nitrate: document.getElementById("s_nitrate").value,
      sample_label: document.getElementById("s_label").value,
      source: "strip",
      strip_image_path: lastStripImagePath,
    };
    await runPrediction(params, document.getElementById("analyzeStripBtn"), document.getElementById("stripFormErrors"));
  });
}

async function handleStripFile(file) {
  const promptEl = document.getElementById("stripUploadPrompt");
  const previewImg = document.getElementById("stripPreviewImg");
  const scanning = document.getElementById("stripScanning");
  const resultEl = document.getElementById("stripResult");
  const errEl = document.getElementById("stripError");

  errEl.hidden = true;
  resultEl.hidden = true;

  const reader = new FileReader();
  reader.onload = () => {
    previewImg.src = reader.result;
    previewImg.hidden = false;
    promptEl.hidden = true;
  };
  reader.readAsDataURL(file);

  scanning.hidden = false;

  const fd = new FormData();
  fd.append("image", file);

  try {
    const res = await fetch("/api/strip-scan", { method: "POST", body: fd });
    const data = await res.json();
    scanning.hidden = true;

    if (!data.ok) {
      errEl.hidden = false;
      errEl.textContent = data.error || "Could not analyze this image.";
      return;
    }

    lastStripImagePath = data.strip_image_path;
    document.getElementById("stripNote").textContent = data.note;

    const badge = document.getElementById("stripConfidenceBadge");
    const conf = data.confidence;
    badge.textContent = `Strip validity: ${conf.toFixed(0)}%`;
    badge.className = "strip-confidence-badge " + (conf >= 85 ? "is-high" : "is-medium");
    renderStripMethod(data);
    document.getElementById("s_ph").value = data.estimated_values.ph;
    document.getElementById("s_chlorine").value = data.estimated_values.chlorine;
    document.getElementById("s_hardness").value = data.estimated_values.hardness;
    document.getElementById("s_nitrate").value = data.estimated_values.nitrate;

    const padStrip = document.getElementById("padStrip");
    padStrip.innerHTML = "";
    data.pad_thumbnails.forEach((thumb, i) => {
      const key = Object.keys(FEATURE_META)[i];
      const wrap = document.createElement("div");
      wrap.className = "pad";
      wrap.innerHTML = `${thumb ? `<img src="${thumb}" alt="${FEATURE_META[key].label} pad">` : ""}<span>${FEATURE_META[key].label}</span>`;
      padStrip.appendChild(wrap);
    });

    resultEl.hidden = false;
  } catch (err) {
    scanning.hidden = true;
    errEl.hidden = false;
    errEl.textContent = "Network error while analyzing the image. Please try again.";
  }
}

// ============================= Prediction call ============================= //
async function runPrediction(rawParams, btn, errEl) {
  errEl.hidden = true;

  const btnLabel = btn.querySelector(".btn-label");
  const btnSpinner = btn.querySelector(".btn-spinner");
  btn.disabled = true;
  if (btnSpinner) btnSpinner.hidden = false;
  if (btnLabel) btnLabel.style.opacity = "0.6";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rawParams),
    });
    const data = await res.json();

    if (!data.ok) {
      errEl.hidden = false;
      errEl.innerHTML = "<ul>" + (data.errors || ["Something went wrong."]).map((e) => `<li>${e}</li>`).join("") + "</ul>";
      return;
    }

    lastAnalysisId = data.analysis_id;
    renderResults(data);
    document.getElementById("resultsEmpty").hidden = true;
    document.getElementById("resultsPanel").hidden = false;
    document.getElementById("resultsPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    hsToast("Analysis complete");
  } catch (err) {
    errEl.hidden = false;
    errEl.textContent = "Network error. Please check your connection and try again.";
  } finally {
    btn.disabled = false;
    if (btnSpinner) btnSpinner.hidden = true;
    if (btnLabel) btnLabel.style.opacity = "1";
  }
}

// ============================= Render results ============================= //
function renderResults(data) {
  const pred = data.prediction; // Safe | Caution | Unsafe
  const predClass = "is-" + pred.toLowerCase();

  const banner = document.getElementById("predictionBanner");
  banner.className = "card prediction-banner " + predClass;
  document.getElementById("predictionLabel").textContent = pred.toUpperCase();
  document.getElementById("predictionSummary").textContent = data.plain_summary;
  renderModelMethod(data);

  // Confidence ring
  const circumference = 326.7;
  const offset = circumference * (1 - data.confidence / 100);
  const ringFill = document.getElementById("ringFill");
  ringFill.style.strokeDashoffset = circumference; // reset
  requestAnimationFrame(() => { ringFill.style.strokeDashoffset = offset; });
  document.getElementById("confidenceValue").textContent = data.confidence.toFixed(1) + "%";

  // Risk marker: blended position from class probabilities (0=Safe .. 100=Unsafe)
  // probs.* are already percentages (0-100) summing to ~100.
  const probs = data.probabilities || {};
  const weighted = ((probs.Caution || 0) * 1 + (probs.Unsafe || 0) * 2) / 100; // 0..2 scale
  const positionPct = Math.min(Math.max((weighted / 2) * 100, 0), 100); // clamp 0..100
  const marker = document.getElementById("riskMarker");
  marker.style.left = "0%";
  requestAnimationFrame(() => { marker.style.left = `calc(${positionPct.toFixed(1)}% - 1px)`; });

  // Parameter cards
  const paramGrid = document.getElementById("paramGrid");
  paramGrid.innerHTML = "";
  Object.keys(FEATURE_META).forEach((key) => {
    const meta = FEATURE_META[key];
    const status = data.parameter_statuses[key];
    const card = document.createElement("div");
    card.className = `param-card status-${status.toLowerCase()}`;
    card.innerHTML = `
      <div class="p-name">${meta.label}</div>
      <div class="p-value">${Number(data.params[key]).toFixed(2)} <span class="p-unit">${meta.unit}</span></div>
      <span class="p-status">${status}</span>
    `;
    paramGrid.appendChild(card);
  });

  // Attribution spectrum
  const spectrum = document.getElementById("attributionSpectrum");
  const legend = document.getElementById("attributionLegend");
  spectrum.innerHTML = "";
  legend.innerHTML = "";
  const ordered = Object.entries(data.contribution_percent).sort((a, b) => b[1] - a[1]);
  ordered.forEach(([key, pct]) => {
    const status = data.parameter_statuses[key];
    const color = STATUS_COLOR[status];
    const seg = document.createElement("div");
    seg.className = "spectrum-segment";
    seg.style.background = color;
    seg.style.flexBasis = "2%";
    seg.innerHTML = `<span class="seg-pct">${pct.toFixed(0)}%</span><span class="seg-name">${FEATURE_META[key].label}</span>`;
    spectrum.appendChild(seg);
    requestAnimationFrame(() => { seg.style.flexBasis = pct + "%"; });

    const li = document.createElement("li");
    li.innerHTML = `<span class="legend-dot" style="background:${color}"></span> ${FEATURE_META[key].label} — ${status} (${pct.toFixed(1)}%)`;
    legend.appendChild(li);
  });

  // XAI reasons
  const reasonsList = document.getElementById("reasonsList");
  reasonsList.innerHTML = data.reasons.map((r) => `<li>${r}</li>`).join("");

  // Causes
  const causesCard = document.getElementById("causesCard");
  const causesList = document.getElementById("causesList");
  if (data.causes && data.causes.length) {
    causesCard.hidden = false;
    causesList.innerHTML = data.causes.map((c) => `<li>${c}</li>`).join("");
  } else {
    causesCard.hidden = true;
  }

  // Recommendations
  document.getElementById("recommendationsList").innerHTML =
    data.recommendations.map((r) => `<li>${r}</li>`).join("");

  // Summary
  document.getElementById("summaryDl").innerHTML = `
    <div><dt>Analysis ID</dt><dd>#${data.analysis_id}</dd></div>
    <div><dt>Date &amp; Time</dt><dd>${data.created_at}</dd></div>
    <div><dt>Top Factor</dt><dd>${data.top_influential_parameter}</dd></div>
    <div><dt>Sample</dt><dd>${data.sample_label || "—"}</dd></div>
  `;
}

// ============================= Confidence transparency ============================= //
function pct(x, d = 1) { return (x * 100).toFixed(d) + "%"; }

function renderModelMethod(data) {
  const d = data.confidence_detail, v = d.validation || {};
  const probs = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([k, p]) => `<tr><td>${k}</td><td class="num">${p.toFixed(1)}%</td></tr>`).join("");
  document.getElementById("modelMethodBody").innerHTML = `
    <p>The confidence is the model's probability for the class it chose, computed in two steps:</p>
    <pre class="formula">1. pₖ = (1/T) · Σₜ pₜ,ₖ        mean vote of T = 200 decision trees
2. qₖ = softmax( ln pₖ / τ )     calibration, τ fitted on out-of-fold data
   confidence = maxₖ qₖ</pre>
    <table class="mini-table">
      <tr><td>Forest vote fraction  p</td><td class="num">${d.raw_vote_fraction.toFixed(1)}%</td></tr>
      <tr><td>Temperature  τ</td><td class="num">${d.temperature_tau.toFixed(3)}</td></tr>
      <tr><td><strong>Calibrated confidence  q</strong></td><td class="num"><strong>${d.calibrated.toFixed(1)}%</strong></td></tr>
    </table>
    <table class="mini-table"><caption>All class probabilities</caption>${probs}</table>
    <p class="method-note"><strong>Validation on unseen data</strong> (${v.n_records || "?"} records, 80/20 split + 5-fold CV):
      accuracy ${v.test_accuracy != null ? pct(v.test_accuracy) : "—"} (5-fold CV ${v.cv5_accuracy_mean != null ? pct(v.cv5_accuracy_mean) : "—"} ± ${v.cv5_accuracy_std != null ? pct(v.cv5_accuracy_std) : "—"}),
      calibration error (ECE) ${v.ece != null ? pct(v.ece) : "—"}. “When the model says 90 %, it is right about 90 % of the time.”
      Trained on a synthetic dataset, so this is a measure of internal consistency, not of real-world accuracy.</p>`;
}

function renderStripMethod(data) {
  const d = data.confidence_detail, v = d.validation || {};
  const rows = d.breakdown.map((r) => `<tr><td>${r.label}</td><td class="num">${r.value}</td>
      <td class="num ${r.pass_probability < 0.6 ? "bad" : ""}">${(r.pass_probability * 100).toFixed(0)}%</td></tr>`).join("");
  document.getElementById("stripMethodBody").innerHTML = `
    <p>Validity = probability that this is a clear, usable 4-pad strip. Every criterion must pass (weakest-link rule):</p>
    <pre class="formula">P(valid) = Π fⱼ ,   fⱼ = 1 / (1 + e^−(aⱼ + bⱼ·zⱼ))
zⱼ = (measured value − μⱼ) / sⱼ</pre>
    <table class="mini-table"><tr><th>Criterion</th><th class="num">Measured</th><th class="num">Passes</th></tr>${rows}
      <tr><td colspan="2"><strong>P(valid) = product</strong></td><td class="num"><strong>${data.confidence.toFixed(1)}%</strong></td></tr></table>
    <p class="method-note">Accepted when P ≥ ${d.threshold}% (a false accept is costed 1.5× a false reject → 1.5/(1.5+1)).
      Validated on ${v.corpus || "held-out photos"}: correct ${v.overall_accuracy != null ? pct(v.overall_accuracy) : "—"},
      valid strips accepted ${v.true_accept_rate != null ? pct(v.true_accept_rate) : "—"},
      invalid photos wrongly accepted ${v.false_accept_rate != null ? pct(v.false_accept_rate) : "—"},
      ROC-AUC ${v.roc_auc ?? "—"}. Synthetic photos only — see docs/CONFIDENCE_METHODOLOGY.md.</p>`;
}

// ============================= Result actions ============================= //
function initResultActions() {
  document.getElementById("downloadReportBtn").addEventListener("click", () => {
    if (!lastAnalysisId) return;
    window.location.href = `/api/report/${lastAnalysisId}`;
  });
  document.getElementById("newAnalysisBtn").addEventListener("click", () => {
    // A full reload guarantees everything resets — the results panel, the
    // manual entry form, and (importantly) the strip-scan tab's uploaded
    // photo preview, pad thumbnails, and confidence badge, which previously
    // stuck around from the last scan when only the form was reset.
    window.location.reload();
  });
}
