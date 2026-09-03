/*
 * First-run model download banner.
 *
 * The desktop VLM weights (2.3-3.3 GB depending on profile) are not shipped with the
 * app; they are fetched from the model server on demand. This polls /api/models/status
 * and, when weights are missing, shows a banner with a Download button that kicks off
 * /api/models/download and reports progress. Fully self-contained so it does not touch
 * the main app.js closure.
 */
(function () {
  "use strict";

  var banner, msg, btn, bar, barFill;
  var polling = null;

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }

  function build() {
    banner = el("div", { id: "model-dl-banner" });
    banner.style.cssText =
      "display:none;position:fixed;top:0;left:0;right:0;z-index:9999;" +
      "background:#0b1f3a;color:#e8f0ff;border-bottom:2px solid #2b6cff;" +
      "padding:10px 16px;font:14px/1.4 system-ui,sans-serif;" +
      "display:none;align-items:center;gap:14px;box-shadow:0 2px 10px rgba(0,0,0,.4)";
    msg = el("span", null, "");
    msg.style.flex = "1";
    btn = el("button", null, "Download local models");
    btn.style.cssText =
      "background:#2b6cff;color:#fff;border:0;border-radius:6px;padding:7px 14px;" +
      "font-weight:600;cursor:pointer";
    btn.addEventListener("click", startDownload);
    bar = el("div");
    bar.style.cssText =
      "display:none;flex:2;height:8px;background:#12305a;border-radius:5px;overflow:hidden";
    barFill = el("div");
    barFill.style.cssText = "height:100%;width:0%;background:#3ba0ff;transition:width .3s";
    bar.appendChild(barFill);
    banner.appendChild(msg);
    banner.appendChild(bar);
    banner.appendChild(btn);
    document.body.appendChild(banner);
  }

  function show() { banner.style.display = "flex"; }
  function hide() { banner.style.display = "none"; }
  function gb(n) { return (n / 1e9).toFixed(2) + " GB"; }
  function mb(n) { return Math.round(n / 1e6) + " MB"; }

  function render(st) {
    if (!st || !st.configured) { hide(); return; }        // no server → stay silent
    if (st.ready) {
      if (polling) { clearInterval(polling); polling = null; }
      hide();
      return;
    }
    show();
    var dl = st.download || {};
    if (dl.running) {
      btn.style.display = "none";
      bar.style.display = "block";
      var pct = dl.file_bytes ? (100 * dl.file_done / dl.file_bytes) : 0;
      barFill.style.width = pct.toFixed(1) + "%";
      msg.textContent =
        "Downloading " + (dl.name || "") +
        " (" + ((dl.index || 0) + 1) + "/" + (dl.total || 1) + ") — " +
        mb(dl.file_done) + " / " + mb(dl.file_bytes);
      ensurePolling();
    } else if (dl.error) {
      btn.style.display = "inline-block";
      btn.textContent = "Retry download";
      bar.style.display = "none";
      msg.textContent = "Model download failed: " + dl.error;
    } else {
      btn.style.display = "inline-block";
      btn.textContent = "Download local models";
      bar.style.display = "none";
      var bytes = st.missing_bytes || 0;
      msg.textContent =
        "Local model weights required (" + gb(bytes) + ", one-time). " +
        "The app cannot answer until these are present.";
    }
  }

  function refresh() {
    fetch("/api/models/status")
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function () { /* server not up yet; ignore */ });
  }

  function ensurePolling() {
    if (polling) return;
    polling = setInterval(refresh, 1000);
  }

  function startDownload() {
    btn.disabled = true;
    btn.textContent = "Starting…";
    fetch("/api/models/download", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function () { btn.disabled = false; ensurePolling(); refresh(); })
      .catch(function (e) {
        btn.disabled = false;
        btn.textContent = "Retry download";
        msg.textContent = "Could not start download: " + e;
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    build();
    refresh();
  });
})();
