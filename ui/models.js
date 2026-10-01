/*
 * Header "Download models" button - the desktop counterpart of the Android model badge.
 *
 * Always visible, so the install state is never a mystery:
 *   ✓ Models ready            everything the active profile needs is on disk
 *   ⬇ Download models · 4.4 GB something is missing; click to fetch it
 *   ■ Stop · 42%              downloading; click to stop (progress is kept)
 *   ⟳ Resume / Retry          stopped or failed; click to carry on
 *
 * "Everything" means the llama.cpp runtime, the MobileCLIP embedder and the active
 * profile's VLM weights - see src/videorag/downloader.py for why each one matters. A
 * slim strip under the header explains what is missing and shows progress; it hides
 * itself once there is nothing to say.
 *
 * Self-contained: it adds its own nodes and styles and does not touch app.js.
 */
(function () {
  "use strict";

  var btn, strip, stripMsg, bar, fill;
  var polling = null;
  var last = null;
  // Whether THIS page watched a download run. The server keeps `done` set after a
  // download finishes, so keying the "all installed" note off that alone replayed it
  // on every later page load - announcing a model server start that happened long ago.
  var sawRunning = false;

  var CSS = [
    ".mdl-btn{display:flex;align-items:center;gap:6px;padding:5px 11px;",
    "  border-radius:var(--radius-sm,6px);font:600 .76rem var(--font-sans,system-ui);",
    "  cursor:pointer;white-space:nowrap;position:relative;overflow:hidden;flex-shrink:0;",
    "  border:1px solid transparent;transition:background .2s,color .2s,border-color .2s}",
    ".mdl-btn[hidden]{display:none!important}",
    ".mdl-btn .mdl-fill{position:absolute;left:0;top:0;bottom:0;width:0;",
    "  background:rgba(2,132,199,.16);transition:width .4s ease;pointer-events:none}",
    ".mdl-btn span{position:relative}",
    ".mdl-ready{background:var(--green-bg,#f0fdf4);color:var(--green-accent,#16a34a);",
    "  border-color:#bbf7d0;cursor:default}",
    ".mdl-missing{background:var(--primary-accent,#0284c7);color:#fff}",
    ".mdl-missing:hover{background:var(--primary-hover,#0369a1)}",
    ".mdl-running{background:var(--blue-bg-subtle,#f0f9ff);color:var(--blue-primary,#0284c7);",
    "  border-color:var(--blue-border,#bae6fd)}",
    ".mdl-running:hover{border-color:var(--red-accent,#dc2626);color:var(--red-accent,#dc2626)}",
    ".mdl-error{background:var(--red-bg,#fef2f2);color:var(--red-accent,#dc2626);border-color:#fecaca}",
    ".mdl-strip{display:flex;align-items:center;gap:14px;margin:8px 0 0;padding:9px 14px;",
    "  background:var(--blue-bg-subtle,#f0f9ff);border:1px solid var(--blue-border,#bae6fd);",
    "  border-radius:var(--radius-md,10px);font:.8rem/1.45 var(--font-sans,system-ui);",
    "  color:var(--text-body,#334155)}",
    ".mdl-strip[hidden]{display:none!important}",
    ".mdl-strip.mdl-strip-error{background:var(--red-bg,#fef2f2);border-color:#fecaca}",
    ".mdl-strip .mdl-msg{flex:1;min-width:0}",
    ".mdl-strip .mdl-bar{flex:0 0 220px;height:6px;background:#dbeafe;border-radius:4px;overflow:hidden}",
    ".mdl-strip .mdl-bar[hidden]{display:none}",
    ".mdl-strip .mdl-bar>div{height:100%;width:0;background:var(--primary-accent,#0284c7);",
    "  transition:width .4s ease}"
  ].join("\n");

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function gb(n) { return (n / 1e9).toFixed(1) + " GB"; }
  function mb(n) { return Math.round(n / 1e6) + " MB"; }
  function size(n) { return n >= 1e9 ? gb(n) : mb(n); }

  function build() {
    var style = el("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    btn = el("button", "mdl-btn");
    btn.type = "button";
    btn.hidden = true;
    btn.appendChild(el("div", "mdl-fill"));
    btn.appendChild(el("span"));
    btn.addEventListener("click", onClick);

    var stats = document.querySelector(".header-stats");
    if (stats) stats.insertBefore(btn, stats.firstChild);
    else document.body.insertBefore(btn, document.body.firstChild);

    strip = el("div", "mdl-strip");
    strip.hidden = true;
    stripMsg = el("div", "mdl-msg");
    bar = el("div", "mdl-bar");
    fill = el("div");
    bar.appendChild(fill);
    strip.appendChild(stripMsg);
    strip.appendChild(bar);

    var header = document.querySelector("header.header-bar");
    if (header && header.parentNode) header.parentNode.insertBefore(strip, header.nextSibling);
    else document.body.insertBefore(strip, document.body.firstChild);
  }

  function setBtn(kind, label, title, pct) {
    btn.hidden = false;
    btn.className = "mdl-btn mdl-" + kind;
    btn.lastChild.textContent = label;
    btn.title = title || "";
    btn.firstChild.style.width = (pct == null ? 0 : pct) + "%";
  }

  function setStrip(text, pct, isError) {
    if (!text) { strip.hidden = true; return; }
    strip.hidden = false;
    strip.classList.toggle("mdl-strip-error", !!isError);
    stripMsg.textContent = text;
    bar.hidden = pct == null;
    if (pct != null) fill.style.width = pct.toFixed(1) + "%";
  }

  function missingList(st) {
    return (st.missing || []).map(function (m) {
      return m.name + " (" + size(m.bytes) + ")";
    });
  }

  function render(st) {
    last = st;
    if (!st || !st.configured) {          // no manifest: nothing we can offer
      btn.hidden = true;
      setStrip(null);
      return;
    }
    var dl = st.download || {};

    if (dl.running) {
      sawRunning = true;
      var pct = dl.file_bytes ? Math.min(100, 100 * dl.file_done / dl.file_bytes) : 0;
      var step = " (" + ((dl.index || 0) + 1) + " of " + (dl.total || 1) + ")";
      var what =
        dl.phase === "verify"  ? "Verifying " + dl.name + step + "…" :
        dl.phase === "extract" ? "Unpacking " + dl.name + step + "…" :
        "Downloading " + dl.name + step + " — " + mb(dl.file_done) + " / " + mb(dl.file_bytes);
      setBtn("running", "■ Stop · " + Math.round(pct) + "%",
             "Stop the download. Progress is kept; press again to resume.", pct);
      setStrip(what, dl.phase === "download" || !dl.phase ? pct : null);
      ensurePolling();
      return;
    }

    if (st.ready) {
      stopPolling();
      var subs = (st.substitutes || []).length;
      setBtn("ready", "✓ Models ready",
             "Everything the " + (st.profile || "active") + " profile needs is installed" +
             (subs ? " (" + subs + " file(s) are a locally supplied build, left as they are)." : "."));
      if (st.restart_needed) {
        setStrip("Models are installed, but the embedder loaded a fallback before MobileCLIP " +
                 "was present. Restart the app so search uses MobileCLIP.", null, true);
      } else if (dl.done && sawRunning) {
        sawRunning = false;
        setStrip("All models installed — the model server is starting. You can index and ask questions now.", null);
        setTimeout(function () { if (last && last.ready) setStrip(null); }, 8000);
      } else {
        setStrip(null);
      }
      return;
    }

    var total = size(st.missing_bytes || 0);
    var items = missingList(st);
    if (dl.error) {
      setBtn("error", "⟳ Retry download", dl.error);
      setStrip("Model download did not finish: " + dl.error, null, true);
    } else if (dl.cancelled) {
      setBtn("missing", "⟳ Resume download · " + total,
             "Still needed: " + items.join(", "));
      setStrip("Download stopped. Press Resume to carry on — what was already downloaded is kept.", null);
    } else {
      setBtn("missing", "⬇ Download models · " + total,
             "Missing from the project folder: " + items.join(", "));
      // State only what is checkable. "Cannot answer until..." was true for the runtime
      // and the VLM weights but not for MobileCLIP, which open_clip can still pull into
      // ~/.cache on its own - so on a machine like that the claim was simply false.
      setStrip("Missing from the project folder: " + items.join(", ") + ". " +
               "One-time download of " + total + ".", null);
    }
  }

  function refresh() {
    return fetch("/api/models/status")
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function () { /* server restarting; next poll will catch up */ });
  }

  function ensurePolling() {
    if (!polling) polling = setInterval(refresh, 1000);
  }

  function stopPolling() {
    if (polling) { clearInterval(polling); polling = null; }
  }

  function onClick() {
    if (!last || !last.configured) return;
    var dl = last.download || {};
    if (dl.running) {
      btn.disabled = true;
      fetch("/api/models/cancel", { method: "POST" })
        .finally(function () { btn.disabled = false; refresh(); });
      return;
    }
    if (last.ready) { refresh(); return; }
    btn.disabled = true;
    setBtn("running", "Starting…", "");
    fetch("/api/models/download", { method: "POST" })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || r.status); });
        return r.json();
      })
      .then(function () { ensurePolling(); return refresh(); })
      .catch(function (e) {
        setBtn("error", "⟳ Retry download", String(e.message || e));
        setStrip("Could not start the download: " + (e.message || e), null, true);
      })
      .finally(function () { btn.disabled = false; });
  }

  document.addEventListener("DOMContentLoaded", function () {
    build();
    refresh();
    // A profile switch changes what is required (4B vs 2B weights).
    document.addEventListener("click", function (ev) {
      if (ev.target.closest && ev.target.closest(".btn-profile-toggle")) {
        setTimeout(refresh, 800);
        setTimeout(refresh, 4000);
      }
    });
    window.addEventListener("focus", refresh);
  });
})();
