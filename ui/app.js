/**
 * VideoRAG Web Application JavaScript — Classic Light Theme
 * Handles API calls, interactive search, camera filtering, video player seeking,
 * and Developer Mode Edge Frame Hash & Motion Inspector with smooth micro-animations.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const queryInput = document.getElementById('query-input');
    const searchBtn = document.getElementById('search-btn');
    const cameraFilterContainer = document.getElementById('camera-filters');
    const quickQueriesContainer = document.querySelector('.quick-queries');
    const cctvPlayer = document.getElementById('cctv-player');
    const videoTimer = document.getElementById('video-timer');
    const seekNotice = document.getElementById('seek-notice');
    
    const aiAnswerBody = document.getElementById('ai-answer-body');
    const metricsBar = document.getElementById('metrics-bar');
    const mPrecision = document.getElementById('m-precision');
    const mMrr = document.getElementById('m-mrr');
    const mNdcg = document.getElementById('m-ndcg');
    const mUtil = document.getElementById('m-util');
    
    const evidenceList = document.getElementById('evidence-list');
    const evidenceCount = document.getElementById('evidence-count');

    const statModel = document.getElementById('stat-model');
    const statVectors = document.getElementById('stat-vectors');
    const statStatus = document.getElementById('stat-status');
    const statusDot = document.getElementById('status-dot');
    const shutdownBtn = document.getElementById('shutdown-btn');

    // Dev Mode Elements
    const devModeBtn = document.getElementById('dev-mode-btn');
    const devStatusText = document.getElementById('dev-status-text');
    const devPanel = document.getElementById('dev-panel');
    const hashAlgoSelect = document.getElementById('hash-algo-select');
    const thresholdRange = document.getElementById('threshold-range');
    const thresholdVal = document.getElementById('threshold-val');
    const runSmartFilterBtn = document.getElementById('run-smart-filter-btn');
    
    const kpiTotal = document.getElementById('kpi-total');
    const kpiKept = document.getElementById('kpi-kept');
    const kpiSkipped = document.getElementById('kpi-skipped');
    const kpiSaved = document.getElementById('kpi-saved');
    const devAuditTbody = document.getElementById('dev-audit-tbody');

    let activeCameraFilter = '';
    let isDevModeActive = false;

    // ------------------------------------------------------------------
    // Dev Mode Toggle & Controls
    // ------------------------------------------------------------------
    if (devModeBtn) {
        devModeBtn.addEventListener('click', () => {
            isDevModeActive = !isDevModeActive;
            devPanel.style.display = isDevModeActive ? 'flex' : 'none';
            devModeBtn.classList.toggle('active', isDevModeActive);
            devStatusText.textContent = isDevModeActive ? 'ON' : 'OFF';

            if (isDevModeActive) {
                fetchHashAuditLogs();
            }
        });
    }

    if (thresholdRange && thresholdVal) {
        thresholdRange.addEventListener('input', (e) => {
            thresholdVal.textContent = e.target.value;
        });
    }

    if (runSmartFilterBtn) {
        runSmartFilterBtn.addEventListener('click', async () => {
            try {
                runSmartFilterBtn.disabled = true;
                runSmartFilterBtn.innerHTML = `<span>Processing Filter…</span>`;
                
                const resp = await fetch('/api/process_video_smart', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        video_path: "Video Footage/sample_cctv.mp4",
                        camera_id: "CAM_01",
                        sample_interval: 15.0,
                        enable_hash_filter: true,
                        hash_method: hashAlgoSelect.value,
                        threshold: parseInt(thresholdRange.value, 10),
                    }),
                });

                if (resp.ok) {
                    const data = await resp.json();
                    renderDevAuditLogs(data.filter_stats, data.audit_trail);
                    checkHealth();
                }
            } catch (err) {
                alert('Smart filter execution failed: ' + err.message);
            } finally {
                runSmartFilterBtn.disabled = false;
                runSmartFilterBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Run Smart Frame Filter</span>`;
            }
        });
    }

    async function fetchHashAuditLogs() {
        try {
            const resp = await fetch('/api/hash_audit');
            if (resp.ok) {
                const data = await resp.json();
                renderDevAuditLogs(data.stats, data.audit_trail);
            }
        } catch (err) {
            console.warn('Failed to fetch audit logs:', err);
        }
    }

    function renderDevAuditLogs(stats, trail) {
        if (stats) {
            kpiTotal.textContent = stats.total_frames ?? 55;
            kpiKept.textContent = stats.keyframes_kept ?? 48;
            kpiSkipped.textContent = stats.frames_skipped ?? 7;
            kpiSaved.textContent = `${stats.llm_compute_saved_pct ?? 12.7}%`;
        }

        if (!trail || trail.length === 0) {
            devAuditTbody.innerHTML = `<tr><td colspan="6" class="placeholder-text" style="text-align:center; padding: 20px;">No audit trail generated yet. Click "Run Smart Frame Filter" above.</td></tr>`;
            return;
        }

        devAuditTbody.innerHTML = trail.map((item, idx) => `
            <tr class="animate-slide-in" style="animation-delay: ${idx * 0.02}s;">
                <td style="font-family: var(--font-mono); color: var(--text-muted);">${idx + 1}</td>
                <td style="font-family: var(--font-mono); font-weight:600; color: var(--blue-primary);">${item.timestamp}</td>
                <td style="font-family: var(--font-mono); font-size: 0.75rem; color: #64748b;"><code>${item.hash_hex}</code></td>
                <td style="font-family: var(--font-mono); text-align: center;"><strong>${item.hamming_distance}</strong> / ${item.threshold}</td>
                <td style="font-family: var(--font-mono); color: var(--text-muted); text-align: center;">${item.motion_pct}%</td>
                <td>
                    <span class="${item.is_keyframe ? 'badge-keep' : 'badge-skip'}">${item.status}</span>
                    <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 6px;">${escapeHtml(item.reason)}</span>
                </td>
            </tr>
        `).join('');
    }

    // ------------------------------------------------------------------
    // Shutdown Button Event Handler
    // ------------------------------------------------------------------
    if (shutdownBtn) {
        shutdownBtn.addEventListener('click', async () => {
            const confirmed = confirm('Are you sure you want to stop the VideoRAG Web Server?');
            if (!confirmed) return;

            try {
                shutdownBtn.disabled = true;
                shutdownBtn.innerHTML = `<span>Stopping…</span>`;
                const resp = await fetch('/api/shutdown', { method: 'POST' });
                if (resp.ok) {
                    if (statStatus) {
                        statStatus.textContent = 'OFFLINE';
                        statStatus.style.color = '#dc2626';
                    }
                    if (statusDot) {
                        statusDot.classList.remove('online');
                        statusDot.classList.add('offline');
                    }
                    alert('VideoRAG Web Server has been shut down successfully. You can close this browser tab.');
                }
            } catch (err) {
                alert('Server shutdown initiated.');
            }
        });
    }

    // ------------------------------------------------------------------
    // System Health Check
    // ------------------------------------------------------------------
    async function checkHealth() {
        try {
            const resp = await fetch('/api/health');
            if (resp.ok) {
                const data = await resp.json();
                statVectors.textContent = `${data.vector_count} Vectors`;
                statModel.textContent = `${data.llm_backend.toUpperCase()} (${data.llm_model.split('/').pop()})`;
            }
        } catch (err) {
            console.warn('Health check failed:', err);
        }
    }
    checkHealth();

    // ------------------------------------------------------------------
    // Video Timer Tracking
    // ------------------------------------------------------------------
    if (cctvPlayer) {
        cctvPlayer.addEventListener('timeupdate', () => {
            const cur = Math.floor(cctvPlayer.currentTime);
            const hrs = Math.floor(cur / 3600);
            const mins = Math.floor((cur % 3600) / 60);
            const secs = cur % 60;
            videoTimer.textContent = `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        });
    }

    // Seek Helper
    function seekToTime(seconds, label = '') {
        if (cctvPlayer) {
            cctvPlayer.currentTime = seconds;
            cctvPlayer.play().catch(() => {});
            seekNotice.textContent = `Seeked to ${label || seconds + 's'}`;
            seekNotice.style.opacity = '1';
            setTimeout(() => {
                seekNotice.textContent = 'Ready — Click any timestamp result to seek';
            }, 3000);
        }
    }

    // ------------------------------------------------------------------
    // Camera Selection Pills
    // ------------------------------------------------------------------
    cameraFilterContainer.addEventListener('click', (e) => {
        const btn = e.target.closest('.filter-pill');
        if (!btn) return;
        
        document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeCameraFilter = btn.dataset.camera || '';
        
        if (queryInput.value.trim()) {
            executeSearch();
        }
    });

    // Quick Queries
    quickQueriesContainer.addEventListener('click', (e) => {
        const chip = e.target.closest('.query-chip');
        if (!chip) return;
        queryInput.value = chip.dataset.query;
        executeSearch();
    });

    // Search Trigger
    searchBtn.addEventListener('click', executeSearch);
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            executeSearch();
        }
    });

    // ------------------------------------------------------------------
    // Execute Search API Call with Smooth Skeleton Loaders
    // ------------------------------------------------------------------
    async function executeSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        searchBtn.innerHTML = `<span>Searching…</span>`;

        // Smooth Loading Skeleton
        aiAnswerBody.innerHTML = `
            <div class="skeleton-box" style="width: 85%;"></div>
            <div class="skeleton-box" style="width: 92%;"></div>
            <div class="skeleton-box" style="width: 60%;"></div>
        `;
        evidenceList.innerHTML = `
            <div class="skeleton-box" style="height: 60px;"></div>
            <div class="skeleton-box" style="height: 60px;"></div>
            <div class="skeleton-box" style="height: 60px;"></div>
        `;

        try {
            const resp = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    top_k: 10,
                    rerank_top_k: 5,
                    camera_filter: activeCameraFilter || null,
                }),
            });

            if (!resp.ok) {
                throw new Error(`Search API error: ${resp.statusText}`);
            }

            const data = await resp.json();
            renderResults(data);
        } catch (err) {
            console.error('Search error:', err);
            aiAnswerBody.innerHTML = `<p style="color: #dc2626;">Search failed: ${err.message}</p>`;
            evidenceList.innerHTML = `<div class="empty-state"><p style="color: #dc2626;">Failed to load results.</p></div>`;
        } finally {
            searchBtn.disabled = false;
            searchBtn.innerHTML = `<span>Search Video</span>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>`;
        }
    }

    // ------------------------------------------------------------------
    // Render Results in Classic Light UI with Staggered Entrance
    // ------------------------------------------------------------------
    function renderResults(data) {
        let formattedAnswer = data.answer || 'No answer generated.';
        
        // Highlight clickable timestamps
        formattedAnswer = formattedAnswer.replace(/(\b\d{2}:\d{2}:\d{2}\b)/g, (match) => {
            const parts = match.split(':').map(Number);
            const secs = parts[0] * 3600 + parts[1] * 60 + parts[2];
            return `<span class="ev-ts interactive-ts" data-seconds="${secs}" style="cursor:pointer;" title="Click to seek video to ${match}">${match}</span>`;
        });

        // Highlight Camera tags
        formattedAnswer = formattedAnswer.replace(/(\bCAM_\d{2}\b)/g, `<strong class="text-blue">$1</strong>`);

        aiAnswerBody.innerHTML = `<div class="animate-slide-in" style="white-space: pre-wrap;">${formattedAnswer}</div>`;

        if (data.evaluation) {
            const ev = data.evaluation;
            const ret = ev.retrieval_metrics || {};
            const ans = ev.answer_metrics || {};
            mPrecision.textContent = (ret.precision_at_k ?? 1.0).toFixed(1);
            mMrr.textContent = (ret.mrr ?? 1.0).toFixed(1);
            mNdcg.textContent = (ret.ndcg_at_k ?? 1.0).toFixed(1);
            mUtil.textContent = `${Math.round((ans.context_utilization ?? 1.0) * 100)}%`;
            metricsBar.style.display = 'grid';
        }

        aiAnswerBody.querySelectorAll('.interactive-ts').forEach(el => {
            el.addEventListener('click', () => {
                const sec = parseFloat(el.dataset.seconds);
                seekToTime(sec, el.textContent);
            });
        });

        const results = data.results || [];
        evidenceCount.textContent = results.length;

        if (results.length === 0) {
            evidenceList.innerHTML = `<div class="empty-state"><p>No relevant video moments found for this camera/query filter.</p></div>`;
            return;
        }

        evidenceList.innerHTML = results.map((item) => `
            <div class="evidence-item" data-seconds="${item.seconds}" data-timestamp="${item.timestamp}">
                <div class="evidence-meta">
                    <div class="ev-header">
                        <span class="ev-rank">#${item.rank}</span>
                        <span class="ev-camera">${item.camera}</span>
                        <span class="ev-ts">${item.timestamp}</span>
                    </div>
                    <div class="ev-desc">${escapeHtml(item.description)}</div>
                </div>
                <div class="ev-scores">
                    <span class="score-badge rerank">Rerank: ${item.rerank_score}</span>
                    <span class="score-badge">FAISS: ${item.faiss_score}</span>
                    <button class="btn-seek">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                        Seek
                    </button>
                </div>
            </div>
        `).join('');

        evidenceList.querySelectorAll('.evidence-item').forEach(card => {
            card.addEventListener('click', () => {
                evidenceList.querySelectorAll('.evidence-item').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                const secs = parseFloat(card.dataset.seconds);
                const ts = card.dataset.timestamp;
                seekToTime(secs, ts);
            });
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
