/**
 * VideoRAG Web Application JavaScript
 * Handles API calls, interactive search, camera filtering, video player seeking,
 * and live glassmorphic UI updates.
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

    let activeCameraFilter = '';

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
            seekNotice.style.borderColor = '#38bdf8';
            setTimeout(() => {
                seekNotice.textContent = 'Click any timestamp result to seek';
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
        
        // Re-execute search if input is present
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
    // Execute Search API Call
    // ------------------------------------------------------------------
    async function executeSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Set Loading State
        searchBtn.disabled = true;
        searchBtn.innerHTML = `<span>Searching…</span>`;
        aiAnswerBody.innerHTML = `<p class="placeholder-text"><span class="text-blue">Local Qwen3-VL GPU</span> is analyzing video context and generating answer…</p>`;
        evidenceList.innerHTML = `<div class="empty-state"><p class="text-blue">Retrieving and reranking video moments…</p></div>`;

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
            aiAnswerBody.innerHTML = `<p style="color: #f87171;">Search failed: ${err.message}</p>`;
            evidenceList.innerHTML = `<div class="empty-state"><p style="color: #f87171;">Failed to load results.</p></div>`;
        } finally {
            searchBtn.disabled = false;
            searchBtn.innerHTML = `<span>Search Video</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>`;
        }
    }

    // ------------------------------------------------------------------
    // Render Results in Glass UI
    // ------------------------------------------------------------------
    function renderResults(data) {
        // Format AI Answer text with clickable timestamps & camera tags
        let formattedAnswer = data.answer || 'No answer generated.';
        
        // Highlight timestamps HH:MM:SS
        formattedAnswer = formattedAnswer.replace(/(\b\d{2}:\d{2}:\d{2}\b)/g, (match) => {
            const parts = match.split(':').map(Number);
            const secs = parts[0] * 3600 + parts[1] * 60 + parts[2];
            return `<span class="ev-ts interactive-ts" data-seconds="${secs}" style="cursor:pointer;" title="Click to jump video to ${match}">${match}</span>`;
        });

        // Highlight Camera tags (e.g. CAM_01)
        formattedAnswer = formattedAnswer.replace(/(\bCAM_\d{2}\b)/g, `<strong class="text-blue">$1</strong>`);

        aiAnswerBody.innerHTML = `<div style="white-space: pre-wrap;">${formattedAnswer}</div>`;

        // Render Metrics
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

        // Add Event Delegate for AI Answer Clickable Timestamps
        aiAnswerBody.querySelectorAll('.interactive-ts').forEach(el => {
            el.addEventListener('click', () => {
                const sec = parseFloat(el.dataset.seconds);
                seekToTime(sec, el.textContent);
            });
        });

        // Render Evidence Cards
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

        // Add Evidence Item Click Handlers
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
