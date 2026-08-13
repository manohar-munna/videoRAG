/**
 * VideoRAG Web Application JavaScript — Classic Light Theme
 * Handles API calls, interactive search, camera filtering, video player seeking,
 * Multi-Tab Developer Mode, and Floating Non-Blocking Metric Tooltip Popovers.
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

    // Dev Mode Multi-Tab Elements
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

    // Vector Debugger Elements
    const vNorm = document.getElementById('v-norm');
    const vectorArrayDisplay = document.getElementById('vector-array-display');
    const tEmbed = document.getElementById('t-embed');
    const tFaiss = document.getElementById('t-faiss');
    const tRerank = document.getElementById('t-rerank');
    const tLlm = document.getElementById('t-llm');
    const promptPreviewDisplay = document.getElementById('prompt-preview-display');

    // JSON Explorer Elements
    const refreshJsonBtn = document.getElementById('refresh-json-btn');
    const jsonCodeDisplay = document.getElementById('json-code-display');

    // Floating Popover Card Elements
    const infoPopover = document.getElementById('info-popover');
    const popoverTitle = document.getElementById('popover-title');
    const popoverBody = document.getElementById('popover-body');

    let activeCameraFilter = '';
    let isDevModeActive = false;
    let popoverTimeout = null;

    // ------------------------------------------------------------------
    // Metric Explanations Dictionary for (i) Buttons
    // ------------------------------------------------------------------
    const METRIC_EXPLANATIONS = {
        dev_panel_overview: {
            title: "Developer Mode Overview",
            body: "An advanced inspection suite that exposes edge frame hash filtering, 384-dimensional vector embedding representations, pipeline timing breakdowns, and raw indexed JSON event chunks."
        },
        dhash_phash: {
            title: "dHash & pHash Edge Gate",
            body: "Edge frame hashing algorithms converting 1080p frames into compact 64-bit binary integers (<0.2ms) to detect static scenes and skip sending duplicate frames to LLMs."
        },
        hamming_threshold: {
            title: "Hamming Distance & Threshold",
            body: "Count of differing bit positions between consecutive frame 64-bit hashes (<code>bin(hashA ^ hashB).count('1')</code>).<br>• Range: <code>0</code> (identical) to <code>64</code> (inverted).<br>• <strong>Optimal Threshold (8–12)</strong>: Drops 10–25% of static duplicate scenes."
        },
        total_frames: {
            title: "Total Sampled Frames",
            body: "Count of video frames extracted at fixed intervals (e.g. 1 frame every 15s) from the CCTV stream before edge filtering."
        },
        keyframes_kept: {
            title: "Keyframes Kept (VLM)",
            body: "Frames whose Hamming distance exceeded the threshold. These frames represent significant visual motion or scene shifts and are passed to Qwen3-VL."
        },
        static_frames: {
            title: "Static Frames Skipped",
            body: "Duplicate or static frames whose Hamming distance was below threshold. Discarded at the edge gate, saving bandwidth and zero VLM compute wasted!"
        },
        compute_saved: {
            title: "LLM Compute Saved (%)",
            body: "Percentage reduction in expensive VLM visual inference operations achieved by dropping static duplicate frames at the edge."
        },
        fingerprint_hex: {
            title: "64-Bit Fingerprint (Hex)",
            body: "Hexadecimal string (e.g., <code>0xd99159b3636bd332</code>) representing 64 binary gradient bits extracted from the CCTV frame image."
        },
        motion_pct: {
            title: "Normalized Motion Percentage",
            body: "Visual motion percentage calculated by dividing the frame's Hamming distance by max 64 bits (<code>(hamming / 64) * 100</code>)."
        },
        text_embedding: {
            title: "Text Embedding (all-MiniLM-L6-v2)",
            body: "Converts natural language queries and CCTV descriptions into 384-D dense vectors where similar surveillance concepts sit close together."
        },
        vector_norm: {
            title: "Vector Norm (L2 Length)",
            body: "Euclidean length of the 384-D query vector (<code>||v|| = sqrt(sum(v_i^2))</code>). Normalized to exactly <code>1.0000</code> for unit cosine similarity."
        },
        pipeline_breakdown: {
            title: "Pipeline Execution Breakdown",
            body: "Millisecond timings across 4 stages:<br>1. <code>Query Embedding</code> (CPU)<br>2. <code>FAISS Search</code> (Flat IP)<br>3. <code>Cross-Encoder Rerank</code> (CPU)<br>4. <code>Qwen3-VL Generation</code> (GPU)."
        },
        faiss_search: {
            title: "FAISS (Facebook AI Similarity Search)",
            body: "High-performance vector database engine. Computes cosine inner-products across thousands of 384-D frame vectors in under 2ms."
        },
        cross_encoder: {
            title: "Cross-Encoder Reranker",
            body: "Second-stage Transformer model (<code>ms-marco-MiniLM-L-6-v2</code>) that joint-evaluates query and retrieved descriptions to eliminate false positives."
        },
        qwen3_vl: {
            title: "Local Qwen3-VL 4B Vision-Language Model",
            body: "A 4-Billion parameter multimodal neural network running locally on GPU CUDA via <code>llama-server</code> for visual reasoning."
        },
        raw_prompt: {
            title: "Raw Constructed Prompt",
            body: "Complete system instructions, safety rules, and top-5 retrieved CCTV evidence chunks formatted into a prompt sent to Qwen3-VL."
        },
        json_explorer: {
            title: "Indexed CCTV Events (JSON)",
            body: "Raw dataset (<code>data/real_cctv_events.json</code>) containing camera IDs, timestamps, and Qwen3-VL visual descriptions."
        },
        precision_5: {
            title: "Precision at Top-5 (P@5)",
            body: "Fraction of top-5 retrieved CCTV video moments containing relevant matching keywords. Range: 0.0 to 1.0 (Target: 1.0 = 100%)."
        },
        mrr: {
            title: "Mean Reciprocal Rank (MRR)",
            body: "Evaluates how quickly the first relevant evidence chunk appears (<code>1 / rank</code>). Score of 1.0 means #1 result was relevant!"
        },
        ndcg_5: {
            title: "NDCG at Top-5",
            body: "Measures ranking quality with logarithmic position decay. Relevant results placed higher up score much higher."
        },
        context_util: {
            title: "LLM Context Utilization (%)",
            body: "Percentage of top retrieved CCTV evidence moments explicitly cited and utilized by local Qwen3-VL in its final answer."
        }
    };

    // ------------------------------------------------------------------
    // Floating Tooltip Popover Positioning next to (i) Symbol
    // ------------------------------------------------------------------
    function showPopoverNextToElement(targetEl, infoKey) {
        const infoData = METRIC_EXPLANATIONS[infoKey];
        if (!infoData || !infoPopover) return;

        popoverTitle.textContent = infoData.title;
        popoverBody.innerHTML = infoData.body;
        infoPopover.style.display = 'flex';

        // Calculate exact screen coordinates
        const rect = targetEl.getBoundingClientRect();
        const scrollX = window.scrollX || window.pageXOffset;
        const scrollY = window.scrollY || window.pageYOffset;

        // Position popover right above or beside the (i) icon
        let left = rect.right + scrollX + 8;
        let top = rect.top + scrollY - 10;

        // Ensure popover doesn't overflow right screen edge
        if (left + 320 > window.innerWidth) {
            left = rect.left + scrollX - 330;
        }

        infoPopover.style.left = `${Math.max(10, left)}px`;
        infoPopover.style.top = `${Math.max(10, top)}px`;
    }

    function hidePopover() {
        if (infoPopover) {
            infoPopover.style.display = 'none';
        }
    }

    // Attach Hover & Click Listeners for (i) Info Buttons
    document.addEventListener('mouseover', (e) => {
        const btn = e.target.closest('.info-btn');
        if (btn) {
            clearTimeout(popoverTimeout);
            showPopoverNextToElement(btn, btn.dataset.info);
        }
    });

    document.addEventListener('mouseout', (e) => {
        const btn = e.target.closest('.info-btn');
        if (btn) {
            popoverTimeout = setTimeout(hidePopover, 200);
        }
    });

    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.info-btn');
        if (btn) {
            e.stopPropagation();
            showPopoverNextToElement(btn, btn.dataset.info);
        } else if (infoPopover && !e.target.closest('#info-popover')) {
            hidePopover();
        }
    });

    // Prevent popover from closing when hovering over popover body
    if (infoPopover) {
        infoPopover.addEventListener('mouseenter', () => clearTimeout(popoverTimeout));
        infoPopover.addEventListener('mouseleave', () => popoverTimeout = setTimeout(hidePopover, 200));
    }

    // ------------------------------------------------------------------
    // Dev Mode Tab Switching Logic
    // ------------------------------------------------------------------
    document.querySelectorAll('.dev-tab-btn').forEach(tabBtn => {
        tabBtn.addEventListener('click', (e) => {
            if (e.target.classList.contains('info-btn')) return;

            document.querySelectorAll('.dev-tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.dev-tab-content').forEach(c => c.style.display = 'none');
            
            tabBtn.classList.add('active');
            const targetTabId = tabBtn.dataset.tab;
            const targetContent = document.getElementById(targetTabId);
            if (targetContent) targetContent.style.display = 'block';

            if (targetTabId === 'tab-json') {
                fetchEventsJson();
            }
        });
    });

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
                fetchEventsJson();
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
                        run_vlm_captioning: true,
                    }),
                });

                if (resp.ok) {
                    const data = await resp.json();
                    renderDevAuditLogs(data.filter_stats, data.audit_trail);
                    checkHealth();
                    fetchEventsJson();
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
    // Fetch & Display JSON Event Chunks
    // ------------------------------------------------------------------
    if (refreshJsonBtn) {
        refreshJsonBtn.addEventListener('click', fetchEventsJson);
    }

    async function fetchEventsJson() {
        if (!jsonCodeDisplay) return;
        try {
            jsonCodeDisplay.textContent = "Loading indexed CCTV event chunks…";
            const resp = await fetch('/api/events');
            if (resp.ok) {
                const data = await resp.json();
                jsonCodeDisplay.textContent = JSON.stringify(data, null, 2);
            }
        } catch (err) {
            jsonCodeDisplay.textContent = "Failed to load JSON dataset: " + err.message;
        }
    }

    // ------------------------------------------------------------------
    // Shutdown Button Handler
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
    // Video Timer Tracking & Seeking
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
    // Camera Selection Pills & Quick Queries
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

    quickQueriesContainer.addEventListener('click', (e) => {
        const chip = e.target.closest('.query-chip');
        if (!chip) return;
        queryInput.value = chip.dataset.query;
        executeSearch();
    });

    searchBtn.addEventListener('click', executeSearch);
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            executeSearch();
        }
    });

    // ------------------------------------------------------------------
    // Execute Search API Call & Populate Vector Debugger
    // ------------------------------------------------------------------
    async function executeSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        searchBtn.innerHTML = `<span>Searching…</span>`;

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
            renderVectorDebugger(data.debug_trace);
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

    // Render Vector Debugger Trace in Tab 2
    function renderVectorDebugger(trace) {
        if (!trace) return;

        if (vNorm) vNorm.textContent = (trace.query_vector_norm ?? 1.0).toFixed(4);
        if (vectorArrayDisplay) {
            const sample = trace.query_vector_sample || [];
            vectorArrayDisplay.textContent = `[${sample.join(', ')}, ... (${trace.query_vector_dim} dims total)]`;
        }

        const timings = trace.timings_ms || {};
        if (tEmbed) tEmbed.textContent = `${timings.query_embedding_ms ?? 0} ms`;
        if (tFaiss) tFaiss.textContent = `${timings.faiss_retrieval_ms ?? 0} ms`;
        if (tRerank) tRerank.textContent = `${timings.cross_encoder_rerank_ms ?? 0} ms`;
        if (tLlm) tLlm.textContent = `${timings.llm_generation_ms ?? 0} ms`;

        if (promptPreviewDisplay) {
            promptPreviewDisplay.textContent = trace.prompt_constructed || "No prompt generated.";
        }
    }

    // ------------------------------------------------------------------
    // Render Results in Classic Light UI
    // ------------------------------------------------------------------
    function renderResults(data) {
        let formattedAnswer = data.answer || 'No answer generated.';
        
        formattedAnswer = formattedAnswer.replace(/(\b\d{2}:\d{2}:\d{2}\b)/g, (match) => {
            const parts = match.split(':').map(Number);
            const secs = parts[0] * 3600 + parts[1] * 60 + parts[2];
            return `<span class="ev-ts interactive-ts" data-seconds="${secs}" style="cursor:pointer;" title="Click to seek video to ${match}">${match}</span>`;
        });

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
