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

    // Lazy VLM & Vector Grounding 4-Stage Horizontal Pipeline Elements
    const lazyPulseDot = document.getElementById('lazy-pulse-dot');
    const lazyPipelineBadge = document.getElementById('lazy-pipeline-badge');
    const pstage1 = document.getElementById('pstage-1');
    const pstage2 = document.getElementById('pstage-2');
    const pstage3 = document.getElementById('pstage-3');
    const pstage4 = document.getElementById('pstage-4');
    const pstage1Status = document.getElementById('pstage-1-status');
    const pstage2Status = document.getElementById('pstage-2-status');
    const pstage3Status = document.getElementById('pstage-3-status');
    const pstage4Status = document.getElementById('pstage-4-status');
    const pstage1Time = document.getElementById('pstage-1-time');
    const pstage2Time = document.getElementById('pstage-2-time');
    const pstage3Time = document.getElementById('pstage-3-time');
    const pstage4Time = document.getElementById('pstage-4-time');
    const pstage1Detail = document.getElementById('pstage-1-detail');
    const pstage2Detail = document.getElementById('pstage-2-detail');
    const pstage3Detail = document.getElementById('pstage-3-detail');
    const pstage4Detail = document.getElementById('pstage-4-detail');
    const lazyStatKeyframes = document.getElementById('lazy-stat-keyframes');
    const lazyStatLatency = document.getElementById('lazy-stat-latency');
    const toggleVectorsGridBtn = document.getElementById('toggle-vectors-grid-btn');
    const vgCollapseArrow = document.getElementById('vg-collapse-arrow');
    const vgToggleText = document.getElementById('vg-toggle-text');
    const vgSearchWrap = document.getElementById('vg-search-wrap');
    const vgSearchInput = document.getElementById('vg-search-input');
    const refreshVectorsBtn = document.getElementById('refresh-vectors-btn');
    const frameVectorsGrid = document.getElementById('frame-vectors-grid');

    // Vector Debugger Elements (512-D MobileCLIP + Qwen3-VL)
    const vNorm = document.getElementById('v-norm');
    const vectorArrayDisplay = document.getElementById('vector-array-display');
    const copyVecBtn = document.getElementById('copy-vec-btn');
    const tEmbed = document.getElementById('t-embed');
    const tFaiss = document.getElementById('t-faiss');
    const tExpandTiming = document.getElementById('t-expand-timing');
    const tLlm = document.getElementById('t-llm');
    const tTotalTrace = document.getElementById('t-total-trace');
    const promptPreviewDisplay = document.getElementById('prompt-preview-display');

    // JSON Keyframe Dataset Explorer Elements
    const refreshJsonBtn = document.getElementById('refresh-json-btn');
    const copyJsonBtn = document.getElementById('copy-json-btn');
    const jsonCodeDisplay = document.getElementById('json-code-display');
    const jsonCardsContainer = document.getElementById('json-cards-container');
    const jsonViewCardsBtn = document.getElementById('json-view-cards-btn');
    const jsonViewRawBtn = document.getElementById('json-view-raw-btn');
    const jsonCameraFilters = document.getElementById('json-camera-filters');
    const jsonSearchInput = document.getElementById('json-search-input');
    const jsonMatchCount = document.getElementById('json-match-count');
    const jsonStatTotal = document.getElementById('json-stat-total');
    const jsonStatCams = document.getElementById('json-stat-cams');
    const jsonStatDim = document.getElementById('json-stat-dim');
    const jsonStatSize = document.getElementById('json-stat-size');

    // Floating Popover Card Elements
    const infoPopover = document.getElementById('info-popover');
    const popoverTitle = document.getElementById('popover-title');
    const popoverBody = document.getElementById('popover-body');

    let activeCameraFilter = '';
    let isDevModeActive = false;
    let popoverTimeout = null;
    let allLazyVectors = [];

    // ------------------------------------------------------------------
    // Metric Explanations Dictionary for (i) Buttons
    // ------------------------------------------------------------------
    const METRIC_EXPLANATIONS = {
        dev_panel_overview: {
            title: "Developer Mode Overview",
            body: "An advanced inspection suite that exposes edge frame hash filtering, 512-dimensional MobileCLIP vector representations, live pipeline timing breakdowns, and raw indexed dataset chunks."
        },
        lazy_vlm: {
            title: "Lazy VLM Architecture",
            body: "A next-generation surveillance paradigm: CCTV frames are embedded into 512-D MobileCLIP vectors at 0ms LLM overhead during ingestion (~10s for 179 frames). Heavy multi-frame forensic reasoning with Qwen3-VL is executed exclusively on-demand when a user performs a search."
        },
        vector_grounding: {
            title: "Frame-to-Vector Grounding",
            body: "A direct mathematical link between an actual extracted frame image in <code>sample_cctv.mp4</code> and its 512-dimensional normalized float vector indexed inside the in-memory FAISS database."
        },
        dhash_phash: {
            title: "dHash & pHash Edge Gate",
            body: "Edge frame hashing algorithms converting 1080p frames into compact 64-bit binary integers (<0.2ms) to detect static scenes and skip sending duplicate frames to LLMs."
        },
        hamming_threshold: {
            title: "Hamming Distance & Threshold",
            body: "Count of differing bit positions between consecutive frame 64-bit hashes (<code>bin(hashA ^ hashB).count('1')</code>).<br>• Range: <code>0</code> (identical) to <code>64</code> (inverted).<br>• <strong>Optimal Threshold (8–12)</strong>: Drops 10–35% of static duplicate scenes."
        },
        total_frames: {
            title: "Total Sampled Frames",
            body: "Count of video frames extracted at fixed intervals (e.g. 1 frame every 3s) from the CCTV stream before edge filtering."
        },
        keyframes_kept: {
            title: "Keyframes Kept (VLM)",
            body: "Frames whose Hamming distance exceeded the threshold. These frames represent significant visual motion or scene shifts and are passed to MobileCLIP for vector indexing."
        },
        static_frames: {
            title: "Static Frames Skipped",
            body: "Duplicate or static frames whose Hamming distance was below threshold. Discarded at the edge gate, saving bandwidth and zero compute wasted!"
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
            title: "Multimodal Embedding (Apple MobileCLIP-S2)",
            body: "Converts natural language queries and CCTV frame images into 512-D dense continuous vectors where visual concepts and text queries share the same embedding space."
        },
        vector_norm: {
            title: "Vector Norm (L2 Length)",
            body: "Euclidean length of the 512-D query vector (<code>||v|| = sqrt(sum(v_i^2))</code>). Normalized to exactly <code>1.0000</code> for cosine inner product similarity."
        },
        pipeline_breakdown: {
            title: "Pipeline Execution Breakdown",
            body: "Millisecond timings across 4 stages:<br>1. <code>Query Embedding</code> (MobileCLIP-S2)<br>2. <code>FAISS Vector Search</code> (Flat IP)<br>3. <code>Temporal Expansion</code> (±15–30s Context)<br>4. <code>Qwen3-VL Forensic Reasoning</code> (GPU)."
        },
        faiss_search: {
            title: "FAISS (Facebook AI Similarity Search)",
            body: "High-performance vector database engine. Computes cosine inner-products across 512-D keyframe vectors in under 2ms."
        },
        cross_encoder: {
            title: "Cross-Encoder Reranker",
            body: "Second-stage Transformer model that evaluates candidate moments to optimize top forensic relevance."
        },
        qwen3_vl: {
            title: "Local Qwen3-VL 4B Vision-Language Model",
            body: "A 4-Billion parameter multimodal neural network running locally on GPU CUDA via <code>llama-server</code> for multi-frame visual forensic reasoning."
        },
        raw_prompt: {
            title: "Raw Constructed Prompt",
            body: "Complete system instructions, safety rules, and top retrieved CCTV evidence moments formatted into a prompt sent to Qwen3-VL."
        },
        json_explorer: {
            title: "Indexed CCTV Events (JSON)",
            body: "Raw dataset containing camera IDs, timestamps, and Qwen3-VL visual observations."
        },
        precision_5: {
            title: "Precision at Top-5 (P@5)",
            body: "Fraction of top-5 retrieved CCTV video moments containing relevant matching concepts. Range: 0.0 to 1.0 (Target: 1.0 = 100%)."
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
        },
        rtsp_manager: {
            title: "Async Multi-Threaded RTSP Stream Capture Engine",
            body: "Captures live CCTV network feeds (RTSP/RTMP) on non-blocking background producer threads using a <strong>Size-1 Ring Buffer</strong>."
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

            if (targetTabId === 'tab-lazy-vlm') {
                fetchLazyVectors();
            } else if (targetTabId === 'tab-json') {
                fetchEventsJson();
            } else if (targetTabId === 'tab-rtsp') {
                fetchRtspStreamsStatus();
                if (!rtspPollInterval) {
                    rtspPollInterval = setInterval(fetchRtspStreamsStatus, 2000);
                }
            } else {
                if (rtspPollInterval) {
                    clearInterval(rtspPollInterval);
                    rtspPollInterval = null;
                }
            }
        });
    });

    let rtspPollInterval = null;

    // ------------------------------------------------------------------
    // Tab 0: Lazy VLM Architecture & Frame-to-Vector Grounding
    // ------------------------------------------------------------------
    async function fetchLazyVectors() {
        if (!frameVectorsGrid) return;
        try {
            frameVectorsGrid.innerHTML = `
                <div class="placeholder-text" style="padding: 24px; text-align: center; width: 100%;">
                    Loading 512-D MobileCLIP vectors from FAISS index…
                </div>
            `;

            const resp = await fetch('/api/lazy_vlm/vectors');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            allLazyVectors = data.items || [];

            if (lazyStatKeyframes) {
                lazyStatKeyframes.textContent = `${data.total || allLazyVectors.length} Vectors`;
            }
            if (statVectors) {
                statVectors.textContent = `${data.total || allLazyVectors.length} Vectors`;
            }
            if (vgToggleText && !isKeyframeGridOpen) {
                vgToggleText.textContent = `Show Images (${data.total || allLazyVectors.length})`;
            } else if (vgToggleText) {
                vgToggleText.textContent = `Hide Images (${data.total || allLazyVectors.length})`;
            }

            renderLazyVectorCards(allLazyVectors);
        } catch (err) {
            console.error('Failed to load lazy vectors:', err);
            if (frameVectorsGrid) {
                frameVectorsGrid.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;"><p style="color: #ef4444;">Failed to load vector data: ${err.message}</p></div>`;
            }
        }
    }

    function renderLazyVectorCards(items) {
        if (!frameVectorsGrid) return;
        if (!items || items.length === 0) {
            frameVectorsGrid.innerHTML = `<div class="empty-state" style="grid-column: 1 / -1;"><p>No vectors match search filter.</p></div>`;
            return;
        }

        frameVectorsGrid.innerHTML = items.map(v => {
            const vecPreview = v.vector_sample && v.vector_sample.length > 0
                ? `[${v.vector_sample.slice(0, 5).join(', ')}, …]`
                : `[512-D float32]`;

            const thumb = v.image_path || '/data/extracted_frames/placeholder.jpg';

            return `
                <div class="frame-vector-card" data-index="${v.index}" data-ts="${escapeHtml(v.timestamp)}" data-secs="${v.seconds || 0}" data-img="${escapeHtml(thumb)}">
                    <div class="fv-thumb-wrap">
                        <img class="fv-thumb" src="${escapeHtml(thumb)}" alt="Keyframe @ ${escapeHtml(v.timestamp)}" loading="lazy" onerror="this.src='/ui/favicon.ico'">
                    </div>
                    <div class="fv-info">
                        <div class="fv-top-row">
                            <span class="fv-ts-badge">⏱️ ${escapeHtml(v.timestamp)}</span>
                            <span class="fv-vec-id">#${v.index} (512-D)</span>
                        </div>
                        <div class="fv-vector-preview" title="MobileCLIP-S2 512-D normalized vector sample">${escapeHtml(vecPreview)}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                            <span style="font-size: 0.65rem; color: #64748b; font-family: monospace;">dHash: ${escapeHtml((v.hash_hex || '').slice(0, 10))}…</span>
                            <span class="fv-score-chip" style="font-size: 0.62rem;">MobileCLIP-S2</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add click listener to seek player to frame timestamp
        frameVectorsGrid.querySelectorAll('.frame-vector-card').forEach(card => {
            card.addEventListener('click', () => {
                const ts = card.dataset.ts;
                const secs = parseFloat(card.dataset.secs) || 0;
                const img = card.dataset.img;
                switchSurveillanceFeed('CAM_01', secs, img, ts);
                if (cctvPlayer) {
                    cctvPlayer.currentTime = secs;
                    cctvPlayer.play().catch(() => {});
                }
            });
        });
    }

    if (vgSearchInput) {
        vgSearchInput.addEventListener('input', () => {
            const term = vgSearchInput.value.trim().toLowerCase();
            if (!term) {
                renderLazyVectorCards(allLazyVectors);
                return;
            }
            const filtered = allLazyVectors.filter(v => 
                (v.timestamp && v.timestamp.toLowerCase().includes(term)) ||
                (v.index !== undefined && String(v.index).includes(term)) ||
                (v.filename && v.filename.toLowerCase().includes(term))
            );
            renderLazyVectorCards(filtered);
        });
    }

    if (refreshVectorsBtn) {
        refreshVectorsBtn.addEventListener('click', () => {
            fetchLazyVectors();
        });
    }

    // Toggle Collapsible Keyframe Images Grid (Collapsed by Default)
    let isKeyframeGridOpen = false;
    if (toggleVectorsGridBtn && frameVectorsGrid) {
        toggleVectorsGridBtn.addEventListener('click', () => {
            isKeyframeGridOpen = !isKeyframeGridOpen;
            const count = allLazyVectors.length || 179;
            if (isKeyframeGridOpen) {
                frameVectorsGrid.classList.remove('collapsed');
                if (vgCollapseArrow) vgCollapseArrow.classList.remove('collapsed');
                if (vgToggleText) vgToggleText.textContent = `Hide Images (${count})`;
                if (vgSearchWrap) vgSearchWrap.style.opacity = '1';
                if (vgSearchWrap) vgSearchWrap.style.pointerEvents = 'auto';
            } else {
                frameVectorsGrid.classList.add('collapsed');
                if (vgCollapseArrow) vgCollapseArrow.classList.add('collapsed');
                if (vgToggleText) vgToggleText.textContent = `Show Images (${count})`;
                if (vgSearchWrap) vgSearchWrap.style.opacity = '0.5';
                if (vgSearchWrap) vgSearchWrap.style.pointerEvents = 'none';
            }
        });
    }

    // Model Benchmark Quick Triggers
    document.querySelectorAll('.benchmark-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            if (query && queryInput) {
                queryInput.value = query;
                executeSearch();
            }
        });
    });

    // ------------------------------------------------------------------
    // Tab 4: RTSP Live Stream Manager & Multi-Camera Stream Controls
    // ------------------------------------------------------------------
    const rtspStreamsList = document.getElementById('rtsp-streams-list');
    const addRtspBtn = document.getElementById('add-rtsp-btn');
    const rtspCamId = document.getElementById('rtsp-cam-id');
    const rtspUrl = document.getElementById('rtsp-url');
    const rtspInterval = document.getElementById('rtsp-interval');

    if (addRtspBtn) {
        addRtspBtn.addEventListener('click', async () => {
            const camId = rtspCamId.value.trim();
            const url = rtspUrl.value.trim();
            const interval = parseFloat(rtspInterval.value) || 5.0;

            if (!camId || !url) {
                alert('Please enter both a Camera ID (e.g. CAM_NORTH) and Stream URL.');
                return;
            }

            try {
                addRtspBtn.disabled = true;
                addRtspBtn.innerHTML = `<span>Starting Capture…</span>`;

                const resp = await fetch('/api/streams/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_id: camId,
                        stream_url: url,
                        sample_interval: interval,
                        hash_method: "dhash",
                        threshold: 10,
                    }),
                });

                if (resp.ok) {
                    rtspCamId.value = '';
                    rtspUrl.value = '';
                    fetchRtspStreamsStatus();
                    fetchCameraFeeds();
                    fetchCameraPills();
                    fetchEventsJson();
                } else {
                    const err = await resp.json();
                    alert('Failed to start stream: ' + (err.detail || 'Unknown error'));
                }
            } catch (err) {
                alert('Stream connection error: ' + err.message);
            } finally {
                addRtspBtn.disabled = false;
                addRtspBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg><span>Start Async RTSP Capture</span>`;
            }
        });
    }

    async function fetchRtspStreamsStatus() {
        if (!rtspStreamsList) return;
        try {
            const resp = await fetch('/api/streams/status');
            if (resp.ok) {
                const data = await resp.json();
                renderRtspStreams(data.active_streams || []);
            }
        } catch (err) {
            console.warn('Failed to fetch RTSP streams status:', err);
        }
    }

    function renderRtspStreams(streams) {
        if (!streams || streams.length === 0) {
            rtspStreamsList.innerHTML = `<div class="empty-state" style="padding: 20px;"><p class="placeholder-text">No active camera streams running. Enter a stream URL above to launch multi-threaded capture.</p></div>`;
            return;
        }

        rtspStreamsList.innerHTML = streams.map(s => {
            const isRunning = s.is_running;
            const isPaused = s.is_paused;
            const isConnected = s.is_connected;
            const isLive = s.is_live ?? (s.camera_type !== 'video_file');
            const camType = s.camera_type || (isLive ? 'youtube_stream' : 'video_file');
            
            let statusBadge = isRunning ? (isConnected ? 'LIVE (TCP)' : 'RECONNECTING') : (isPaused ? 'PAUSED' : 'STOPPED');
            let statusColor = isRunning && isConnected ? 'text-green' : (isPaused ? 'text-blue' : 'text-dim');

            let typeBadgeClass = 'badge-live';
            let typeBadgeLabel = '🔴 24/7 LIVE';
            if (camType === 'video_file') {
                typeBadgeClass = 'badge-recorded';
                typeBadgeLabel = '📹 RECORDED MP4';
            } else if (camType === 'youtube_video') {
                typeBadgeClass = 'badge-yt';
                typeBadgeLabel = '▶️ YT VIDEO';
            }

            const progressPct = s.progress_pct != null ? s.progress_pct : (camType === 'video_file' ? 100 : null);

            return `
                <div class="stream-card ${isRunning ? 'active' : ''}">
                    <div class="stream-card-header">
                        <div class="stream-cam-title">
                            <span class="status-indicator ${isConnected ? 'online' : (isPaused ? 'warning' : 'offline')}"></span>
                            ${escapeHtml(s.camera_id)} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(${escapeHtml(s.name || s.camera_id)})</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span class="cctv-yt-badge ${typeBadgeClass}">${typeBadgeLabel}</span>
                            <span class="feed-badge ${isRunning ? 'live' : ''}">${statusBadge}</span>
                        </div>
                    </div>
                    <div class="stream-url-tag">URL: ${escapeHtml(s.stream_url)}</div>

                    <!-- YouTube-Style Play & Extraction Progress Bar -->
                    <div class="cctv-yt-progress-container ${isLive ? 'live-stream-track' : ''}">
                        <div class="cctv-yt-progress-header">
                            <span style="font-size:0.72rem; font-weight:600; color:var(--text-muted);">
                                ${camType === 'video_file' ? 'Indexing & Playback Timeline' : 'Live Ingestion & Frame Extraction'}
                            </span>
                            ${isLive ? `
                                <span class="cctv-yt-live-pulse">● Live Edge Capture</span>
                            ` : `
                                <span class="cctv-yt-progress-val">${progressPct != null ? progressPct + '%' : '100%'} Processed</span>
                            `}
                        </div>
                        <div class="cctv-yt-progress-track ${isLive ? 'live-track' : ''}">
                            <div class="cctv-yt-progress-fill ${isLive ? 'live-fill' : ''}" style="width: ${progressPct != null ? progressPct : 100}%;"></div>
                            ${!isLive ? `<div class="cctv-yt-progress-scrubber" style="left: ${progressPct != null ? progressPct : 100}%;"></div>` : ''}
                        </div>
                        <div class="cctv-yt-progress-sub">
                            <span>${(s.total_frames_read || 0).toLocaleString()} frames read</span>
                            <span>${s.keyframes_kept || 0} keyframes extracted & indexed</span>
                        </div>
                    </div>

                    <div class="stream-metrics-grid">
                        <div class="sm-item"><span class="sm-label">Status</span><span class="sm-val ${statusColor}">${statusBadge}</span></div>
                        <div class="sm-item"><span class="sm-label">FPS</span><span class="sm-val">${s.fps}</span></div>
                        <div class="sm-item"><span class="sm-label">Duration</span><span class="sm-val">${s.total_duration_sec ? Math.round(s.total_duration_sec) + 's' : '24/7 LIVE'}</span></div>
                        <div class="sm-item"><span class="sm-label">Ring Dropped</span><span class="sm-val text-dim">${s.total_frames_dropped}</span></div>
                        <div class="sm-item"><span class="sm-label">Keyframes Kept</span><span class="sm-val text-blue">${s.keyframes_kept}</span></div>
                        <div class="sm-item"><span class="sm-label">Compute Saved</span><span class="sm-val text-green">${s.llm_compute_saved_pct}%</span></div>
                    </div>
                    <div class="stream-card-actions">
                        <button class="index-stream-btn" data-cam="${escapeHtml(s.camera_id)}" style="padding: 3px 8px; font-size: 0.72rem;" ${s.keyframes_kept === 0 ? 'disabled' : ''} title="Manual trigger or sync VLM indexing">
                            ⚡ Indexed (${s.keyframes_kept})
                        </button>
                        ${isPaused ? `
                            <button class="btn-stream-action btn-resume-stream resume-stream-btn" data-cam="${escapeHtml(s.camera_id)}" title="Resume stream extraction">
                                ▶️ Resume Extraction
                            </button>
                        ` : `
                            <button class="btn-stream-action btn-pause-stream pause-stream-btn" data-cam="${escapeHtml(s.camera_id)}" title="Pause stream extraction">
                                ⏸️ Pause Extraction
                            </button>
                        `}
                        <button class="btn-stream-action btn-remove-stream remove-stream-btn" data-cam="${escapeHtml(s.camera_id)}" title="Remove camera from registry">
                            🗑️ Remove
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        // Pause button handler
        rtspStreamsList.querySelectorAll('.pause-stream-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const camId = btn.dataset.cam;
                await fetch('/api/streams/pause', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_id: camId })
                });
                fetchRtspStreamsStatus();
                fetchCameraFeeds();
            });
        });

        // Resume button handler
        rtspStreamsList.querySelectorAll('.resume-stream-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const camId = btn.dataset.cam;
                await fetch('/api/streams/resume', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_id: camId })
                });
                fetchRtspStreamsStatus();
                fetchCameraFeeds();
            });
        });

        // Remove button handler
        rtspStreamsList.querySelectorAll('.remove-stream-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const camId = btn.dataset.cam;
                const confirmed = confirm(`Are you sure you want to remove ${camId} from the persistent registry?`);
                if (!confirmed) return;

                await fetch('/api/streams/remove', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_id: camId })
                });
                fetchRtspStreamsStatus();
                fetchCameraFeeds();
                fetchCameraPills();
                fetchEventsJson();
            });
        });

        // Index Keyframes handler
        rtspStreamsList.querySelectorAll('.index-stream-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const camId = btn.dataset.cam;
                try {
                    btn.disabled = true;
                    btn.innerHTML = `<span>Indexing VLM…</span>`;
                    const resp = await fetch('/api/streams/index_now', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ camera_id: camId })
                    });
                    if (resp.ok) {
                        const data = await resp.json();
                        alert(`Successfully indexed ${data.indexed_count} keyframes for ${camId}! Total vectors in database: ${data.total_vectors}`);
                        checkHealth();
                        fetchCameraPills();
                        fetchEventsJson();
                    } else {
                        const err = await resp.json();
                        alert('Indexing failed: ' + (err.detail || 'Unknown error'));
                    }
                } catch (err) {
                    alert('Indexing error: ' + err.message);
                } finally {
                    btn.disabled = false;
                }
            });
        });
    }

    let serverTimeOffsetMs = 0; // Tracks clock drift between client and server

    async function checkHealth() {
        try {
            const t0 = Date.now();
            const resp = await fetch('/api/health');
            const t1 = Date.now();

            if (resp.ok) {
                const data = await resp.json();

                if (data && data.server_time) {
                    const latency = (t1 - t0) / 2;
                    const serverMs = data.server_time * 1000;
                    serverTimeOffsetMs = serverMs - (t1 - latency);
                    console.log(`System clocks synchronized. Offset: ${serverTimeOffsetMs}ms`);
                }

                const healthDot = document.getElementById('health-indicator');
                const healthText = document.getElementById('health-status-text');
                if (healthDot && healthText) {
                    healthDot.className = 'status-dot online';
                    healthText.textContent = `Online (${data.vector_count || 0} vectors)`;
                }
                const vectorCountBadge = document.getElementById('vector-count-badge');
                if (vectorCountBadge) {
                    vectorCountBadge.textContent = `${data.vector_count || 179} VECTORS`;
                }
                if (statVectors) {
                    statVectors.textContent = `${data.vector_count || 179} Vectors`;
                }
            }
        } catch (err) {
            console.warn('Health check failed:', err);
        }
    }
    checkHealth();

    async function fetchCameraPills() {
        if (!cameraFilterContainer) return;
        try {
            const resp = await fetch('/api/cameras');
            if (resp.ok) {
                const data = await resp.json();
                const cams = data.cameras || [];
                cameraFilterContainer.innerHTML = `
                    <span class="filter-label">Camera:</span>
                    <button class="filter-pill ${activeCameraFilter === '' ? 'active' : ''}" data-camera="">All Feeds</button>
                    ${cams.map(c => `
                        <button class="filter-pill ${activeCameraFilter === c ? 'active' : ''}" data-camera="${escapeHtml(c)}">${escapeHtml(c)}</button>
                    `).join('')}
                `;
            }
        } catch (err) {
            console.warn('Failed to fetch camera pills:', err);
        }
    }
    fetchCameraPills();

    // ------------------------------------------------------------------
    // Tab 3: JSON Event Chunks Explorer (Dynamic & Interactive)
    // ------------------------------------------------------------------
    let loadedJsonEvents = [];
    let activeJsonCameraFilter = '';
    let jsonSearchTerm = '';
    let jsonViewMode = 'cards'; // 'cards' | 'raw'
    let autoRefreshInterval = null;

    if (jsonViewCardsBtn && jsonViewRawBtn) {
        jsonViewCardsBtn.addEventListener('click', () => {
            jsonViewMode = 'cards';
            jsonViewCardsBtn.classList.add('active');
            jsonViewRawBtn.classList.remove('active');
            if (jsonCardsContainer) jsonCardsContainer.style.display = 'flex';
            if (jsonCodeDisplay) jsonCodeDisplay.style.display = 'none';
            renderJsonExplorer();
        });

        jsonViewRawBtn.addEventListener('click', () => {
            jsonViewMode = 'raw';
            jsonViewRawBtn.classList.add('active');
            jsonViewCardsBtn.classList.remove('active');
            if (jsonCardsContainer) jsonCardsContainer.style.display = 'none';
            if (jsonCodeDisplay) jsonCodeDisplay.style.display = 'block';
            renderJsonExplorer();
        });
    }

    if (jsonSearchInput) {
        jsonSearchInput.addEventListener('input', (e) => {
            jsonSearchTerm = e.target.value.trim().toLowerCase();
            renderJsonExplorer();
        });
    }

    if (refreshJsonBtn) {
        refreshJsonBtn.addEventListener('click', () => {
            fetchEventsJson(false);
        });
    }

    if (copyJsonBtn) {
        copyJsonBtn.addEventListener('click', async () => {
            const filtered = getFilteredJsonEvents();
            try {
                await navigator.clipboard.writeText(JSON.stringify(filtered, null, 2));
                const origText = copyJsonBtn.querySelector('span');
                if (origText) {
                    const prev = origText.textContent;
                    origText.textContent = 'Copied!';
                    setTimeout(() => { origText.textContent = prev; }, 1500);
                }
            } catch (err) {
                alert('Copy failed: ' + err.message);
            }
        });
    }

    function getFilteredJsonEvents() {
        return loadedJsonEvents.filter(item => {
            if (activeJsonCameraFilter && item.camera !== activeJsonCameraFilter) {
                return false;
            }
            if (jsonSearchTerm) {
                const desc = (item.description || '').toLowerCase();
                const cam = (item.camera || '').toLowerCase();
                const ts = (item.timestamp || '').toLowerCase();
                if (!desc.includes(jsonSearchTerm) && !cam.includes(jsonSearchTerm) && !ts.includes(jsonSearchTerm)) {
                    return false;
                }
            }
            return true;
        });
    }

    async function fetchEventsJson(silent = false) {
        try {
            if (!silent && refreshJsonBtn) {
                refreshJsonBtn.disabled = true;
                const span = refreshJsonBtn.querySelector('span');
                if (span) span.textContent = 'Refreshing…';
            }
            const resp = await fetch('/api/events?detailed=true');
            if (resp.ok) {
                const data = await resp.json();
                const newEvents = data.events || [];
                const countChanged = newEvents.length !== loadedJsonEvents.length;
                loadedJsonEvents = newEvents;
                
                // Update stats chips
                if (jsonStatTotal) jsonStatTotal.textContent = `${data.total_count || loadedJsonEvents.length} Keyframes`;
                if (jsonStatCams) jsonStatCams.textContent = `${(data.cameras || []).length || 1} Camera`;
                if (jsonStatDim) jsonStatDim.textContent = `512-D MobileCLIP`;
                if (jsonStatSize) {
                    jsonStatSize.textContent = `0.0s LLM Wait`;
                }

                // Render camera filter pills for JSON tab
                renderJsonCameraFilters(data.cameras || ['CAM_01']);
                renderJsonExplorer();
            } else {
                if (jsonCodeDisplay && !silent) jsonCodeDisplay.textContent = 'Failed to load CCTV events dataset.';
            }
        } catch (err) {
            if (!silent) {
                console.warn('Failed to fetch events JSON:', err);
                if (jsonCodeDisplay) jsonCodeDisplay.textContent = 'Error loading JSON: ' + err.message;
            }
        } finally {
            if (!silent && refreshJsonBtn) {
                refreshJsonBtn.disabled = false;
                const span = refreshJsonBtn.querySelector('span');
                if (span) span.textContent = 'Refresh JSON Data';
            }
        }
    }

    function renderJsonCameraFilters(cameras) {
        if (!jsonCameraFilters) return;
        
        // Count per camera
        const counts = {};
        loadedJsonEvents.forEach(e => {
            counts[e.camera] = (counts[e.camera] || 0) + 1;
        });

        jsonCameraFilters.innerHTML = `
            <button class="json-cam-pill ${activeJsonCameraFilter === '' ? 'active' : ''}" data-camera="">
                All Cameras (${loadedJsonEvents.length})
            </button>
            ${cameras.map(c => `
                <button class="json-cam-pill ${activeJsonCameraFilter === c ? 'active' : ''}" data-camera="${escapeHtml(c)}">
                    ${escapeHtml(c)} (${counts[c] || 0})
                </button>
            `).join('')}
        `;

        jsonCameraFilters.querySelectorAll('.json-cam-pill').forEach(btn => {
            btn.addEventListener('click', () => {
                jsonCameraFilters.querySelectorAll('.json-cam-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeJsonCameraFilter = btn.dataset.camera;
                renderJsonExplorer();
            });
        });
    }

    function renderJsonExplorer() {
        const filtered = getFilteredJsonEvents();
        
        if (jsonMatchCount) {
            jsonMatchCount.textContent = `${filtered.length} / ${loadedJsonEvents.length} items`;
        }

        // Render Cards View
        if (jsonCardsContainer) {
            if (filtered.length === 0) {
                const camLabel = activeJsonCameraFilter ? ` for camera <strong>${escapeHtml(activeJsonCameraFilter)}</strong>` : '';
                jsonCardsContainer.innerHTML = `
                    <div class="empty-state" style="padding: 32px 16px; text-align: center;">
                        <p class="placeholder-text" style="font-size: 0.92rem;">No indexed event chunks${camLabel} yet.</p>
                        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 6px;">
                            Keyframes from active streams are automatically processed by Qwen3-VL and will appear here in real-time.
                        </p>
                    </div>
                `;
            } else {
                jsonCardsContainer.innerHTML = filtered.map((item, idx) => {
                    const imgUrl = item.image_url || (item.image_path ? '/' + item.image_path.replace(/\\/g, '/') : '');
                    const secs = item.seconds ?? (function(ts) {
                        if (!ts) return 0;
                        const p = ts.split(':').map(Number);
                        return p.length === 3 ? p[0]*3600 + p[1]*60 + p[2] : 0;
                    })(item.timestamp);

                    return `
                        <div class="json-event-card" data-camera="${escapeHtml(item.camera || '')}" data-seconds="${secs}" data-timestamp="${escapeHtml(item.timestamp || '')}" data-epoch="${item.epoch_time || ''}" data-image="${escapeHtml(imgUrl)}">
                            <div class="json-card-thumb-wrap" title="Click to preview in Surveillance Monitor">
                                ${imgUrl ? `<img src="${escapeHtml(imgUrl)}" class="json-card-thumb" alt="Frame preview" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">` : ''}
                                <div class="json-card-thumb-placeholder" style="${imgUrl ? 'display:none;' : ''}">
                                    <span>${escapeHtml(item.camera || 'CCTV')}</span>
                                </div>
                            </div>
                            <div class="json-card-main">
                                <div class="json-card-header">
                                    <span class="json-card-index">#${idx + 1}</span>
                                    <span class="json-card-cam">${escapeHtml(item.camera || 'UNKNOWN')}</span>
                                    <span class="json-card-ts">${escapeHtml(item.timestamp || '00:00:00')}</span>
                                </div>
                                <div class="json-card-desc">${escapeHtml(item.description || '')}</div>
                                <div class="json-card-source">${escapeHtml(item.image_path || '')}</div>
                            </div>
                            <div class="json-card-actions">
                                <button class="btn-card-seek" title="Redirect Surveillance Monitor to this camera moment">
                                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                                    <span>Seek</span>
                                </button>
                                <button class="btn-card-copy" title="Copy this chunk as JSON">
                                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                    <span>JSON</span>
                                </button>
                            </div>
                        </div>
                    `;
                }).join('');

                // Attach Card Listeners
                jsonCardsContainer.querySelectorAll('.json-event-card').forEach(card => {
                    const cam = card.dataset.camera;
                    const sec = parseFloat(card.dataset.seconds) || 0;
                    const ts = card.dataset.timestamp;
                    const epoch = card.dataset.epoch ? parseFloat(card.dataset.epoch) : null;
                    const img = card.dataset.image;

                    // Thumb click & Seek click
                    const thumbWrap = card.querySelector('.json-card-thumb-wrap');
                    const seekBtn = card.querySelector('.btn-card-seek');
                    const handleSeek = () => {
                        seekToTime(sec, ts, cam, img, epoch);
                    };
                    if (thumbWrap) thumbWrap.addEventListener('click', handleSeek);
                    if (seekBtn) seekBtn.addEventListener('click', handleSeek);

                    // Copy chunk click
                    const copyBtn = card.querySelector('.btn-card-copy');
                    if (copyBtn) {
                        copyBtn.addEventListener('click', async (e) => {
                            e.stopPropagation();
                            const chunkData = {
                                camera: cam,
                                timestamp: ts,
                                seconds: sec,
                                description: card.querySelector('.json-card-desc')?.textContent || '',
                                image_path: card.querySelector('.json-card-source')?.textContent || ''
                            };
                            try {
                                await navigator.clipboard.writeText(JSON.stringify(chunkData, null, 2));
                                const span = copyBtn.querySelector('span');
                                if (span) {
                                    span.textContent = 'Copied!';
                                    setTimeout(() => { span.textContent = 'JSON'; }, 1200);
                                }
                            } catch (err) {
                                alert('Copy failed: ' + err.message);
                            }
                        });
                    }
                });
            }
        }

        // Render Raw JSON View
        if (jsonCodeDisplay) {
            jsonCodeDisplay.textContent = JSON.stringify(filtered, null, 2);
        }
    }

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
                fetchEventsJson(false);
                fetchRtspStreamsStatus();
                checkHealth();
                
                // Live background synchronization for auto-indexed keyframes
                if (!autoRefreshInterval) {
                    autoRefreshInterval = setInterval(() => {
                        if (isDevModeActive) {
                            fetchEventsJson(true);
                            checkHealth();
                        }
                    }, 3500);
                }
            } else {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                }
            }
        });
    }

    // ------------------------------------------------------------------
    // Multi-Camera Surveillance Feed Switcher & Multi-Monitor View
    // ------------------------------------------------------------------
    const cameraFeedSwitcher = document.getElementById('camera-feed-switcher');
    const singleVideoWrapper = document.getElementById('single-video-wrapper');
    const allFeedsContainer = document.getElementById('all-feeds-container');
    const ytPlayer = document.getElementById('yt-player');
    const cctvSnapshotView = document.getElementById('cctv-snapshot-view');
    const snapshotScreenImg = document.getElementById('snapshot-screen-img');
    const monitorModeBar = document.getElementById('monitor-mode-bar');
    const btnShowEvidence = document.getElementById('btn-show-evidence');
    const btnShowLive = document.getElementById('btn-show-live');
    const evidenceBtnText = document.getElementById('evidence-btn-text');
    const liveBtnText = document.getElementById('live-btn-text');
    const hudCamId = document.getElementById('hud-cam-id');
    const hudTimestamp = document.getElementById('hud-timestamp');
    const hudStatus = document.getElementById('hud-status');
    const currentFeedName = document.getElementById('current-feed-name');
    const subSource = document.getElementById('sub-source');
    const subFps = document.getElementById('sub-fps');
    const subDuration = document.getElementById('sub-duration');

    let activeFeedCamId = 'CAM_01';
    let availableFeeds = [];
    let activeEvidenceState = null;

    // ------------------------------------------------------------------
    // Rock-Solid YouTube IFrame Player & Live Stream DVR Controller
    // ------------------------------------------------------------------
    let ytPlayerInstance = null;
    let isYtPlayerReady = false;
    let pendingDvrSeekOffset = null; // One-shot deltaSeconds to seek

    function getYtPlayerInstance(vidId, onReadyCb) {
        if (ytPlayerInstance && isYtPlayerReady && typeof ytPlayerInstance.seekTo === 'function') {
            if (onReadyCb) onReadyCb(ytPlayerInstance);
            return ytPlayerInstance;
        }

        if (window.YT && window.YT.Player) {
            try {
                ytPlayerInstance = new YT.Player('yt-player', {
                    videoId: vidId,
                    playerVars: {
                        autoplay: 1,
                        mute: 1,
                        enablejsapi: 1,
                        origin: window.location.origin
                    },
                    events: {
                        onReady: (event) => {
                            isYtPlayerReady = true;
                            if (pendingDvrSeekOffset !== null && pendingDvrSeekOffset > 0) {
                                executeOneShotSeek(event.target, pendingDvrSeekOffset);
                                pendingDvrSeekOffset = null;
                            }
                            if (onReadyCb) onReadyCb(event.target);
                        },
                        onStateChange: (event) => {
                            // When video starts playing, execute pending seek ONCE only
                            if (event.data === 1 && pendingDvrSeekOffset !== null && pendingDvrSeekOffset > 0) {
                                const offset = pendingDvrSeekOffset;
                                pendingDvrSeekOffset = null; // Clear immediately to prevent any looping!
                                executeOneShotSeek(event.target, offset);
                            }
                        }
                    }
                });
                return ytPlayerInstance;
            } catch (err) {
                console.warn('YT.Player instantiation error:', err);
            }
        }
        return null;
    }

    function executeOneShotSeek(player, deltaSeconds) {
        try {
            const cur = (typeof player.getCurrentTime === 'function') ? player.getCurrentTime() : 0;
            const dur = (typeof player.getDuration === 'function') ? player.getDuration() : 0;
            const ref = (dur && dur > 0) ? dur : cur;
            if (ref > 0 && deltaSeconds > 0) {
                const seekPos = Math.max(0, ref - deltaSeconds);
                player.seekTo(seekPos, true);
                player.playVideo();
            }
        } catch (err) {
            console.warn('executeOneShotSeek error:', err);
        }
    }

    function seekYouTubeLiveDVR(vidId, deltaSeconds, displayTs, camera_id) {
        if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
        if (cctvSnapshotView) { cctvSnapshotView.style.display = 'none'; }

        if (ytPlayer) {
            ytPlayer.style.display = 'block';
            const embedUrl = `https://www.youtube-nocookie.com/embed/${vidId}?autoplay=1&mute=1&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}`;
            if (!ytPlayer.src || !ytPlayer.src.includes(vidId)) {
                ytPlayer.src = embedUrl;
            }
        }

        // If player is already initialized and active, perform seek immediately
        if (ytPlayerInstance && isYtPlayerReady && typeof ytPlayerInstance.seekTo === 'function') {
            executeOneShotSeek(ytPlayerInstance, deltaSeconds);
        } else {
            // Queue one-shot seek for when player finishes handshake
            pendingDvrSeekOffset = deltaSeconds;
            getYtPlayerInstance(vidId, (player) => {
                if (pendingDvrSeekOffset !== null) {
                    executeOneShotSeek(player, pendingDvrSeekOffset);
                    pendingDvrSeekOffset = null;
                }
            });
        }

        if (seekNotice) {
            const mins = Math.round(deltaSeconds / 60);
            seekNotice.textContent = `▶️ Rewound Live Stream: -${mins}m (${displayTs})`;
            seekNotice.style.opacity = '1';
            setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
        }
    }

    if (btnShowEvidence && btnShowLive) {
        btnShowEvidence.addEventListener('click', () => {
            btnShowEvidence.classList.add('active');
            btnShowLive.classList.remove('active');
            if (ytPlayer) { ytPlayer.style.display = 'none'; }
            if (cctvPlayer) { cctvPlayer.style.display = 'none'; cctvPlayer.pause(); }
            if (cctvSnapshotView) {
                cctvSnapshotView.style.display = 'block';
                if (snapshotScreenImg && activeEvidenceState?.targetImage) {
                    snapshotScreenImg.src = activeEvidenceState.targetImage;
                }
            }
            if (hudStatus) hudStatus.textContent = '📸 MOMENT EVIDENCE';
        });

        btnShowLive.addEventListener('click', () => {
            btnShowLive.classList.add('active');
            btnShowEvidence.classList.remove('active');
            if (cctvSnapshotView) { cctvSnapshotView.style.display = 'none'; }
            if (cctvPlayer) { cctvPlayer.style.display = 'none'; cctvPlayer.pause(); }

            const feed = availableFeeds.find(f => f.camera_id === activeFeedCamId) || {};
            const isLiveStream = (feed.type === 'youtube_stream' || feed.type === 'rtsp_stream' || feed.stream_url?.includes('youtube.com') || feed.stream_url?.includes('youtu.be'));

            if (isLiveStream) {
                const vidUrl = feed.stream_url || feed.embed_url || '1EiC9bvVGnk';
                let vidId = '1EiC9bvVGnk';
                const match = vidUrl.match(/(?:v=|\/|embed\/|live\/)([0-9A-Za-z_-]{11})/);
                if (match) vidId = match[1];

                if (ytPlayer) {
                    ytPlayer.style.display = 'block';
                    const embedUrl = `https://www.youtube-nocookie.com/embed/${vidId}?autoplay=1&mute=1&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}`;
                    if (!ytPlayer.src || !ytPlayer.src.includes(vidId)) {
                        ytPlayer.src = embedUrl;
                    }
                }
                if (hudStatus) hudStatus.textContent = '🔴 LIVE STREAM';
            } else if (feed.type === 'video_file') {
                if (cctvPlayer) {
                    cctvPlayer.style.display = 'block';
                    cctvPlayer.play().catch(() => {});
                }
                if (hudStatus) hudStatus.textContent = 'PLAYING ● 1080P';
            }
        });
    }

    async function fetchCameraFeeds() {
        if (!cameraFeedSwitcher) return;
        try {
            const resp = await fetch('/api/cameras/feeds');
            if (resp.ok) {
                const data = await resp.json();
                availableFeeds = data.feeds || [];
                renderCameraFeedSwitcher();
                if (activeFeedCamId === 'ALL') {
                    renderAllFeedsGrid();
                }
            }
        } catch (err) {
            console.warn('Failed to load camera feeds:', err);
        }
    }

    function renderCameraFeedSwitcher() {
        if (!cameraFeedSwitcher) return;

        // First button: ALL FEEDS (Multi-Monitor View)
        let html = `
            <button class="feed-btn ${activeFeedCamId === 'ALL' ? 'active' : ''}" data-cam="ALL">
                <span>🔲 ALL FEEDS</span>
                <span class="feed-badge mp4">MULTI-VIEW</span>
            </button>
        `;

        html += availableFeeds.map(f => {
            const isActive = f.camera_id === activeFeedCamId;
            let badgeClass = 'feed-badge';
            let badgeText = f.status || 'FEED';
            if (f.type === 'youtube_stream' || f.status.includes('LIVE')) {
                badgeClass += ' live';
                badgeText = '🔴 LIVE';
            } else if (f.type === 'video_file') {
                badgeClass += ' mp4';
                badgeText = '📹 MP4';
            } else if (f.type === 'youtube_video') {
                badgeClass += ' mp4';
                badgeText = '▶️ YT';
            } else if (f.status === 'PAUSED') {
                badgeText = '⏸️ PAUSED';
            }

            return `
                <button class="feed-btn ${isActive ? 'active' : ''}" data-cam="${escapeHtml(f.camera_id)}">
                    <span>${escapeHtml(f.camera_id)}</span>
                    <span class="${badgeClass}">${badgeText}</span>
                </button>
            `;
        }).join('');

        cameraFeedSwitcher.innerHTML = html;

        cameraFeedSwitcher.querySelectorAll('.feed-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const camId = btn.dataset.cam;
                switchSurveillanceFeed(camId);
            });
        });
    }

    function switchSurveillanceFeed(camId, targetSeconds = null, targetImage = null, targetTimestamp = null, targetEpochTime = null) {
        activeFeedCamId = camId;
        renderCameraFeedSwitcher();

        // 1. If "ALL FEEDS" mode
        if (camId === 'ALL') {
            if (singleVideoWrapper) singleVideoWrapper.style.display = 'none';
            if (monitorModeBar) monitorModeBar.style.display = 'none';
            if (allFeedsContainer) {
                allFeedsContainer.style.display = 'flex';
                renderAllFeedsGrid();
            }
            if (currentFeedName) currentFeedName.textContent = 'Multi-Monitor View (All Feeds Active)';
            if (subSource) subSource.innerHTML = `Mode: <code>Multi-Feed Overview</code>`;
            if (subFps) subFps.innerHTML = `Total Feeds: <code>${availableFeeds.length}</code>`;
            if (subDuration) subDuration.innerHTML = `Status: <strong class="text-green">ALL ONLINE</strong>`;
            return;
        }

        // 2. Single Camera Focus Mode
        if (allFeedsContainer) allFeedsContainer.style.display = 'none';
        if (singleVideoWrapper) singleVideoWrapper.style.display = 'block';

        const feed = availableFeeds.find(f => f.camera_id === camId) || {
            camera_id: camId,
            name: camId,
            type: camId === 'CAM_01' ? 'video_file' : (camId === 'CAM_3000' ? 'youtube_stream' : 'snapshot'),
            src: camId === 'CAM_01' ? '/video/sample_cctv.mp4' : '',
            embed_url: camId === 'CAM_3000' ? 'https://www.youtube-nocookie.com/embed/1EiC9bvVGnk?autoplay=1&mute=1' : '',
            stream_url: camId === 'CAM_3000' ? 'https://www.youtube.com/watch?v=1EiC9bvVGnk' : '',
            fps: 30.0,
            duration: '24h CCTV',
        };

        if (currentFeedName) currentFeedName.textContent = `${feed.camera_id} (${feed.name || feed.camera_id})`;
        const displayTs = targetTimestamp || (targetSeconds !== null ? formatSecondsToTs(targetSeconds) : '00:00:00');

        // -------------------------------------------------------------
        // Case 1: Video File (e.g. CAM_01 / Local MP4)
        // -------------------------------------------------------------
        if (feed.type === 'video_file' || (feed.stream_url && feed.stream_url.endsWith('.mp4'))) {
            const maxDuration = feed.total_duration_sec || (cctvPlayer?.duration) || 811.0;
            const isOutOfBounds = (targetSeconds !== null && !isNaN(targetSeconds) && targetSeconds > maxDuration);

            // If outside duration bounds and snapshot image is available, display snapshot evidence
            if (isOutOfBounds && targetImage) {
                if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
                if (ytPlayer) { ytPlayer.style.display = 'none'; ytPlayer.src = ''; }
                if (monitorModeBar) monitorModeBar.style.display = 'none';

                if (cctvSnapshotView && snapshotScreenImg) {
                    cctvSnapshotView.style.display = 'block';
                    snapshotScreenImg.src = targetImage;
                }
                if (hudCamId) hudCamId.textContent = camId;
                if (hudTimestamp) hudTimestamp.textContent = displayTs;
                if (hudStatus) hudStatus.textContent = '📸 SNAPSHOT (OUT OF BOUNDS)';
                if (videoTimer) videoTimer.textContent = displayTs;
                if (seekNotice) {
                    seekNotice.textContent = `📸 Displaying archived snapshot for ${feed.camera_id} @ ${displayTs}`;
                    seekNotice.style.opacity = '1';
                    setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
                }
                return;
            }

            // Normal in-bounds video playback
            if (ytPlayer) { ytPlayer.style.display = 'none'; ytPlayer.src = ''; }
            if (cctvSnapshotView) cctvSnapshotView.style.display = 'none';
            if (monitorModeBar) monitorModeBar.style.display = 'none';

            if (cctvPlayer) {
                cctvPlayer.style.display = 'block';
                if (!cctvPlayer.src || !cctvPlayer.src.includes('sample_cctv.mp4')) {
                    cctvPlayer.src = '/video/sample_cctv.mp4';
                }
                if (targetSeconds !== null && !isNaN(targetSeconds)) {
                    cctvPlayer.currentTime = Math.min(targetSeconds, maxDuration);
                }
                cctvPlayer.play().catch(err => console.log('Playback started:', err));
            }

            if (hudCamId) hudCamId.textContent = camId;
            if (hudTimestamp) hudTimestamp.textContent = displayTs;
            if (hudStatus) hudStatus.textContent = 'PLAYING ● 1080P';
            if (videoTimer) videoTimer.textContent = displayTs;

            if (subSource) subSource.innerHTML = `Source: <code>${feed.src || 'sample_cctv.mp4'}</code>`;
            if (subFps) subFps.innerHTML = `FPS: <code>${feed.fps || 30.0}</code>`;
            if (subDuration) subDuration.innerHTML = `Duration: <code>${feed.duration || '13m 31s'}</code>`;

            if (seekNotice) {
                seekNotice.textContent = `▶️ Playing ${feed.camera_id} at ${displayTs}`;
                seekNotice.style.opacity = '1';
                setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
            }
            return;
        }

        // -------------------------------------------------------------
        // Case 2: Static / Recorded YouTube Video (Non-live with fixed duration)
        // -------------------------------------------------------------
        if (feed.type === 'youtube_video') {
            const vidUrl = feed.stream_url || feed.embed_url || '';
            let vidId = '';
            const match = vidUrl.match(/(?:v=|\/|embed\/)([0-9A-Za-z_-]{11})/);
            if (match) vidId = match[1];

            const maxDuration = feed.total_duration_sec || 3600;
            const isOutOfBounds = (targetSeconds !== null && !isNaN(targetSeconds) && targetSeconds > maxDuration);

            if (isOutOfBounds && targetImage) {
                if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
                if (ytPlayer) { ytPlayer.style.display = 'none'; ytPlayer.src = ''; }
                if (monitorModeBar) monitorModeBar.style.display = 'none';

                if (cctvSnapshotView && snapshotScreenImg) {
                    cctvSnapshotView.style.display = 'block';
                    snapshotScreenImg.src = targetImage;
                }
                if (hudCamId) hudCamId.textContent = camId;
                if (hudTimestamp) hudTimestamp.textContent = displayTs;
                if (hudStatus) hudStatus.textContent = '📸 SNAPSHOT (OUT OF BOUNDS)';
                return;
            }

            if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
            if (cctvSnapshotView) cctvSnapshotView.style.display = 'none';
            if (monitorModeBar) monitorModeBar.style.display = 'none';

            if (ytPlayer && vidId) {
                ytPlayer.style.display = 'block';
                const startSec = Math.floor(targetSeconds || 0);
                const embedUrl = `https://www.youtube-nocookie.com/embed/${vidId}?autoplay=1&mute=1&enablejsapi=1&start=${startSec}&origin=${encodeURIComponent(window.location.origin)}`;
                ytPlayer.src = embedUrl;
            }

            if (hudCamId) hudCamId.textContent = camId;
            if (hudTimestamp) hudTimestamp.textContent = displayTs;
            if (hudStatus) hudStatus.textContent = '▶️ YT VIDEO';
            return;
        }

        // -------------------------------------------------------------
        // Case 3: 24/7 Live Stream (YouTube Live / RTSP Camera)
        // -------------------------------------------------------------
        if (feed.type === 'youtube_stream' || feed.type === 'rtsp_stream' || (feed.stream_url && (feed.stream_url.includes('youtube.com') || feed.stream_url.includes('youtu.be')))) {
            const vidUrl = feed.stream_url || feed.embed_url || '1EiC9bvVGnk';
            let vidId = '1EiC9bvVGnk';
            const match = vidUrl.match(/(?:v=|\/|embed\/|live\/)([0-9A-Za-z_-]{11})/);
            if (match) vidId = match[1];

            // Compute delta in seconds: Current Wall Clock - Event Time
            let deltaSeconds = 0;
            const targetEpoch = targetEpochTime || activeEvidenceState?.targetEpochTime;

            if (targetEpoch && targetEpoch > 0) {
                // Use the synchronized clock offset to bypass client timezone and clock drift
                const adjustedNowSec = (Date.now() + serverTimeOffsetMs) / 1000;
                deltaSeconds = Math.max(0, adjustedNowSec - targetEpoch);
            } else if (targetTimestamp) {
                // Safe UTC-based fallback if epoch_time is not available
                const now = new Date();
                const nowSec = now.getUTCHours() * 3600 + now.getUTCMinutes() * 60 + now.getUTCSeconds();
                const parts = targetTimestamp.split(':').map(Number);
                const eventSec = parts.length === 3 ? parts[0]*3600 + parts[1]*60 + parts[2] : (parts.length === 2 ? parts[0]*60 + parts[1] : 0);

                let diff = nowSec - eventSec;
                if (diff < 0) diff += 86400; // midnight rollover
                deltaSeconds = diff;
            }

            activeEvidenceState = { camId, targetSeconds, targetImage, targetTimestamp, targetEpochTime, deltaSeconds, vidId };

            // When seeking a specific moment evidence:
            if (targetImage || targetTimestamp) {
                // If event happened within available live buffer (e.g. under 12 hours)
                if (deltaSeconds > 0 && deltaSeconds <= 12 * 3600) {
                    seekYouTubeLiveDVR(vidId, deltaSeconds, displayTs, camId);

                    if (hudCamId) hudCamId.textContent = camId;
                    const mins = Math.round(deltaSeconds / 60);
                    if (hudTimestamp) hudTimestamp.textContent = `${displayTs} (-${mins}m)`;
                    if (hudStatus) hudStatus.textContent = '▶️ PLAYING (DVR)';
                    if (videoTimer) videoTimer.textContent = displayTs;

                    if (monitorModeBar) {
                        monitorModeBar.style.display = 'flex';
                        if (btnShowEvidence) btnShowEvidence.classList.remove('active');
                        if (btnShowLive) btnShowLive.classList.add('active');
                        if (evidenceBtnText) evidenceBtnText.textContent = `📸 Snapshot Evidence (${displayTs})`;
                        if (liveBtnText) liveBtnText.textContent = `▶️ Rewound Video (-${mins}m)`;
                    }

                    if (subSource) subSource.innerHTML = `Source: <code>YouTube Live DVR (-${mins}m)</code>`;
                    if (subFps) subFps.innerHTML = `FPS: <code>30.0</code>`;
                    if (subDuration) subDuration.innerHTML = `Status: <strong class="text-green">▶️ REWOUND PLAYBACK</strong>`;
                    return;
                } else {
                    // Out-of-bounds (>12h old) -> Fallback to snapshot image
                    if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
                    if (ytPlayer) { ytPlayer.style.display = 'none'; }

                    if (cctvSnapshotView && snapshotScreenImg) {
                        cctvSnapshotView.style.display = 'block';
                        const imgPath = targetImage || `/data/cameras/${camId}/extracted_frames/${camId}_snapshot.jpg`;
                        snapshotScreenImg.src = imgPath;
                    }

                    if (hudCamId) hudCamId.textContent = camId;
                    if (hudTimestamp) hudTimestamp.textContent = displayTs;
                    if (hudStatus) hudStatus.textContent = '📸 ARCHIVED SNAPSHOT (>12h)';
                    if (videoTimer) videoTimer.textContent = displayTs;

                    if (monitorModeBar) {
                        monitorModeBar.style.display = 'flex';
                        if (btnShowEvidence) btnShowEvidence.classList.add('active');
                        if (btnShowLive) btnShowLive.classList.remove('active');
                        if (evidenceBtnText) evidenceBtnText.textContent = `📸 Archived Snapshot (${displayTs})`;
                        if (liveBtnText) liveBtnText.textContent = `🔴 Switch to Live Video`;
                    }
                    return;
                }
            }

            // Normal Live Stream View (Camera selected directly without specific moment)
            if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
            if (cctvSnapshotView) cctvSnapshotView.style.display = 'none';
            if (monitorModeBar) monitorModeBar.style.display = 'none';

            if (ytPlayer) {
                ytPlayer.style.display = 'block';
                const embedUrl = `https://www.youtube-nocookie.com/embed/${vidId}?autoplay=1&mute=1&enablejsapi=1&origin=${encodeURIComponent(window.location.origin)}`;
                if (!ytPlayer.src || !ytPlayer.src.includes(vidId)) {
                    ytPlayer.src = embedUrl;
                }
            }

            if (hudCamId) hudCamId.textContent = camId;
            if (hudTimestamp) hudTimestamp.textContent = displayTs;
            if (hudStatus) hudStatus.textContent = '🔴 LIVE STREAM';
            if (videoTimer) videoTimer.textContent = displayTs;

            if (subSource) subSource.innerHTML = `Source: <code>YouTube Live Stream</code>`;
            if (subFps) subFps.innerHTML = `FPS: <code>30.0</code>`;
            if (subDuration) subDuration.innerHTML = `Status: <strong class="text-green">🔴 LIVE STREAM</strong>`;

            if (seekNotice) {
                seekNotice.textContent = `🔴 Live Stream Active: ${feed.camera_id}`;
                seekNotice.style.opacity = '1';
                setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
            }
            return;
        }

        // Fallback: If snapshot-only feed or camera offline
        const cctvOfflineView = document.getElementById('cctv-offline-view');
        const offlineDescText = document.getElementById('offline-desc-text');
        const offlineCamTag = document.getElementById('offline-cam-tag');

        const isFeedAvailable = availableFeeds.some(f => f.camera_id === camId && f.status !== 'offline' && f.status !== 'PAUSED');
        
        if (!isFeedAvailable && camId !== 'CAM_01' && !targetImage) {
            if (cctvPlayer) { cctvPlayer.pause(); cctvPlayer.style.display = 'none'; }
            if (ytPlayer) { ytPlayer.style.display = 'none'; ytPlayer.src = ''; }
            if (cctvSnapshotView) { cctvSnapshotView.style.display = 'none'; }
            if (monitorModeBar) { monitorModeBar.style.display = 'none'; }

            if (cctvOfflineView) {
                cctvOfflineView.style.display = 'flex';
                if (offlineCamTag) offlineCamTag.textContent = camId;
                if (offlineDescText) {
                    offlineDescText.textContent = `Camera '${camId}' is currently offline or not transmitting. Select CAM_01 for active sample footage.`;
                }
            }

            if (currentFeedName) currentFeedName.innerHTML = `<span style="color: #ef4444;">${camId} (Offline / Not Available)</span>`;
            if (videoTimer) videoTimer.textContent = '--:--:--';
            if (subSource) subSource.innerHTML = `Status: <strong style="color: #ef4444;">OFFLINE</strong>`;
            if (subFps) subFps.innerHTML = `FPS: <code>0.0</code>`;
            if (subDuration) subDuration.innerHTML = `Feed: <code>Unavailable</code>`;
            return;
        }

        if (cctvOfflineView) cctvOfflineView.style.display = 'none';

        if (cctvPlayer) {
            cctvPlayer.pause();
            cctvPlayer.style.display = 'none';
        }
        if (ytPlayer) {
            ytPlayer.style.display = 'none';
            ytPlayer.src = '';
        }
        if (monitorModeBar) monitorModeBar.style.display = 'none';

        if (cctvSnapshotView && snapshotScreenImg) {
            cctvSnapshotView.style.display = 'block';
            const imgPath = targetImage || feed.preview_image || `/data/cameras/${camId}/extracted_frames/${camId}_snapshot.jpg`;
            snapshotScreenImg.src = imgPath;

            if (hudCamId) hudCamId.textContent = camId;
            if (hudTimestamp) hudTimestamp.textContent = displayTs;
            if (hudStatus) hudStatus.textContent = 'REC ● 1080P';
            if (videoTimer) videoTimer.textContent = displayTs;

            if (subSource) subSource.innerHTML = `Source: <code>${imgPath.split('/').pop()}</code>`;
            if (subFps) subFps.innerHTML = `FPS: <code>${feed.fps || 15.0}</code>`;
            if (subDuration) subDuration.innerHTML = `Frame: <code>${displayTs}</code>`;

            if (seekNotice) {
                seekNotice.textContent = `Surveillance View — ${camId} @ ${displayTs}`;
                seekNotice.style.opacity = '1';
                setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
            }
        }
    }

    function renderAllFeedsGrid() {
        if (!allFeedsContainer) return;
        if (availableFeeds.length === 0) {
            allFeedsContainer.innerHTML = `<div class="empty-state"><p>No camera feeds available.</p></div>`;
            return;
        }

        allFeedsContainer.innerHTML = availableFeeds.map(f => {
            let screenContent = '';
            if (f.type === 'video_file') {
                screenContent = `
                    <video controls muted autoplay loop style="width:100%; height:100%; object-fit:contain;">
                        <source src="/video/sample_cctv.mp4" type="video/mp4">
                    </video>
                `;
            } else if (f.type === 'youtube_stream') {
                const embedUrl = f.embed_url || 'https://www.youtube-nocookie.com/embed/1EiC9bvVGnk?autoplay=1&mute=1';
                screenContent = `
                    <iframe src="${embedUrl}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="width:100%; height:100%; border:none;"></iframe>
                `;
            } else {
                const img = f.preview_image || `/data/cameras/${f.camera_id}/extracted_frames/${f.camera_id}_snapshot.jpg`;
                screenContent = `
                    <img src="${img}" alt="${escapeHtml(f.camera_id)}" style="width:100%; height:100%; object-fit:contain;">
                `;
            }

            return `
                <div class="all-feed-card">
                    <div class="all-feed-header">
                        <div class="all-feed-title">
                            <span class="status-indicator ${f.status.includes('LIVE') ? 'online' : (f.status === 'PAUSED' ? 'warning' : 'online')}"></span>
                            <strong>${escapeHtml(f.camera_id)}</strong> — ${escapeHtml(f.name)}
                        </div>
                        <button class="btn-primary focus-feed-btn" data-cam="${escapeHtml(f.camera_id)}" style="padding: 2px 8px; font-size: 0.72rem;">
                            Focus Feed
                        </button>
                    </div>
                    <div class="all-feed-screen">
                        ${screenContent}
                    </div>
                </div>
            `;
        }).join('');

        allFeedsContainer.querySelectorAll('.focus-feed-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const camId = btn.dataset.cam;
                switchSurveillanceFeed(camId);
            });
        });
    }

    function formatSecondsToTs(totalSecs) {
        const s = Math.floor(totalSecs || 0);
        const hrs = Math.floor(s / 3600);
        const mins = Math.floor((s % 3600) / 60);
        const secs = s % 60;
        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    fetchCameraFeeds();
    fetchLazyVectors();

    // ------------------------------------------------------------------
    // Video Timer Tracking for MP4 Player
    // ------------------------------------------------------------------
    if (cctvPlayer) {
        cctvPlayer.addEventListener('timeupdate', () => {
            if (cctvPlayer.style.display !== 'none') {
                const cur = Math.floor(cctvPlayer.currentTime);
                videoTimer.textContent = formatSecondsToTs(cur);
            }
        });
    }

    function seekToTime(seconds, label = '', camera = 'CAM_01', imagePath = '', epochTime = null) {
        switchSurveillanceFeed(camera, seconds, imagePath, label, epochTime);
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
    // Execute Search API Call & Populate 4-Stage Pipeline & Vector Debugger
    // ------------------------------------------------------------------
    async function executeSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        searchBtn.innerHTML = `<span>Searching…</span>`;

        // 1. Activate Live 4-Stage Horizontal Pipeline
        if (lazyPipelineBadge) {
            lazyPipelineBadge.textContent = 'EXECUTING ● ACTIVE';
            lazyPipelineBadge.className = 'badge-lazy running';
        }
        if (lazyPulseDot) lazyPulseDot.className = 'status-pulse-dot running';

        // Stage 1: Active
        if (pstage1) { pstage1.className = 'stage-box active'; }
        if (pstage1Status) pstage1Status.textContent = 'RUNNING';
        if (pstage1Time) pstage1Time.textContent = 'Embedding…';
        if (pstage1Detail) pstage1Detail.textContent = 'Generating 512-D MobileCLIP tensor';

        // Stages 2, 3, 4: Standby
        if (pstage2) { pstage2.className = 'stage-box'; }
        if (pstage2Status) pstage2Status.textContent = 'QUEUED';
        if (pstage2Time) pstage2Time.textContent = 'Queued';

        if (pstage3) { pstage3.className = 'stage-box'; }
        if (pstage3Status) pstage3Status.textContent = 'QUEUED';
        if (pstage3Time) pstage3Time.textContent = 'Queued';

        if (pstage4) { pstage4.className = 'stage-box'; }
        if (pstage4Status) pstage4Status.textContent = 'QUEUED';
        if (pstage4Time) pstage4Time.textContent = 'Queued';

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
            renderLazyPipelineTrace(data.debug_trace, data);
        } catch (err) {
            console.error('Search error:', err);
            aiAnswerBody.innerHTML = `<p style="color: #dc2626;">Search failed: ${err.message}</p>`;
            evidenceList.innerHTML = `<div class="empty-state"><p style="color: #dc2626;">Failed to load results.</p></div>`;
            if (lazyPipelineBadge) {
                lazyPipelineBadge.textContent = 'ERROR ● FAILED';
                lazyPipelineBadge.className = 'badge-lazy';
            }
            if (lazyPulseDot) lazyPulseDot.className = 'status-pulse-dot';
            if (pstage1Status) pstage1Status.textContent = 'ERROR';
        } finally {
            searchBtn.disabled = false;
            searchBtn.innerHTML = `<span>Search Video</span>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>`;
        }
    }

    // Render Real-time 4-Stage Horizontal Live Operating Pipeline
    function renderLazyPipelineTrace(trace, data) {
        if (!trace) return;
        const timings = trace.timings_ms || {};

        // Stage 1: Done
        if (pstage1) pstage1.className = 'stage-box done';
        if (pstage1Status) pstage1Status.textContent = 'COMPLETE';
        if (pstage1Time) pstage1Time.textContent = `${timings.query_embedding_ms || 0} ms`;
        if (pstage1Detail) pstage1Detail.textContent = `Norm ||v||: ${(trace.query_vector_norm || 1.0).toFixed(4)} (512-D)`;

        // Stage 2: Done
        if (pstage2) pstage2.className = 'stage-box done';
        if (pstage2Status) pstage2Status.textContent = 'COMPLETE';
        if (pstage2Time) pstage2Time.textContent = `${timings.faiss_retrieval_ms || timings.temporal_retrieval_ms || 0} ms`;
        let topScore = 'Matched';
        if (data.results && data.results.length > 0 && data.results[0].faiss_score !== undefined) {
            topScore = Number(data.results[0].faiss_score).toFixed(4);
        } else if (data.storyboard && data.storyboard.length > 0) {
            const anchor = data.storyboard.find(f => f.is_anchor) || data.storyboard[0];
            topScore = Number(anchor.score || 0).toFixed(4);
        }
        if (pstage2Detail) pstage2Detail.textContent = `Top Cosine Similarity: ${topScore}`;

        // Stage 3: Done
        if (pstage3) pstage3.className = 'stage-box done';
        if (pstage3Status) pstage3Status.textContent = 'COMPLETE';
        if (pstage3Time) pstage3Time.textContent = `${timings.temporal_expansion_ms || 0.5} ms`;
        const epFramesCount = (data.storyboard && data.storyboard.length) ? data.storyboard.length : 3;
        if (pstage3Detail) pstage3Detail.textContent = `Window: ${epFramesCount} Chronological Frames`;

        // Stage 4: Done
        if (pstage4) pstage4.className = 'stage-box done';
        if (pstage4Status) pstage4Status.textContent = 'COMPLETE';
        if (pstage4Time) {
            const vlmSec = ((timings.vlm_reasoning_ms || timings.llm_generation_ms || 0) / 1000).toFixed(2);
            pstage4Time.textContent = `${vlmSec} s (${timings.vlm_reasoning_ms || timings.llm_generation_ms || 0} ms)`;
        }
        if (pstage4Detail) pstage4Detail.textContent = `Engine: Qwen3-VL 4B (llama-server)`;

        // Overall status
        if (lazyPipelineBadge) {
            lazyPipelineBadge.textContent = 'SUCCESS ● COMPLETE';
            lazyPipelineBadge.className = 'badge-lazy';
        }
        if (lazyPulseDot) lazyPulseDot.className = 'status-pulse-dot';

        if (lazyStatLatency) {
            const totalSec = ((timings.total_ms || 0) / 1000).toFixed(2);
            lazyStatLatency.textContent = `${totalSec} s (${timings.total_ms || 0} ms)`;
        }

        // Highlight matched vector cards in Frame-to-Vector Grounding Inspector
        if (frameVectorsGrid && data && data.storyboard) {
            const topTimestamps = new Set(data.storyboard.map(f => f.timestamp));
            frameVectorsGrid.querySelectorAll('.frame-vector-card').forEach(card => {
                const ts = card.dataset.ts;
                if (topTimestamps.has(ts)) {
                    card.classList.add('matched-hit');
                } else {
                    card.classList.remove('matched-hit');
                }
            });
        }
    }

    // Render Vector Debugger Trace in Tab 1 (512-D MobileCLIP + Qwen3-VL)
    let lastQueryVectorArray = [];
    function renderVectorDebugger(trace) {
        if (!trace) return;

        if (vNorm) vNorm.textContent = (trace.query_vector_norm ?? 1.0).toFixed(4);
        if (vectorArrayDisplay) {
            const sample = trace.query_vector_sample || [];
            lastQueryVectorArray = sample;
            vectorArrayDisplay.textContent = `[${sample.join(', ')}, ... (${trace.query_vector_dim || 512} continuous dimensions)]`;
        }

        const timings = trace.timings_ms || {};
        if (tEmbed) tEmbed.textContent = `${timings.query_embedding_ms ?? 0} ms`;
        if (tFaiss) tFaiss.textContent = `${timings.faiss_retrieval_ms ?? timings.temporal_retrieval_ms ?? 0} ms`;
        if (tExpandTiming) tExpandTiming.textContent = `${timings.temporal_expansion_ms ?? 0.5} ms`;
        if (tLlm) {
            const vlmSec = ((timings.vlm_reasoning_ms ?? timings.llm_generation_ms ?? 0) / 1000).toFixed(2);
            tLlm.textContent = `${vlmSec} s (${timings.vlm_reasoning_ms ?? timings.llm_generation_ms ?? 0} ms)`;
        }
        if (tTotalTrace) {
            const totalSec = ((timings.total_ms ?? 0) / 1000).toFixed(2);
            tTotalTrace.textContent = `${totalSec} s (${timings.total_ms ?? 0} ms)`;
        }

        if (promptPreviewDisplay) {
            promptPreviewDisplay.textContent = trace.prompt_constructed || "No prompt generated.";
        }
    }

    if (copyVecBtn) {
        copyVecBtn.addEventListener('click', () => {
            if (lastQueryVectorArray.length > 0) {
                navigator.clipboard.writeText(JSON.stringify(lastQueryVectorArray));
                copyVecBtn.textContent = 'Copied!';
                setTimeout(() => { copyVecBtn.textContent = 'Copy'; }, 2000);
            }
        });
    }

    // ------------------------------------------------------------------
    // Intelligent AI Security Analysis Markdown & Interactive Timestamp Formatter
    // ------------------------------------------------------------------
    function formatAiSecurityAnalysis(rawText, topResults = []) {
        if (!rawText) return '<p class="ai-para">No analysis generated.</p>';

        const defaultCam = topResults.length > 0 ? topResults[0].camera : 'CAM_01';
        const buttonTokens = [];

        function registerTsButton(cam, ts, customLabel = null) {
            const parts = ts.split(':').map(Number);
            let secs = 0;
            if (parts.length === 3) secs = parts[0]*3600 + parts[1]*60 + parts[2];
            else if (parts.length === 2) secs = parts[0]*60 + parts[1];

            const labelText = customLabel || `${cam} | ${ts}`;
            const idx = buttonTokens.length;
            const btnHtml = `<button class="answer-ts-btn" data-camera="${escapeHtml(cam)}" data-timestamp="${escapeHtml(ts)}" data-seconds="${secs}" title="Jump video footage to ${escapeHtml(cam)} @ ${escapeHtml(ts)}"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>${escapeHtml(labelText)}</span></button>`;
            buttonTokens.push(btnHtml);
            return `@@@TS_BTN_${idx}@@@`;
        }

        // Parse line by line to maintain document structure and infer context per event line
        const lines = rawText.split('\n');
        const formattedLines = lines.map(line => {
            const trimmed = line.trim();
            if (!trimmed) return '';

            // Detect camera name in this line
            const camMatch = line.match(/\b(CAM_[A-Za-z0-9_]+)\b/);
            const lineCam = camMatch ? camMatch[1] : defaultCam;

            let processed = line;

            // 1. Match combined (Camera + Timestamp): e.g. "**CAM_02 | 08:10:05**" or "CAM_01 @ 00:01:30"
            processed = processed.replace(/(?:\*\*)?(CAM_[A-Za-z0-9_]+)\s*(?:\||@|at|:)\s*(\d{1,2}:\d{2}(?::\d{2})?)(?:\*\*)?/gi, (match, cam, ts) => {
                return registerTsButton(cam, ts, `${cam} | ${ts}`);
            });

            // 2. Match timestamp followed by camera in parentheses: e.g. "**08:10:05** (CAM_02)" or "08:10:05 (CAM_02)"
            processed = processed.replace(/(?:\*\*)?(\d{1,2}:\d{2}(?::\d{2})?)(?:\*\*)?\s*\((CAM_[A-Za-z0-9_]+)\)/gi, (match, ts, cam) => {
                return registerTsButton(cam, ts, `${ts} (${cam})`);
            });

            // 3. Match standalone timestamps: e.g. "**08:10:05**" or "00:01:30"
            processed = processed.replace(/(?:\*\*)?(\b\d{1,2}:\d{2}:\d{2}\b|\b\d{1,2}:\d{2}\b)(?:\*\*)?/g, (match, ts) => {
                return registerTsButton(lineCam, ts, ts);
            });

            // 4. Highlight remaining standalone camera mentions
            processed = processed.replace(/\b(CAM_[A-Za-z0-9_]+)\b/g, `<strong class="text-blue">$1</strong>`);

            // 5. Convert standard Markdown (**bold**, *italic*)
            processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            processed = processed.replace(/\*(.*?)\*/g, '<em>$1</em>');

            // 6. Substitute all registered button placeholders back in
            processed = processed.replace(/@@@TS_BTN_(\d+)@@@/g, (m, idx) => {
                return buttonTokens[parseInt(idx, 10)] || '';
            });

            // 7. Format bullet items vs normal paragraphs
            if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                const content = processed.replace(/^\s*[-*]\s*/, '');
                return `<div class="ai-bullet-item"><span class="ai-bullet-dot">▪</span><div>${content}</div></div>`;
            }

            return `<p class="ai-para">${processed}</p>`;
        });

        return formattedLines.filter(Boolean).join('');
    }

    // ------------------------------------------------------------------
    // Helper: Parse Timestamp string to Seconds
    // ------------------------------------------------------------------
    function parseTsToSeconds(ts, fallbackSec = 0) {
        if (fallbackSec && Number(fallbackSec) > 0) return Number(fallbackSec);
        if (!ts) return 0;
        try {
            const parts = String(ts).trim().split(':');
            if (parts.length === 3) {
                return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
            } else if (parts.length === 2) {
                return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
            }
            return parseFloat(ts) || 0;
        } catch (e) {
            return 0;
        }
    }

    // ------------------------------------------------------------------
    // Render Results in Classic Light UI
    // ------------------------------------------------------------------
    function renderResults(data) {
        const results = data.results || [];
        evidenceCount.textContent = results.length;

        // 1. Render AI Security Analysis + Forensic Storyboard Strip
        let storyboardHtml = '';
        if (data.storyboard && data.storyboard.length > 0) {
            storyboardHtml = `
                <div class="storyboard-container">
                    <div class="storyboard-header">
                        <span class="storyboard-title">🎬 Chronological Forensic Storyboard (${data.storyboard.length} frames)</span>
                        <span class="storyboard-hint">Click any frame to jump player</span>
                    </div>
                    <div class="storyboard-strip">
                        ${data.storyboard.map((f, i) => {
                            const secVal = parseTsToSeconds(f.timestamp, f.seconds);
                            return `
                            <div class="storyboard-card ${f.is_anchor ? 'anchor-frame' : ''}" data-seconds="${secVal}" data-timestamp="${escapeHtml(f.timestamp)}" data-epoch="${f.epoch_time || ''}" data-image="${escapeHtml(f.image_path || '')}" data-camera="${escapeHtml(f.camera || '')}" title="${f.is_anchor ? 'Target Match Moment' : 'Surrounding Context Frame'} @ ${escapeHtml(f.timestamp)}">
                                <div class="storyboard-thumb-wrap">
                                    ${f.image_path ? `<img src="${escapeHtml(f.image_path)}" class="storyboard-thumb" alt="Frame ${i+1}" onerror="this.style.display='none';">` : ''}
                                    ${f.is_anchor ? `<span class="anchor-badge">TARGET</span>` : ''}
                                </div>
                                <div class="storyboard-meta">
                                    <span class="storyboard-ts">${escapeHtml(f.timestamp)}</span>
                                </div>
                            </div>
                        `;}).join('')}
                    </div>
                </div>
            `;
        }

        const formattedAnswer = formatAiSecurityAnalysis(data.answer, results);
        aiAnswerBody.innerHTML = `<div class="animate-slide-in">${storyboardHtml}${formattedAnswer}</div>`;

        // Attach Click Listeners to Storyboard Cards
        aiAnswerBody.querySelectorAll('.storyboard-card').forEach(card => {
            card.addEventListener('click', (e) => {
                e.preventDefault();
                aiAnswerBody.querySelectorAll('.storyboard-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');

                const ts = card.dataset.timestamp || '';
                const sec = parseTsToSeconds(ts, parseFloat(card.dataset.seconds) || 0);
                const cam = card.dataset.camera || (results.length > 0 ? results[0].camera : 'CAM_01');
                const img = card.dataset.image || '';
                const epoch = card.dataset.epoch ? parseFloat(card.dataset.epoch) : null;
                seekToTime(sec, ts, cam, img, epoch);
            });
        });

        // 2. Attach Click Listeners to Interactive Timestamp Buttons in LLM Output
        aiAnswerBody.querySelectorAll('.answer-ts-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();

                // Highlight active button
                aiAnswerBody.querySelectorAll('.answer-ts-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const ts = btn.dataset.timestamp || '';
                const sec = parseTsToSeconds(ts, parseFloat(btn.dataset.seconds) || 0);
                const cam = btn.dataset.camera || (results.length > 0 ? results[0].camera : 'CAM_01');

                // Lookup matching result item to get high-res moment image and epoch timestamp
                const matchingItem = results.find(r => r.camera === cam && (r.timestamp === ts || Math.abs(parseTsToSeconds(r.timestamp, r.seconds) - sec) < 2));
                const img = matchingItem ? (matchingItem.image_path || '') : '';
                const epoch = matchingItem && matchingItem.epoch_time ? parseFloat(matchingItem.epoch_time) : null;

                // Seek and display/play video footage or evidence snapshot with live DVR rewind
                seekToTime(sec, ts, cam, img, epoch);
            });
        });

        // 3. Render Evaluation Metrics
        if (data.evaluation) {
            const ev = data.evaluation;
            const ret = ev.retrieval || ev.retrieval_metrics || {};
            const ans = ev.answer || ev.answer_metrics || {};
            mPrecision.textContent = (ret.precision_at_k !== undefined ? ret.precision_at_k : 0.0).toFixed(2);
            mMrr.textContent = (ret.mrr !== undefined ? ret.mrr : 0.0).toFixed(2);
            mNdcg.textContent = (ret.ndcg_at_k !== undefined ? ret.ndcg_at_k : 0.0).toFixed(2);
            mUtil.textContent = `${Math.round((ans.context_utilization !== undefined ? ans.context_utilization : 0.0) * 100)}%`;
            metricsBar.style.display = 'grid';
        }

        // 4. Render Retrieved Video Moments
        if (results.length === 0) {
            evidenceList.innerHTML = `<div class="empty-state"><p>No relevant video moments found for this camera/query filter.</p></div>`;
            return;
        }

        evidenceList.innerHTML = results.map((item) => {
            const secVal = parseTsToSeconds(item.timestamp, item.seconds);
            return `
            <div class="evidence-item" data-seconds="${secVal}" data-timestamp="${item.timestamp}" data-epoch="${item.epoch_time || ''}" data-image="${escapeHtml(item.image_path || '')}" data-camera="${escapeHtml(item.camera)}" data-feed-type="${escapeHtml(item.feed_type || '')}">
                <div class="evidence-meta">
                    <div class="ev-header">
                        <span class="ev-rank">#${item.rank}</span>
                        <span class="ev-camera">${escapeHtml(item.camera)}</span>
                        <span class="ev-ts">${escapeHtml(item.timestamp)}</span>
                    </div>
                    <div class="ev-desc">${escapeHtml(item.description)}</div>
                </div>
                <div class="ev-scores">
                    <span class="score-badge rerank">Rerank: ${item.rerank_score}</span>
                    <span class="score-badge">FAISS: ${item.faiss_score}</span>
                    <button class="btn-seek" title="Play video / seek DVR to this moment">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                        Seek
                    </button>
                </div>
            </div>
        `;}).join('');

        // 5. Attach Click Listeners to Evidence Cards and Seek Buttons
        evidenceList.querySelectorAll('.evidence-item').forEach(card => {
            const handleItemSeek = (e) => {
                if (e) e.stopPropagation();
                evidenceList.querySelectorAll('.evidence-item').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                const ts = card.dataset.timestamp || '';
                const secs = parseTsToSeconds(ts, parseFloat(card.dataset.seconds) || 0);
                const img = card.dataset.image || '';
                const cam = card.dataset.camera || 'CAM_01';
                const epoch = card.dataset.epoch ? parseFloat(card.dataset.epoch) : null;

                // Seek and play footage with DVR calculation
                seekToTime(secs, ts, cam, img, epoch);
            };

            card.addEventListener('click', handleItemSeek);
            const seekBtn = card.querySelector('.btn-seek');
            if (seekBtn) {
                seekBtn.addEventListener('click', handleItemSeek);
            }
        });
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
