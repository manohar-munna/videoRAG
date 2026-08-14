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
        },
        rtsp_manager: {
            title: "Async Multi-Threaded RTSP Stream Capture Engine",
            body: "Captures live CCTV network feeds (RTSP/RTMP) on non-blocking background producer threads using a <strong>Size-1 Ring Buffer</strong>. Eliminates streaming lag, handles automatic socket reconnection, and extracts keyframes via edge dHash filtering without stalling server GIL."
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
            let statusBadge = isRunning ? (isConnected ? 'LIVE (TCP)' : 'RECONNECTING') : (isPaused ? 'PAUSED' : 'STOPPED');
            let statusColor = isRunning && isConnected ? 'text-green' : (isPaused ? 'text-blue' : 'text-dim');

            return `
                <div class="stream-card ${isRunning ? 'active' : ''}">
                    <div class="stream-card-header">
                        <div class="stream-cam-title">
                            <span class="status-indicator ${isConnected ? 'online' : (isPaused ? 'warning' : 'offline')}"></span>
                            ${escapeHtml(s.camera_id)} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(${escapeHtml(s.name || s.camera_id)})</span>
                        </div>
                        <span class="feed-badge ${isRunning ? 'live' : ''}">${statusBadge}</span>
                    </div>
                    <div class="stream-url-tag">URL: ${escapeHtml(s.stream_url)}</div>
                    <div class="stream-metrics-grid">
                        <div class="sm-item"><span class="sm-label">Status</span><span class="sm-val ${statusColor}">${statusBadge}</span></div>
                        <div class="sm-item"><span class="sm-label">FPS</span><span class="sm-val">${s.fps}</span></div>
                        <div class="sm-item"><span class="sm-label">Frames Read</span><span class="sm-val">${s.total_frames_read}</span></div>
                        <div class="sm-item"><span class="sm-label">Ring Dropped</span><span class="sm-val text-dim">${s.total_frames_dropped}</span></div>
                        <div class="sm-item"><span class="sm-label">Keyframes Kept</span><span class="sm-val text-blue">${s.keyframes_kept}</span></div>
                        <div class="sm-item"><span class="sm-label">Compute Saved</span><span class="sm-val text-green">${s.llm_compute_saved_pct}%</span></div>
                    </div>
                    <div class="stream-card-actions">
                        <button class="btn-primary index-stream-btn" data-cam="${escapeHtml(s.camera_id)}" style="padding: 3px 8px; font-size: 0.72rem;" ${s.keyframes_kept === 0 ? 'disabled' : ''}>
                            ⚡ Index Keyframes (${s.keyframes_kept})
                        </button>
                        ${isPaused ? `
                            <button class="btn-stream-action btn-resume-stream resume-stream-btn" data-cam="${escapeHtml(s.camera_id)}">
                                ▶️ Resume Extraction
                            </button>
                        ` : `
                            <button class="btn-stream-action btn-pause-stream pause-stream-btn" data-cam="${escapeHtml(s.camera_id)}">
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
                fetchRtspStreamsStatus();
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
    const hudCamId = document.getElementById('hud-cam-id');
    const hudTimestamp = document.getElementById('hud-timestamp');
    const hudStatus = document.getElementById('hud-status');
    const currentFeedName = document.getElementById('current-feed-name');
    const subSource = document.getElementById('sub-source');
    const subFps = document.getElementById('sub-fps');
    const subDuration = document.getElementById('sub-duration');

    let activeFeedCamId = 'CAM_01';
    let availableFeeds = [];

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

    function switchSurveillanceFeed(camId, targetSeconds = null, targetImage = null, targetTimestamp = null) {
        activeFeedCamId = camId;
        renderCameraFeedSwitcher();

        // 1. If "ALL FEEDS" mode
        if (camId === 'ALL') {
            if (singleVideoWrapper) singleVideoWrapper.style.display = 'none';
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
            fps: 30.0,
            duration: '24h CCTV',
        };

        if (currentFeedName) currentFeedName.textContent = `${feed.camera_id} (${feed.name})`;

        // Video File (e.g. CAM_01)
        if (feed.type === 'video_file') {
            if (cctvPlayer) {
                cctvPlayer.style.display = 'block';
                if (targetSeconds !== null && !isNaN(targetSeconds)) {
                    cctvPlayer.currentTime = targetSeconds;
                    cctvPlayer.play().catch(() => {});
                }
            }
            if (ytPlayer) {
                ytPlayer.style.display = 'none';
                ytPlayer.src = '';
            }
            if (cctvSnapshotView) cctvSnapshotView.style.display = 'none';

            if (subSource) subSource.innerHTML = `Source: <code>sample_cctv.mp4</code>`;
            if (subFps) subFps.innerHTML = `FPS: <code>${feed.fps || 30.0}</code>`;
            if (subDuration) subDuration.innerHTML = `Duration: <code>${feed.duration || '13m 31s'}</code>`;

            if (targetSeconds !== null && seekNotice) {
                seekNotice.textContent = `Seeked ${feed.camera_id} to ${targetTimestamp || targetSeconds + 's'}`;
                seekNotice.style.opacity = '1';
                setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
            }
            return;
        }

        // YouTube Live Stream (e.g. CAM_3000)
        if (feed.type === 'youtube_stream' && !targetImage) {
            if (cctvPlayer) {
                cctvPlayer.pause();
                cctvPlayer.style.display = 'none';
            }
            if (cctvSnapshotView) cctvSnapshotView.style.display = 'none';
            if (ytPlayer) {
                ytPlayer.style.display = 'block';
                const embedUrl = feed.embed_url || 'https://www.youtube-nocookie.com/embed/1EiC9bvVGnk?autoplay=1&mute=1';
                if (!ytPlayer.src || !ytPlayer.src.includes('1EiC9bvVGnk')) {
                    ytPlayer.src = embedUrl;
                }
            }

            if (subSource) subSource.innerHTML = `Source: <code>YouTube Live Stream</code>`;
            if (subFps) subFps.innerHTML = `FPS: <code>30.0</code>`;
            if (subDuration) subDuration.innerHTML = `Status: <strong class="text-green">🔴 LIVE STREAM</strong>`;

            if (seekNotice) {
                seekNotice.textContent = `Active Stream: ${feed.camera_id} (YouTube Live Feed)`;
                seekNotice.style.opacity = '1';
                setTimeout(() => { seekNotice.style.opacity = '0'; }, 3500);
            }
            return;
        }

        // Snapshot / Keyframe Image Display
        if (cctvPlayer) {
            cctvPlayer.pause();
            cctvPlayer.style.display = 'none';
        }
        if (ytPlayer) {
            ytPlayer.style.display = 'none';
            ytPlayer.src = '';
        }

        if (cctvSnapshotView && snapshotScreenImg) {
            cctvSnapshotView.style.display = 'block';
            const imgPath = targetImage || feed.preview_image || `/data/cameras/${camId}/extracted_frames/${camId}_snapshot.jpg`;
            snapshotScreenImg.src = imgPath;

            const ts = targetTimestamp || (targetSeconds !== null ? formatSecondsToTs(targetSeconds) : '00:00:00');
            if (hudCamId) hudCamId.textContent = camId;
            if (hudTimestamp) hudTimestamp.textContent = ts;
            if (hudStatus) hudStatus.textContent = 'REC ● 1080P';
            if (videoTimer) videoTimer.textContent = ts;

            if (subSource) subSource.innerHTML = `Source: <code>${imgPath.split('/').pop()}</code>`;
            if (subFps) subFps.innerHTML = `FPS: <code>${feed.fps || 15.0}</code>`;
            if (subDuration) subDuration.innerHTML = `Frame: <code>${ts}</code>`;

            if (seekNotice) {
                seekNotice.textContent = `Surveillance View — ${camId} @ ${ts}`;
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

    function seekToTime(seconds, label = '', camera = 'CAM_01', imagePath = '') {
        switchSurveillanceFeed(camera, seconds, imagePath, label);
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
            return `<span class="ev-ts interactive-ts" data-seconds="${secs}" style="cursor:pointer;" title="Click to redirect monitor to ${match}">${match}</span>`;
        });

        formattedAnswer = formattedAnswer.replace(/(\bCAM_\w+\b)/g, `<strong class="text-blue">$1</strong>`);

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

        const results = data.results || [];
        evidenceCount.textContent = results.length;

        // Auto-detect camera from primary top result for interactive timestamp clicks in answer
        const topCam = results.length > 0 ? results[0].camera : 'CAM_01';
        aiAnswerBody.querySelectorAll('.interactive-ts').forEach(el => {
            el.addEventListener('click', () => {
                const sec = parseFloat(el.dataset.seconds);
                seekToTime(sec, el.textContent, topCam);
            });
        });

        if (results.length === 0) {
            evidenceList.innerHTML = `<div class="empty-state"><p>No relevant video moments found for this camera/query filter.</p></div>`;
            return;
        }

        evidenceList.innerHTML = results.map((item) => `
            <div class="evidence-item" data-seconds="${item.seconds}" data-timestamp="${item.timestamp}" data-image="${escapeHtml(item.image_path || '')}" data-camera="${escapeHtml(item.camera)}" data-feed-type="${escapeHtml(item.feed_type || '')}">
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
                    <button class="btn-seek" title="Redirect monitor to this camera moment">
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
                const img = card.dataset.image;
                const cam = card.dataset.camera;

                seekToTime(secs, ts, cam, img);
            });
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
