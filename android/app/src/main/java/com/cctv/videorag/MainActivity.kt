package com.cctv.videorag

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.widget.NestedScrollView
import androidx.lifecycle.lifecycleScope
import com.cctv.videorag.indexing.*
import com.cctv.videorag.ingestion.*
import com.cctv.videorag.llm.*
import com.cctv.videorag.ui.ChatView
import com.cctv.videorag.ui.DebugPanel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/**
 * Conversational front end: import a video, then ask questions about it in a thread.
 *
 * Replaces the previous single-shot form (one query box, one result blob, a storyboard
 * strip and a wall of telemetry). Answers carry forward as context, so follow-ups like
 * "what colour was it?" resolve against the previous turn, and timestamps the model
 * cites are tappable and seek the inline player.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var orchestrator: MemoryOrchestrator
    private val vectorStore = MobileVectorStore()
    private val sqliteFts by lazy { SQLiteFtsHelper(this) }
    private val frameDecoder by lazy { VideoFrameDecoder(this) }

    // ── ingestion state ───────────────────────────────────────────
    private var lastFrameHash: Long? = null
    private var acceptedFramesCount = 0
    private var droppedFramesCount = 0
    private val hashThreshold = MobileFrameFilter.DEFAULT_HAMMING_THRESHOLD
    private var currentVideoUri: Uri? = null
    /** Stable key for the imported video, so its saved index can be found again. */
    private var currentVideoKey: String = ""
    private var lastSelectedTimestampMs = 0
    private var isBusy = false
    private var downloading = false

    /** Sampling rate. Fixed at 1 FPS now that the FPS pills are gone. */
    private val sampleFps = 1.0

    /**
     * Cosine above which two retrieved frames count as the same shot and the later one
     * is not worth spending ~17 s to encode. See dropNearDuplicates().
     */
    private val MAX_KEEP_SIMILARITY = 0.92f

    /**
     * Below this CLIP score the subject is treated as absent - see answerQuestion().
     *
     * Calibrated against the measured distribution over the 463-keyframe index, using
     * the full-caption score (not the max-pooled one):
     *
     *   white truck                        top 0.236   present
     *   what is written on the truck       top 0.235   present
     *   people wearing pink costumes       top 0.225   present
     *   an elephant wearing a hat          top 0.196   absent
     *   a red double decker bus in the snow top 0.163  absent
     *
     * Present and absent separate at 0.225 / 0.196, so 0.21 sits between them with room
     * either side. The previous 0.19 was set on a sparser index and fell just below the
     * elephant case, which would have been answered as though it were there.
     *
     * Note a relative rule - top score against the index's own median - was measured and
     * is WORSE, not better: absent queries score low across the board, so their
     * top-minus-median margin (0.082, 0.094) is LARGER than a present query's (0.053 to
     * 0.063). The absolute score is what carries the signal.
     */
    private val MIN_RELEVANCE = 0.21f

    /**
     * Cap on the extra "also matched" timestamps listed under an answer. Enough to restore
     * the coverage lost by analysing 3 frames instead of 5, without turning the footer into
     * a wall of weak hits.
     */
    private val MAX_ALSO_MATCHED = 6

    /**
     * Minimum gap between two frames sent to the model, in seconds. See
     * dropNearDuplicates(). Small enough that two genuinely different events seconds
     * apart both survive; large enough that consecutive samples of one shot do not.
     */
    private val MIN_SECONDS_APART = 5

    /** Prior turns, oldest first, fed back to the model so follow-ups have context. */
    private val conversation = mutableListOf<ConversationTurn>()

    // Diagnostics captured per query, surfaced by the debug panel rather than on screen.
    private var lastQuery: String? = null
    private var lastHits: List<Triple<String, String, Float>> = emptyList()
    private var lastFramesSent: List<String> = emptyList()
    private var lastDroppedTimestamps: List<String> = emptyList()
    private var lastGenStats: String = ""
    private var lastLatencyMs: Long? = null
    private var tokenizerStatus: String = "not run"
    private var modelHintShown = false

    // ── views ─────────────────────────────────────────────────────
    private lateinit var scrollView: NestedScrollView
    private lateinit var chatContainer: LinearLayout
    private lateinit var tvModelBadge: TextView
    private lateinit var tvIngestionInfo: TextView
    private lateinit var pbIngestion: ProgressBar
    private lateinit var btnPickVideo: Button
    private lateinit var btnClearAll: Button
    private lateinit var btnSelectModelFolder: Button
    private lateinit var btnDebug: Button
    private lateinit var etQuery: EditText
    private lateinit var btnSearch: Button
    private lateinit var cardVideoPlayback: CardView
    private lateinit var tvPlayerTimestamp: TextView
    private lateinit var videoViewPlayback: VideoView
    private lateinit var btnClosePlayer: Button
    private lateinit var btnPlayPause: Button
    private lateinit var btnReplayTimestamp: Button

    private val pickVideoLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { currentVideoUri = it; processSelectedVideoUri(it) }
        }

    private val selectFolderLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
            uri?.let { treeUri ->
                try {
                    contentResolver.takePersistableUriPermission(
                        treeUri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                } catch (_: Exception) {}
                val docId = android.provider.DocumentsContract.getTreeDocumentId(treeUri)
                val split = docId.split(":")
                val realPath = if (split.size > 1) {
                    if ("primary".equals(split[0], true)) "/storage/emulated/0/${split[1]}"
                    else "/storage/${split[0]}/${split[1]}"
                } else docId

                // the customModelDirectory setter loads the model, so keep it off Main
                lifecycleScope.launch(Dispatchers.IO) {
                    val vlm = orchestrator.getFrameDescriber()
                    vlm.customModelDirectory = realPath
                    val ok = vlm.isNativeGGUFAvailable()
                    withContext(Dispatchers.Main) {
                        updateModelBadge()
                        ChatView.addSystemNote(
                            chatContainer,
                            if (ok) "Model files found in $realPath" else "No .gguf files in $realPath",
                            isError = !ok
                        )
                    }
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initViews()
        // Both models live in one permission-free directory; see ModelPaths.
        orchestrator = MemoryOrchestrator(
            this,
            ModelPaths.modelsDir(this).absolutePath,
            ModelPaths.modelsDir(this).absolutePath
        )
        setupListeners()
        checkAndRequestStoragePermission()
        runTokenizerSelfTest()
        updateModelBadge()
        ChatView.addSystemNote(chatContainer, "Import a video, then ask questions about it.")
        restoreLastIndexIfAny()
    }

    override fun onResume() {
        super.onResume()
        updateModelBadge()
    }

    private fun initViews() {
        scrollView = findViewById(R.id.mainScrollView)
        chatContainer = findViewById(R.id.layoutChatMessages)
        tvModelBadge = findViewById(R.id.tvModelBadge)
        tvIngestionInfo = findViewById(R.id.tvIngestionInfo)
        pbIngestion = findViewById(R.id.pbIngestion)
        btnPickVideo = findViewById(R.id.btnPickVideo)
        btnClearAll = findViewById(R.id.btnClearAll)
        btnSelectModelFolder = findViewById(R.id.btnSelectModelFolder)
        btnDebug = findViewById(R.id.btnDebug)
        etQuery = findViewById(R.id.etQuery)
        btnSearch = findViewById(R.id.btnSearch)
        cardVideoPlayback = findViewById(R.id.cardVideoPlayback)
        tvPlayerTimestamp = findViewById(R.id.tvPlayerTimestamp)
        videoViewPlayback = findViewById(R.id.videoViewPlayback)
        btnClosePlayer = findViewById(R.id.btnClosePlayer)
        btnPlayPause = findViewById(R.id.btnPlayPause)
        btnReplayTimestamp = findViewById(R.id.btnReplayTimestamp)
    }

    private fun setupListeners() {
        btnPickVideo.setOnClickListener { pickVideoLauncher.launch("video/*") }
        btnSelectModelFolder.setOnClickListener { selectFolderLauncher.launch(null) }
        tvModelBadge.setOnClickListener { if (tvModelBadge.text == "Download models") downloadModels() }
        btnClearAll.setOnClickListener { resetAllData() }
        btnDebug.setOnClickListener { showDebugPanel() }

        btnSearch.setOnClickListener { submitQuestion() }
        etQuery.setOnEditorActionListener { _, _, _ -> submitQuestion(); true }

        btnClosePlayer.setOnClickListener {
            videoViewPlayback.stopPlayback()
            cardVideoPlayback.visibility = View.GONE
        }
        btnPlayPause.setOnClickListener {
            if (videoViewPlayback.isPlaying) {
                videoViewPlayback.pause(); btnPlayPause.text = "▶"
            } else {
                videoViewPlayback.start(); btnPlayPause.text = "⏸"
            }
        }
        btnReplayTimestamp.setOnClickListener {
            videoViewPlayback.seekTo(lastSelectedTimestampMs)
            videoViewPlayback.start()
            btnPlayPause.text = "⏸"
        }
        videoViewPlayback.setOnPreparedListener { it.isLooping = false }
        videoViewPlayback.setOnErrorListener { _, w, e ->
            Log.w("VideoView", "playback error what=$w extra=$e"); true
        }
    }

    private fun checkAndRequestStoragePermission() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R &&
            !android.os.Environment.isExternalStorageManager()
        ) {
            try {
                startActivity(
                    android.content.Intent(
                        android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                )
            } catch (_: Exception) {}
        }
    }

    /**
     * Reports whether the model weights are present, and offers to fetch them if not.
     *
     * Network-free: it checks local files only (a GGUF the native loader accepts, plus
     * both CLIP towers), so it is cheap to call on every resume. The manifest fetch that
     * decides exactly what to download happens later, only when the user taps the badge.
     * A fresh install therefore shows "Download models" rather than a dead "No model".
     */
    private fun updateModelBadge() {
        if (downloading) return
        lifecycleScope.launch(Dispatchers.IO) {
            val gguf = try { orchestrator.getFrameDescriber().isNativeGGUFAvailable() }
                       catch (_: Throwable) { false }
            val clip = ModelPaths.find(this@MainActivity) {
                it.name == "mobileclip_image.onnx" || it.name == "mobileclip_image.int8.onnx"
            } != null && ModelPaths.find(this@MainActivity) {
                it.name == "mobileclip_text.onnx" || it.name == "mobileclip_text.int8.onnx"
            } != null
            val ok = gguf && clip
            val canDownload = !ok && ModelDownloader.isConfigured(this@MainActivity)
            withContext(Dispatchers.Main) {
                tvModelBadge.text = when {
                    ok -> "Model ready"
                    canDownload -> "Download models"
                    else -> "No model"
                }
                if (!ok && !canDownload && !modelHintShown) {
                    modelHintShown = true
                    ChatView.addSystemNote(chatContainer, ModelPaths.instructions(this@MainActivity), isError = true)
                }
            }
        }
    }

    /**
     * Pull the missing weights from the model server, showing progress on the ingestion
     * bar. On success the badge flips to "Model ready"; on failure the partial file is
     * kept so a retry resumes rather than restarting.
     */
    private fun downloadModels() {
        if (downloading) return
        downloading = true
        tvModelBadge.text = "Downloading…"
        pbIngestion.visibility = View.VISIBLE
        pbIngestion.isIndeterminate = true
        tvIngestionInfo.text = "Preparing model download…"
        ChatView.addSystemNote(chatContainer, "Downloading on-device models. This is a one-time ~2 GB download; keep the app open.")
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                ModelDownloader.downloadMissing(this@MainActivity) { p ->
                    val doneMb = p.fileDone / 1_000_000
                    val totMb = if (p.fileBytes > 0) p.fileBytes / 1_000_000 else 0
                    val pct = if (p.fileBytes > 0) ((p.fileDone * 100) / p.fileBytes).toInt() else 0
                    lifecycleScope.launch(Dispatchers.Main) {
                        pbIngestion.isIndeterminate = false
                        pbIngestion.progress = pct
                        tvIngestionInfo.text = "Downloading ${p.name} (${p.index + 1}/${p.total}) — $doneMb/$totMb MB"
                    }
                }
                withContext(Dispatchers.Main) {
                    downloading = false
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "Models downloaded. Import a video to start."
                    ChatView.addSystemNote(chatContainer, "Models ready.")
                    modelHintShown = false
                    updateModelBadge()
                }
            } catch (e: Throwable) {
                withContext(Dispatchers.Main) {
                    downloading = false
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "Model download failed."
                    ChatView.addSystemNote(chatContainer,
                        "Model download failed: ${e.message}. Tap the badge to resume.", isError = true)
                    updateModelBadge()
                }
            }
        }
    }

    /** The Kotlin CLIP BPE port must match Python's ids or retrieval silently degrades. */
    private fun runTokenizerSelfTest() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val tk = ClipTokenizer.fromAssets(this@MainActivity)
                val r = ClipTokenizerSelfTest.run(this@MainActivity, tk)
                tokenizerStatus = if (r.ok) "CLIP BPE, ${r.passed}/${r.passed} match Python" else "MISMATCH (${r.failed} failed)"
                if (!r.ok) withContext(Dispatchers.Main) {
                    ChatView.addSystemNote(
                        chatContainer,
                        "Tokenizer self-test failed (${r.failed}) — text search may be unreliable",
                        isError = true
                    )
                }
            } catch (e: Throwable) {
                Log.e("VideoRAG_Tokenizer", "self-test error", e)
            }
        }
    }

    private fun resetAllData() {
        vectorStore.clear()
        sqliteFts.clearAll()
        sqliteFts.clearVectors()          // otherwise the saved index is restored again
        acceptedFramesCount = 0
        droppedFramesCount = 0
        lastFrameHash = null
        currentVideoUri = null
        conversation.clear()
        videoViewPlayback.stopPlayback()
        cardVideoPlayback.visibility = View.GONE
        try {
            File(filesDir, "extracted_frames").takeIf { it.exists() }?.deleteRecursively()
        } catch (_: Exception) {}
        ChatView.clear(chatContainer)
        ChatView.addSystemNote(chatContainer, "Cleared. Import a video to start again.")
        tvIngestionInfo.text = "Import a video to start."
        pbIngestion.visibility = View.GONE
    }

    // ── ingestion ─────────────────────────────────────────────────

    /**
     * Bring back the last index at launch, without waiting for a re-import.
     *
     * The vectors and the keyframe JPEGs both survive an app restart - and a reinstall,
     * since app data is kept - but until now the index only loaded as a side effect of
     * picking the video again, so every launch looked like an empty app sitting on top of
     * a perfectly good index. That also made testing expensive: each reinstall cost a
     * manual re-import before anything could be asked.
     *
     * Questions work immediately from this state. Only the inline player needs the
     * content URI, which does not survive the process, so playVideoAt() falls back to
     * "No video loaded" until the video is imported again - the note below says so.
     */
    private fun restoreLastIndexIfAny() {
        lifecycleScope.launch(Dispatchers.Default) {
            val key = sqliteFts.mostRecentVideoKey() ?: return@launch
            val saved = sqliteFts.loadMoments(key)
            if (saved.isEmpty()) return@launch
            vectorStore.clear()
            saved.forEach { vectorStore.addMoment(it) }
            currentVideoKey = key
            val frames = saved.distinctBy { it.imagePath }.size
            acceptedFramesCount = frames
            withContext(Dispatchers.Main) {
                tvIngestionInfo.text = "$frames keyframes - ${vectorStore.size} vectors (restored)"
                ChatView.addSystemNote(
                    chatContainer,
                    "Restored the last index: $frames keyframes. Ask a question now, " +
                    "or import the video again to play clips from timestamps."
                )
            }
        }
    }

    private fun processSelectedVideoUri(uri: Uri) {
        currentVideoUri = uri
        currentVideoKey = uri.toString()
        val startedAt = System.currentTimeMillis()

        lifecycleScope.launch(Dispatchers.Default) {
            // If this video was indexed before, load it back rather than spending ~7
            // minutes re-encoding frames that have not changed.
            val saved = sqliteFts.loadMoments(currentVideoKey)
            if (saved.isNotEmpty()) {
                vectorStore.clear()
                saved.forEach { vectorStore.addMoment(it) }
                val frames = saved.distinctBy { it.imagePath }.size
                val onDisk = saved.count { File(it.imagePath).exists() }
                acceptedFramesCount = frames
                conversation.clear()
                withContext(Dispatchers.Main) {
                    ChatView.clear(chatContainer)
                    tvIngestionInfo.text = "$frames keyframes · ${vectorStore.size} vectors (restored)"
                    ChatView.addSystemNote(
                        chatContainer,
                        "Restored a saved index for this video: $frames keyframes. Ask a question below."
                    )
                    if (onDisk < saved.size) ChatView.addSystemNote(
                        chatContainer,
                        "Some keyframe images are missing from storage; tap Reset to re-index.",
                        isError = true
                    )
                }
                return@launch
            }
            vectorStore.clear(); sqliteFts.clearAll()
            acceptedFramesCount = 0; droppedFramesCount = 0; lastFrameHash = null
            conversation.clear()

            withContext(Dispatchers.Main) {
                isBusy = true
                ChatView.clear(chatContainer)
                ChatView.addSystemNote(chatContainer, "Indexing video…")
                pbIngestion.visibility = View.VISIBLE
                pbIngestion.isIndeterminate = true
                cardVideoPlayback.visibility = View.GONE
            }

            try {
                frameDecoder.decodeVideoUri(
                    videoUri = uri,
                    cameraName = "cam_user",
                    sampleFps = sampleFps,
                    onProgress = { cur, total, _ ->
                        lifecycleScope.launch(Dispatchers.Main) {
                            pbIngestion.isIndeterminate = false
                            pbIngestion.progress = if (total > 0) ((cur * 100) / total).toInt() else 0
                            tvIngestionInfo.text = String.format(
                                Locale.US, "Indexing %02d:%02d / %02d:%02d — %d keyframes kept",
                                cur / 60, cur % 60, total / 60, total % 60, acceptedFramesCount
                            )
                        }
                    },
                    onKeyframeDecoded = { bmp, ts, epoch, path ->
                        ingestAndIndexFrame(bmp, "cam_user", ts, epoch, path)
                    }
                )
                val secs = (System.currentTimeMillis() - startedAt) / 1000
                withContext(Dispatchers.Main) {
                    isBusy = false
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text =
                        "$acceptedFramesCount keyframes · ${vectorStore.size} vectors · ${droppedFramesCount} duplicates dropped"
                    ChatView.addSystemNote(
                        chatContainer,
                        "Indexed $acceptedFramesCount keyframes in ${secs}s. Ask a question below."
                    )
                }
            } catch (e: Throwable) {
                Log.e("VideoRAG_Ingest", "Ingestion failed", e)
                val isModel = e is OnDeviceEmbedder.ModelUnavailableException ||
                              e.cause is OnDeviceEmbedder.ModelUnavailableException
                withContext(Dispatchers.Main) {
                    isBusy = false
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "Import failed."
                    ChatView.addSystemNote(
                        chatContainer,
                        (if (isModel) "CLIP model error: " else "Could not read video: ") +
                        (e.message ?: e.javaClass.simpleName),
                        isError = true
                    )
                }
            }
        }
    }

    /** dHash gate, then index the frame as 6 CLIP-embedded spatial regions. */
    private suspend fun ingestAndIndexFrame(
        bitmap: Bitmap, camera: String, timestamp: String, epochTime: Long, imagePath: String
    ) {
        val currentHash = MobileFrameFilter.calculateDHash(bitmap)
        if (!MobileFrameFilter.isKeyframe(lastFrameHash, currentHash, hashThreshold)) {
            droppedFramesCount++
            return  // lastFrameHash not advanced: compare against the last KEPT frame
        }
        lastFrameHash = currentHash
        acceptedFramesCount++

        // cheap per-frame stats only; the VLM is not loaded during ingestion
        val vlm = orchestrator.getFrameDescriber()
        val frameJson = vlm.describeFrameAsJson(bitmap, timestamp, acceptedFramesCount, imagePath)
        val frameDescription = frameJson.getString("visual_description")

        val embedder = orchestrator.getActiveEmbedder()
        for (crop in SpatialCropper.generatePyramidCrops(bitmap)) {
            val v = embedder.embedImage(crop.bitmap)
            val moment = IndexedMoment(
                id = "${camera}_${timestamp}_${crop.label}",
                camera = camera, timestamp = timestamp, epochTime = epochTime,
                vector = v, cropRegion = crop.label, imagePath = imagePath,
                description = frameDescription, jsonMetadata = frameJson.toString()
            )
            vectorStore.addMoment(moment)
            sqliteFts.saveMoment(moment, currentVideoKey)   // survives restart
            if (crop.bitmap != bitmap) crop.bitmap.recycle()
        }

        sqliteFts.insertMoment(
            momentId = "${camera}_${timestamp}", camera = camera, timestamp = timestamp,
            epochTime = epochTime, cropRegion = "frame", imagePath = imagePath,
            visualTokens = frameDescription
        )
    }

    // ── conversation ──────────────────────────────────────────────

    private fun submitQuestion() {
        val q = etQuery.text.toString().trim()
        if (q.isEmpty()) return
        if (isBusy) {
            Toast.makeText(this, "Still working — one moment", Toast.LENGTH_SHORT).show()
            return
        }
        if (vectorStore.size == 0) {
            ChatView.addUserMessage(chatContainer, q)
            ChatView.addSystemNote(chatContainer, "Import a video first.", isError = true)
            etQuery.setText(""); scrollToBottom()
            return
        }

        etQuery.setText("")
        hideKeyboard()
        ChatView.addUserMessage(chatContainer, q)
        ChatView.addPending(chatContainer)
        scrollToBottom()
        isBusy = true
        btnSearch.isEnabled = false

        lastQuery = q
        val startedAt = System.currentTimeMillis()
        lifecycleScope.launch(Dispatchers.Default) {
            val result = try { answerQuestion(q) }
                         catch (e: Throwable) {
                             Log.e("VideoRAG_Query", "query failed", e)
                             Answer("Something went wrong: ${e.message ?: e.javaClass.simpleName}", emptyList())
                         }
            lastLatencyMs = System.currentTimeMillis() - startedAt
            conversation.add(ConversationTurn(q, result.text))
            withContext(Dispatchers.Main) {
                isBusy = false
                btnSearch.isEnabled = true
                ChatView.replacePending(
                    chatContainer, result.text,
                    onTimestamp = { seconds -> playVideoAt(seconds) },
                    frames = result.frames
                )
                scrollToBottom()
            }
        }
    }

    /** Retrieve with CLIP (max-pooled across regions), then answer with the VLM. */
    private data class Answer(val text: String, val frames: List<ChatView.FrameRef>)

    private suspend fun answerQuestion(question: String): Answer {
        val embedder = orchestrator.getActiveEmbedder()

        // Each frame is indexed as 6 regions; collapse to the best region per frame so a
        // small object competes on its strongest crop rather than the whole scene.
        val bestPerFrame = HashMap<String, Pair<IndexedMoment, Float>>()
        val queryVectors = embedder.embedTextVariants(question)
        for ((moment, score) in vectorStore.searchMulti(queryVectors, topK = 60)) {
            val prev = bestPerFrame[moment.imagePath]
            if (prev == null || score > prev.second) bestPerFrame[moment.imagePath] = moment to score
        }
        val ranked = bestPerFrame.values.sortedByDescending { it.second }
        Log.i("VideoRAG_Query", "top=" + ranked.take(5).joinToString {
            "${it.first.timestamp}[${it.first.cropRegion}]=%.3f".format(it.second)
        })
        lastHits = ranked.take(10).map { Triple(it.first.timestamp, it.first.cropRegion, it.second) }
        if (ranked.isEmpty()) {
            return Answer("I couldn't find anything matching that in this video.", emptyList())
        }

        // Answer "not in this video" from the retrieval scores, before spending three
        // minutes of VLM time proving it the hard way.
        //
        // search() always returns its top K, however poor the match, so without this the
        // best of a bad set is handed to the model as though it were evidence - and a
        // 2B VLM shown five irrelevant frames will generally describe something rather
        // than object. Measured over the 13-minute clip after the caption fix:
        //   "white truck"              (present) 0.233
        //   "people in pink costumes"  (present) 0.218
        //   "red double decker bus in the snow" (absent) 0.163, falling to 0.106
        // Present subjects sit around 0.22-0.23 and absent ones near 0.16, so 0.19
        // separates them with margin on both sides. Deliberately set low: a false
        // "not found" is worse than a slow answer, because the user cannot tell whether
        // the footage lacks the subject or the search failed.
        // Judge presence on the FULL caption, not the max-pooled score.
        //
        // embedTextVariants() also searches the bare head noun to protect recall, and
        // searchMulti() keeps whichever scores higher. That is right for ranking and wrong
        // for deciding whether the subject is in the video at all: "a red double decker bus
        // in the snow" reduces to a head-noun variant of "a photo of a snow", which scores
        // above 0.19 against ordinary daylight frames even though nothing in the clip
        // resembles the question. The app then answered "White truck with a black fence on
        // the back at 00:02:41 ..." to a query about a bus in snow.
        //
        // The full caption is what the user actually asked, so it is what absence is
        // measured against. One extra scan over the vectors already in memory.
        val primary = vectorStore.search(queryVectors.first(), topK = 400)
        val topScore = primary.firstOrNull()?.second ?: 0f
        val primaryScores = primary.map { it.second }.sortedDescending()
        val median = if (primaryScores.isEmpty()) 0f
                     else primaryScores[primaryScores.size / 2]
        val p90 = if (primaryScores.isEmpty()) 0f
                  else primaryScores[(primaryScores.size * 10) / 100]
        Log.i("VideoRAG_Query", "scores: n=%d top=%.3f p90=%.3f median=%.3f margin=%.3f"
            .format(primaryScores.size, topScore, p90, median, topScore - median))
        if (topScore < MIN_RELEVANCE) {
            Log.i("VideoRAG_Query", "top score %.3f < %.2f - answering absent"
                .format(topScore, MIN_RELEVANCE))
            return Answer(
                "I couldn't find anything matching that in this video. " +
                "The closest frame was at ${ranked.first().first.timestamp}, but it is " +
                "not a strong enough match to report.",
                emptyList()
            )
        }

        // Drop near-duplicate hits before they reach the model.
        //
        // Surveillance retrieval is highly redundant: asking for "white truck" over a
        // static camera returns several frames of the same parked truck in the same
        // scene. Each one costs ~17 s to encode and adds nothing the previous frame did
        // not already show. The CLIP vectors are already computed, so cosine similarity
        // between selected frames is free - keep a frame only if it differs enough from
        // everything already chosen.
        val moments = dropNearDuplicates(ranked, MAX_KEEP_SIMILARITY)
        // the VLM only looks at the first MAX_FRAMES_TO_ANALYSE, in time order - mirror
        // that exactly so the strip shows what was actually sent, not what was retrieved
        val sent = moments.sortedBy { it.timestamp }
        lastFramesSent = sent.map { it.timestamp }
        // Log what is actually sent, not just the pre-dedup ranking above. dropNearDuplicates
        // runs over the whole ranked list, so the frames the model sees can legitimately
        // differ from the logged top-5 - which makes an answer's timestamps look invented
        // when they are not. Every timestamp in an answer should be checkable against this.
        Log.i("VideoRAG_Query", "frames sent to VLM: " +
            sent.take(OnDeviceVLM.MAX_FRAMES_TO_ANALYSE).joinToString {
                "${it.timestamp}[${it.cropRegion}]"
            })

        // CLIP has done its one job for this query (embedText, above); retrieval since
        // then was pure Kotlin. Drop ~400 MB of fp32 towers before the VLM starts
        // generating, which is the memory-tightest phase of the whole app.
        orchestrator.releaseEmbedder()

        val vlm = orchestrator.getActiveVLM()
        val text = vlm.answerFromRetrievedContext(
            query = question,
            top5Moments = moments,
            history = conversation.takeLast(3)
        )
        lastDroppedTimestamps = vlm.lastDroppedTimestamps
        lastGenStats = vlm.lastGenStats

        // Report the matches the model did not get to look at.
        //
        // Only MAX_FRAMES_TO_ANALYSE frames are encoded, because each costs ~18 s, but
        // retrieval has already scored every frame in the index for free. Cutting 5 to 3
        // therefore made answers cite three occurrences where they used to cite five - the
        // timestamps stayed true, the list stopped being complete, and "when does the truck
        // appear" quietly under-reports. That is the wrong failure for a forensic tool.
        //
        // Deliberately worded as matching the SEARCH, not as containing the subject: these
        // frames cleared the same relevance bar as the analysed ones, but no model looked
        // at them, so claiming the subject is present would be inventing evidence - the
        // thing dropUnsupportedTimestamps() exists to prevent. They are still tappable, so
        // the user can check them directly.
        val analysed = sent.take(OnDeviceVLM.MAX_FRAMES_TO_ANALYSE).map { it.timestamp }.toSet()
        // take() before sorted(), not after: `ranked` is in score order, so sorting first
        // would cap the list to the EARLIEST matches rather than the strongest. That is
        // what it did initially - "white truck" listed 00:02:54 through 00:06:38 and
        // dropped 00:10:54, a frame the model had itself described as showing the truck.
        // Same spacing rule as the frames themselves. At 1 fps the above-threshold hits
        // are dominated by consecutive seconds of one shot, so an unfiltered list read
        // "00:03:40, 00:03:42, 00:05:27, 00:07:21, 00:07:23, 00:07:25" - six entries
        // covering three moments. Keep the strongest of each cluster.
        val spread = mutableListOf<String>()
        for ((m, sc) in ranked) {
            if (sc < MIN_RELEVANCE) continue
            val ts = m.timestamp
            if (ts in analysed) continue
            val secs = parseTimestampSeconds(ts)
            if (spread.any { kotlin.math.abs(parseTimestampSeconds(it) - secs) < MIN_SECONDS_APART }) continue
            if (analysed.any { kotlin.math.abs(parseTimestampSeconds(it) - secs) < MIN_SECONDS_APART }) continue
            spread += ts
        }
        val candidates = spread
        val alsoMatched = candidates.take(MAX_ALSO_MATCHED).sorted()
        val more = candidates.size - alsoMatched.size
        val fullText = if (alsoMatched.isEmpty()) text
                       else text + "\nAlso matched this search, not analysed: " +
                            alsoMatched.joinToString(", ") +
                            (if (more > 0) " (+$more more)" else "")
        // Show the strip what the model was actually shown. Since the VLM now receives the
        // matched region rather than the whole frame, displaying imagePath here would put
        // a different picture under the answer than the one it was reasoning about, which
        // defeats the point of the strip as a way to check the answer.
        return Answer(fullText, sent.map {
            ChatView.FrameRef(vlm.regionCropPath(it), it.timestamp, parseTimestampSeconds(it.timestamp))
        })
    }

    /**
     * Keep a retrieved frame only if it is visually distinct from the ones already kept.
     *
     * Cosine on the CLIP image vectors we already hold. 0.92 is deliberately permissive:
     * it removes frames of the same static scene while keeping a frame where the subject
     * has moved or the scene has changed, which is the information a viewer actually
     * wants across a long recording.
     */
    private fun dropNearDuplicates(
        ranked: List<Pair<IndexedMoment, Float>>,
        maxSimilarity: Float
    ): List<IndexedMoment> {
        val kept = mutableListOf<IndexedMoment>()
        for ((moment, _) in ranked) {
            val tooSimilar = kept.any { cosine(it.vector, moment.vector) > maxSimilarity }
            // Cosine alone is not enough now that sampling is genuinely 1 fps. Adjacent
            // seconds of the same shot score below 0.92 - CLIP notices small movement -
            // so they survive the similarity filter and the ranking clusters on one
            // moment. Asked "what is written on the truck" the top five were 00:07:22,
            // 00:03:41, 00:07:21, 00:07:23, 00:03:40, and two of the three frames sent
            // were one second apart: a third of the budget spent twice on one instant.
            //
            // For a question like "when does this appear", three moments spread across
            // the recording are worth more than three views of one, so require a minimum
            // separation in time as well. Frames rejected here are not lost - they are
            // still listed under "also matched".
            val tooClose = kept.any {
                kotlin.math.abs(parseTimestampSeconds(it.timestamp) -
                                parseTimestampSeconds(moment.timestamp)) < MIN_SECONDS_APART
            }
            if (!tooSimilar && !tooClose) kept.add(moment)
            if (kept.size >= OnDeviceVLM.MAX_FRAMES_TO_ANALYSE) break
        }
        val dropped = ranked.size.coerceAtMost(20) - kept.size
        if (dropped > 0) Log.i("VideoRAG_Query", "dedup: kept ${kept.size} distinct frames")
        return kept
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) dot += a[i] * b[i]
        return dot   // vectors are already L2-normalised by the embedder
    }

    /** "HH:MM:SS" or "MM:SS" -> seconds. */
    private fun parseTimestampSeconds(ts: String): Int {
        val p = ts.split(":").mapNotNull { it.toIntOrNull() }
        return when (p.size) {
            3 -> p[0] * 3600 + p[1] * 60 + p[2]
            2 -> p[0] * 60 + p[1]
            else -> 0
        }
    }

    // ── playback ──────────────────────────────────────────────────

    /** Seek the inline player to [seconds]; wired to timestamps inside chat messages. */
    private fun playVideoAt(seconds: Int) {
        val uri = currentVideoUri ?: run {
            Toast.makeText(this, "No video loaded", Toast.LENGTH_SHORT).show(); return
        }
        lastSelectedTimestampMs = seconds * 1000
        cardVideoPlayback.visibility = View.VISIBLE
        tvPlayerTimestamp.text = String.format(Locale.US, "Playing from %02d:%02d", seconds / 60, seconds % 60)
        videoViewPlayback.setVideoURI(uri)
        videoViewPlayback.seekTo(lastSelectedTimestampMs)
        videoViewPlayback.start()
        btnPlayPause.text = "⏸"
    }

    /** Diagnostics that used to clutter the main screen, now one tap away. */
    private fun showDebugPanel() {
        lifecycleScope.launch(Dispatchers.IO) {
            val vlmInfo = try {
                val v = orchestrator.getFrameDescriber()
                if (v.isNativeGGUFAvailable()) v.getDiagnosticInfo() else "no GGUF found"
            } catch (e: Throwable) { "error: ${e.message}" }

            val embInfo = try {
                orchestrator.getActiveEmbedder(); "MobileCLIP-S2 dual tower, 512-D, ONNX"
            } catch (e: Throwable) { "unavailable: ${e.message?.take(90)}" }

            val json = vectorStore.getAllMoments()
                .distinctBy { it.imagePath }
                .take(40)
                .joinToString(",\n") { it.toJsonObject().toString(1) }
                .ifEmpty { null }?.let { "[\n$it\n]" }

            val state = DebugPanel.State(
                modelInfo = vlmInfo,
                embedderInfo = embInfo,
                tokenizerInfo = tokenizerStatus,
                keyframesKept = acceptedFramesCount,
                duplicatesDropped = droppedFramesCount,
                vectorCount = vectorStore.size,
                ftsRows = sqliteFts.size(),
                sampleFps = sampleFps,
                gateEnabled = true,
                gateThreshold = hashThreshold,
                lastQuery = lastQuery,
                lastHits = lastHits,
                framesSentToModel = lastFramesSent,
                droppedTimestamps = lastDroppedTimestamps,
                genStats = lastGenStats,
                lastLatencyMs = lastLatencyMs,
                indexedJson = json
            )
            withContext(Dispatchers.Main) { DebugPanel.show(this@MainActivity, state) }
        }
    }

    private fun scrollToBottom() = scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }

    private fun hideKeyboard() {
        (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
            .hideSoftInputFromWindow(etQuery.windowToken, 0)
    }

    override fun onDestroy() {
        super.onDestroy()
        try { videoViewPlayback.stopPlayback() } catch (_: Exception) {}
        lifecycleScope.launch(Dispatchers.IO) { orchestrator.releaseAll() }
    }
}
