package com.cctv.videorag

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
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
import com.cctv.videorag.service.VideoRAGService
import com.cctv.videorag.ui.ChatView
import com.cctv.videorag.ui.DebugPanel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
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

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var currentQueryJob: Job? = null
    private var currentIngestJob: Job? = null
    private var currentDownloadJob: Job? = null

    private lateinit var orchestrator: MemoryOrchestrator
    private val vectorStore = MobileVectorStore()
    private val sqliteFts by lazy { SQLiteFtsHelper(this) }
    private val frameDecoder by lazy { VideoFrameDecoder(this) }

    // ── ingestion state ───────────────────────────────────────────
    private var lastFrameHash: Long? = null
    private var acceptedFramesCount = 0
    /** Time spent turning existing vectors into object labels, across one ingest. */
    private var labelNanosTotal = 0L
    private var droppedFramesCount = 0
    private val hashThreshold = MobileFrameFilter.DEFAULT_HAMMING_THRESHOLD
    private var currentVideoUri: Uri? = null
    /** Stable key for the imported video, so its saved index can be found again. */
    private var currentVideoKey: String = ""
    private var lastSelectedTimestampMs = 0
    private var isBusy = false
    private var downloading = false

    /**
     * Keyframe sampling rate.
     *
     * 0.2 FPS (one frame per 5 s) deliberately, not 1 FPS. This is the density the app's
     * best answers were produced at: before the decoder was fixed, OPTION_CLOSEST_SYNC
     * silently returned one distinct frame per ~5.2 s, giving a ~132-keyframe index whose
     * top-5 for a query landed on genuinely different moments -
     *
     *   "White truck with a man in a grey t-shirt ... at 00:03:42.
     *    White truck with the words 'motion picture' on the side at 00:06:38.
     *    White truck with a man in a white t-shirt ... at 00:07:20, 00:08:40 and 00:10:54."
     *
     * True 1 FPS made the index 3.5x denser (466 keyframes). Retrieval then filled every
     * slot with near-identical views of whichever moment scored highest, the model wrote
     * the same sentence for each, and groupBySubject collapsed them to one flat line.
     * Widening the candidate pool, the time-spread rule and the duplicate threshold were
     * all tried and none restored the variety - the density is the cause, not the
     * selection.
     *
     * Sampling here rather than reverting the decoder keeps the sequential MediaCodec
     * pass (1b9abec), so this is the old index density reached the fast way: ~5x fewer
     * frames to embed, which is the ingest bottleneck.
     *
     * The trade is explicit: an event visible for less than ~5 s can fall between samples.
     * That was true of every build whose answers were good. It is now the FLOOR rather
     * than the rate - see sampleFpsFor() for why a fixed rate cannot serve both a
     * 13-minute clip and a 48-second one.
     */
    private val MIN_SAMPLE_FPS = 0.2

    /**
     * Frames to aim for across the whole video, whatever its length.
     *
     * 156 samples over the 13-minute clip produced 146 keyframes, the density every good
     * answer in this project was measured at, so that is the number worth reproducing.
     */
    private val TARGET_KEYFRAMES = 60.0

    /** Ceiling, so a very short clip does not turn into thousands of embeddings. */
    private val MAX_SAMPLE_FPS = 2.0

    /** Rate actually used for the last ingest. Reported in the debug panel. */
    private var sampleFps = MIN_SAMPLE_FPS

    /** Length of the indexed video. 0 when unknown; drives minSecondsApart(). */
    private var videoDurationSec = 0

    /**
     * How densely to sample this particular video.
     *
     * A fixed 0.2 fps was tuned on the 13-minute clip. Applied to a short video it
     * starves retrieval: a 48-second traffic clip yields ~10 samples and 7 keyframes,
     * so the store holds 42 vectors and there is almost nothing for the ranking to choose
     * between. Every query then returns the same two or three frames and the answer thins
     * to a single line - which reads like a model problem and is not one.
     *
     * So aim for a frame budget instead of a rate. The lower clamp is the old constant,
     * so anything long enough to have been sampled well before is sampled identically and
     * its results are unchanged; only videos short enough to be starved sample faster.
     */
    /** Length of [uri] in whole seconds, or 0 if it cannot be read. */
    private fun durationSecondsOf(uri: Uri): Int {
        val r = android.media.MediaMetadataRetriever()
        return try {
            r.setDataSource(this, uri)
            ((r.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L) / 1000L).toInt()
        } catch (e: Throwable) {
            Log.w("VideoRAG_Main", "could not read duration (${e.message})")
            0
        } finally {
            try { r.release() } catch (_: Throwable) {}
        }
    }

    private fun sampleFpsFor(uri: Uri): Double {
        val seconds = durationSecondsOf(uri).toDouble()
        videoDurationSec = seconds.toInt()
        val fps = if (seconds <= 0.0) MIN_SAMPLE_FPS
                  else (TARGET_KEYFRAMES / seconds).coerceIn(MIN_SAMPLE_FPS, MAX_SAMPLE_FPS)
        sampleFps = fps
        Log.i("VideoRAG_Main", "video ${seconds.toInt()}s -> sampling at $fps fps " +
              "(~${(seconds * fps).toInt()} frames)")
        return fps
    }

    /**
     * Cosine above which two retrieved frames count as the same shot and the later one
     * is not worth spending ~17 s to encode. See dropNearDuplicates().
     *
     * Tightened from 0.92 once the 1 fps index (1b9abec) made near-duplicates abundant:
     * 466 keyframes instead of 126 means the top of the ranking fills with many views of
     * one moment, all scoring just under 0.92, so five slots were spent on the same shot.
     * The model then wrote the same sentence for every frame and groupBySubject collapsed
     * them into a single flat line - "White truck with people on it at 00:02:42, 00:03:40,
     * 00:05:27 and 00:07:23" - where the app used to give a line per distinct moment.
     *
     * 0.82 forces the slots onto visually different frames, which is what gives the model
     * something different to say about each one.
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
    private val MIN_RELEVANCE = 0.19f

    /**
     * Cap on the extra "also matched" timestamps listed under an answer. Enough to restore
     * the coverage lost by analysing 3 frames instead of 5, without turning the footer into
     * a wall of weak hits.
     */
    private val MAX_ALSO_MATCHED = 6

    /**
     * Minimum gap between two frames sent to the model, in seconds. See
     * dropNearDuplicates().
     *
     * Raised from 5 s after the 1 fps decoder fix (1b9abec) made the index 3.7x denser:
     * with 466 keyframes instead of 126, the top-scoring frames all come from the same
     * moment, so the model was handed five near-identical views and described them
     * identically - groupBySubject then collapsed them to a single flat line:
     *
     *   5 s apart   "White truck with black ladder on back at 00:02:42, 00:03:40,
     *                00:04:08, 00:05:27 and 00:07:23."   (all five one scene)
     *
     * The richer answers this app used to give came from frames spread across the whole
     * recording (00:03:42 / 00:06:38 / 00:07:20 / 00:08:40 / 00:10:54), where each frame
     * showed something different and earned its own sentence. 45 s forces that spread.
     *
     * The top-ranked frame is always kept first, so widening the gap never costs the best
     * match - it only stops the remaining slots being spent on its neighbours.
     */
    private val MAX_SECONDS_APART = 15

    /** Never demand a gap so wide that five frames cannot fit in the video. */
    private val FLOOR_SECONDS_APART = 4

    /**
     * Minimum gap between two frames sent to the model, for THIS video.
     *
     * 15 s was measured on the 13-minute clip and is right there. It is wrong for a short
     * one: fitting MAX_FRAMES_TO_ANALYSE (5) frames at 15 s apart needs 60 s of video, so
     * a 47-second clip could never supply more than three - and in practice supplied two,
     * no matter how densely it was indexed. Both queries on the traffic clip came back
     * "Analysed 2 keyframes" off a 44-keyframe index for exactly this reason; the density
     * was there and the spacing rule would not let retrieval spend it.
     *
     * duration/6 leaves five frames spanning two-thirds of the recording, which is the
     * spread that makes each one worth a separate sentence. Anything 90 s or longer still
     * gets the measured 15 s, so the 13-minute results are untouched.
     */
    private fun minSecondsApart(): Int =
        if (videoDurationSec <= 0) MAX_SECONDS_APART
        else (videoDurationSec / 6).coerceIn(FLOOR_SECONDS_APART, MAX_SECONDS_APART)

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
    private lateinit var btnStop: Button
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
        btnStop = findViewById(R.id.btnStop)
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
        // Unconditional: downloadModels() decides for itself whether anything is
        // missing. Gating on the badge's label was fragile - TextView.getText() is a
        // CharSequence that does not reliably compare equal to a String literal, so the
        // tap silently did nothing while the badge plainly read "Download models".
        tvModelBadge.setOnClickListener { downloadModels() }
        btnClearAll.setOnClickListener { resetAllData() }
        btnDebug.setOnClickListener { showDebugPanel() }

        btnSearch.setOnClickListener { submitQuestion() }
        btnStop.setOnClickListener { stopCurrentOperation() }
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
        videoViewPlayback.setOnPreparedListener { mp ->
            mp.isLooping = false
            fitPlayerToVideo(mp.videoWidth, mp.videoHeight)
        }
        videoViewPlayback.setOnErrorListener { _, w, e ->
            Log.w("VideoView", "playback error what=$w extra=$e"); true
        }
    }

    /**
     * Open the all-files-access settings page. Deliberately NOT called on launch.
     *
     * It used to run from onCreate, which meant a fresh install dropped the user straight
     * into a system Settings screen before they had seen the app - and asked for the most
     * invasive storage permission Android has, to boot.
     *
     * Nothing in the shipping flow needs it. Weights download to getExternalFilesDir, which
     * ModelPaths searches first and which requires no permission; the video arrives through
     * the system picker, which grants access to just that one item; frames and the local
     * video copy live in filesDir. The permission only ever mattered for weights sideloaded
     * into /sdcard/Download, which ModelPaths still keeps as a fallback. Left here so that
     * path stays reachable if it is ever wanted, rather than deleted outright.
     */
    @Suppress("unused")
    private fun requestAllFilesAccess() {
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
        // Nothing to do when the weights are already present - tapping a "Model ready"
        // badge should be a no-op, not a spurious download.
        appScope.launch(Dispatchers.IO) {
            val nothingMissing = try { ModelDownloader.missing(this@MainActivity).isEmpty() }
                                 catch (_: Throwable) { false }
            if (nothingMissing) return@launch
            withContext(Dispatchers.Main) { startModelDownload() }
        }
    }

    private fun stopCurrentOperation() {
        if (!isBusy && !downloading) return
        Log.i("VideoRAG_Main", "User requested stop of current operation")

        // 1. Abort VLM inference & cancel query job
        try {
            orchestrator.abortVLM()
        } catch (e: Throwable) {
            Log.w("VideoRAG_Main", "Error aborting VLM: ${e.message}")
        }
        currentQueryJob?.cancel()
        currentQueryJob = null

        // 2. Cancel video decoding & indexing job
        try {
            frameDecoder.cancel()
        } catch (e: Throwable) {
            Log.w("VideoRAG_Main", "Error cancelling frameDecoder: ${e.message}")
        }
        currentIngestJob?.cancel()
        currentIngestJob = null

        // 3. Cancel model download job
        try {
            ModelDownloader.cancel()
        } catch (e: Throwable) {
            Log.w("VideoRAG_Main", "Error cancelling ModelDownloader: ${e.message}")
        }
        currentDownloadJob?.cancel()
        currentDownloadJob = null

        // 4. Stop Foreground Service
        VideoRAGService.stop(this)

        // 5. Update UI
        isBusy = false
        if (downloading) {
            downloading = false
            pbIngestion.visibility = View.GONE
            tvIngestionInfo.text = "Model download stopped."
            ChatView.addSystemNote(chatContainer, "Model download stopped by user.")
            updateModelBadge()
        } else {
            pbIngestion.visibility = View.GONE
            ChatView.replacePending(chatContainer, "[Operation stopped by user]")
        }

        btnStop.visibility = View.GONE
        btnSearch.visibility = View.VISIBLE
        btnSearch.isEnabled = true
        Toast.makeText(this, "Stopped", Toast.LENGTH_SHORT).show()
    }

    private fun startModelDownload() {
        if (downloading) return
        downloading = true
        tvModelBadge.text = "Downloading…"
        pbIngestion.visibility = View.VISIBLE
        pbIngestion.isIndeterminate = true
        tvIngestionInfo.text = "Preparing model download…"
        btnSearch.visibility = View.GONE
        btnStop.visibility = View.VISIBLE
        btnStop.isEnabled = true
        VideoRAGService.start(this, "VideoRAG", "Preparing model download…")
        ChatView.addSystemNote(chatContainer, "Downloading on-device models in background. Tap Stop at any time to cancel.")

        currentDownloadJob = appScope.launch(Dispatchers.IO) {
            try {
                ModelDownloader.downloadMissing(this@MainActivity) { p ->
                    val doneMb = p.fileDone / 1_000_000
                    val totMb = if (p.fileBytes > 0) p.fileBytes / 1_000_000 else 0
                    val pct = if (p.fileBytes > 0) ((p.fileDone * 100) / p.fileBytes).toInt() else 0
                    val msg = "Downloading ${p.name} (${p.index + 1}/${p.total}) — $doneMb/$totMb MB"
                    VideoRAGService.update(this@MainActivity, "VideoRAG Model Download", msg, pct, 100)
                    appScope.launch(Dispatchers.Main) {
                        pbIngestion.isIndeterminate = false
                        pbIngestion.progress = pct
                        tvIngestionInfo.text = msg
                    }
                }
                withContext(Dispatchers.Main) {
                    downloading = false
                    btnStop.visibility = View.GONE
                    btnSearch.visibility = View.VISIBLE
                    btnSearch.isEnabled = true
                    VideoRAGService.stop(this@MainActivity)
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "Models downloaded. Import a video to start."
                    ChatView.addSystemNote(chatContainer, "Models ready.")
                    modelHintShown = false
                    updateModelBadge()
                }
            } catch (e: Throwable) {
                withContext(Dispatchers.Main) {
                    downloading = false
                    btnStop.visibility = View.GONE
                    btnSearch.visibility = View.VISIBLE
                    btnSearch.isEnabled = true
                    VideoRAGService.stop(this@MainActivity)
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = if (ModelDownloader.isCancelled) "Model download stopped." else "Model download failed."
                    ChatView.addSystemNote(chatContainer,
                        if (ModelDownloader.isCancelled) "Model download stopped by user."
                        else "Model download failed: ${e.message}. Tap the badge to resume.",
                        isError = !ModelDownloader.isCancelled
                    )
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
            // Prefer our own copy: the original content:// URI is dead by now.
            localVideoFile().takeIf { it.isFile && it.length() > 0 }?.let {
                currentVideoUri = Uri.fromFile(it)
                // Recover the duration as well: minSecondsApart() falls back to the
                // 15 s default without it, which would undo the spacing fix on every
                // restart even though the index itself restored fine.
                videoDurationSec = durationSecondsOf(Uri.fromFile(it))
            }
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

        currentIngestJob = appScope.launch(Dispatchers.Default) {
            // Keep our own copy so timestamps stay tappable after a restart.
            //
            // The picked content:// URI is only readable for the life of this process -
            // ACTION_GET_CONTENT grants no persistable permission, and the Photos picker
            // hands back a provider URI that cannot be re-opened later. So once the index
            // was restored at launch, every timestamp answered "No video loaded" even
            // though the frames and vectors were all there.
            //
            // Copying into filesDir means playback works from a cold start, and the copy
            // is removed when the app is uninstalled, like the models and the index.
            cacheVideoLocally(uri)
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
            labelNanosTotal = 0L
            conversation.clear()

            withContext(Dispatchers.Main) {
                isBusy = true
                btnSearch.visibility = View.GONE
                btnStop.visibility = View.VISIBLE
                btnStop.isEnabled = true
                VideoRAGService.start(this@MainActivity, "VideoRAG", "Indexing video…")
                ChatView.clear(chatContainer)
                ChatView.addSystemNote(chatContainer, "Indexing video in background… Tap Stop to cancel.")
                pbIngestion.visibility = View.VISIBLE
                pbIngestion.isIndeterminate = true
                cardVideoPlayback.visibility = View.GONE
            }

            try {
                frameDecoder.decodeVideoUri(
                    videoUri = uri,
                    cameraName = "cam_user",
                    sampleFps = sampleFpsFor(uri),
                    onProgress = { cur, total, _ ->
                        val pct = if (total > 0) ((cur * 100) / total).toInt() else 0
                        val msg = String.format(
                            Locale.US, "Indexing %02d:%02d / %02d:%02d — %d keyframes kept",
                            cur / 60, cur % 60, total / 60, total % 60, acceptedFramesCount
                        )
                        VideoRAGService.update(this@MainActivity, "VideoRAG Indexing", msg, pct, 100)
                        appScope.launch(Dispatchers.Main) {
                            pbIngestion.isIndeterminate = false
                            pbIngestion.progress = pct
                            tvIngestionInfo.text = msg
                        }
                    },
                    onKeyframeDecoded = { bmp, ts, epoch, path ->
                        if (!frameDecoder.isCancelled) {
                            ingestAndIndexFrame(bmp, "cam_user", ts, epoch, path)
                        }
                    }
                )
                val secs = (System.currentTimeMillis() - startedAt) / 1000
                val labelMs = labelNanosTotal / 1_000_000
                Log.i("VideoRAG_Main", "ingest $secs s total, of which object labelling " +
                      "$labelMs ms (${if (secs > 0) labelMs * 100.0 / (secs * 1000) else 0.0}%)")
                withContext(Dispatchers.Main) {
                    isBusy = false
                    btnStop.visibility = View.GONE
                    btnSearch.visibility = View.VISIBLE
                    btnSearch.isEnabled = true
                    VideoRAGService.stop(this@MainActivity)
                    pbIngestion.visibility = View.GONE
                    if (frameDecoder.isCancelled) {
                        tvIngestionInfo.text = "Indexing stopped ($acceptedFramesCount keyframes kept)."
                        ChatView.addSystemNote(chatContainer, "Indexing stopped by user.")
                    } else {
                        tvIngestionInfo.text =
                            "$acceptedFramesCount keyframes · ${vectorStore.size} vectors · ${droppedFramesCount} duplicates dropped"
                        ChatView.addSystemNote(
                            chatContainer,
                            "Indexed $acceptedFramesCount keyframes in ${secs}s. Ask a question below."
                        )
                    }
                }
            } catch (e: Throwable) {
                Log.e("VideoRAG_Ingest", "Ingestion failed", e)
                val isModel = e is OnDeviceEmbedder.ModelUnavailableException ||
                              e.cause is OnDeviceEmbedder.ModelUnavailableException
                withContext(Dispatchers.Main) {
                    isBusy = false
                    btnStop.visibility = View.GONE
                    btnSearch.visibility = View.VISIBLE
                    btnSearch.isEnabled = true
                    VideoRAGService.stop(this@MainActivity)
                    pbIngestion.visibility = View.GONE
                    if (frameDecoder.isCancelled) {
                        tvIngestionInfo.text = "Indexing stopped."
                        ChatView.addSystemNote(chatContainer, "Indexing stopped by user.")
                    } else {
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

        val embedder = orchestrator.getActiveEmbedder()
        // One batched run for the frame's six crops instead of six single runs. Ingest is
        // CLIP-bound - 466 keyframes x 6 crops is 2796 forward passes - so per-call
        // overhead is paid thousands of times per video.
        val crops = SpatialCropper.generatePyramidCrops(bitmap)
        val vectors = embedder.embedImages(crops.map { it.bitmap })

        // Name what the crops contain, reusing the vectors just computed. Timed
        // separately because "does this slow indexing down" deserves a number.
        val labelT0 = System.nanoTime()
        val objects = embedder.labelsFor(vectors)
        labelNanosTotal += System.nanoTime() - labelT0

        // cheap per-frame stats only; the VLM is not loaded during ingestion
        val vlm = orchestrator.getFrameDescriber()
        val frameJson = vlm.describeFrameAsJson(
            bitmap, timestamp, acceptedFramesCount, imagePath, objects)
        val frameDescription = frameJson.getString("visual_description")
        val momentsToSave = mutableListOf<IndexedMoment>()
        for ((crop, v) in crops.zip(vectors)) {
            val moment = IndexedMoment(
                id = "${camera}_${timestamp}_${crop.label}",
                camera = camera, timestamp = timestamp, epochTime = epochTime,
                vector = v, cropRegion = crop.label, imagePath = imagePath,
                description = frameDescription, jsonMetadata = frameJson.toString()
            )
            vectorStore.addMoment(moment)
            momentsToSave.add(moment)
        }
        sqliteFts.saveMoments(momentsToSave, currentVideoKey)   // survives restart in single tx
        for (crop in crops) if (crop.bitmap != bitmap) crop.bitmap.recycle()

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
        val questionBubble = ChatView.addUserMessage(chatContainer, q)
        ChatView.addPending(chatContainer)
        scrollToBottom()
        isBusy = true
        btnSearch.visibility = View.GONE
        btnStop.visibility = View.VISIBLE
        btnStop.isEnabled = true

        VideoRAGService.start(this, "VideoRAG", "Answering: \"$q\"")

        lastQuery = q
        val startedAt = System.currentTimeMillis()
        currentQueryJob = appScope.launch(Dispatchers.Default) {
            val result = try { answerQuestion(q) }
                         catch (e: Throwable) {
                             if (orchestrator.isVLMAborted()) {
                                 Answer("Query stopped by user.", emptyList())
                             } else {
                                 Log.e("VideoRAG_Query", "query failed", e)
                                 Answer("Something went wrong: ${e.message ?: e.javaClass.simpleName}", emptyList())
                             }
                         }
            lastLatencyMs = System.currentTimeMillis() - startedAt
            if (!orchestrator.isVLMAborted()) {
                conversation.add(ConversationTurn(q, result.text))
            }
            withContext(Dispatchers.Main) {
                isBusy = false
                btnStop.visibility = View.GONE
                btnSearch.visibility = View.VISIBLE
                btnSearch.isEnabled = true
                VideoRAGService.stop(this@MainActivity)
                ChatView.replacePending(
                    chatContainer, result.text,
                    onTimestamp = { seconds -> playVideoAt(seconds) },
                    frames = result.frames
                )
                lastLatencyMs?.let { ChatView.setUserMessageTiming(questionBubble, q, it) }
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
        // Scale the candidate pool with the index, do not fix it.
        //
        // topK=60 was tuned against a 756-vector index (8% of it). The 1 fps decoder fix
        // (1b9abec) grew the index to ~2796 vectors, where 60 is 2% - and those 60 all
        // come from whichever moment scores highest, so every candidate was a view of one
        // shot. dropNearDuplicates() and the time-spread rule then had nothing to choose
        // between, and the model wrote the same sentence for all of them:
        //   "White truck with people on it at 00:02:42, 00:03:40, 00:05:27, 00:07:23"
        // where this app used to give a line per distinct moment across the recording.
        //
        // A tenth of the index keeps roughly the old proportion. The scan is a dot product
        // over unit vectors already in memory, so a wider pool costs microseconds.
        val poolSize = (vectorStore.size / 10).coerceIn(60, 400)
        for ((moment, score) in vectorStore.searchMulti(queryVectors, topK = poolSize)) {
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
            if (spread.any { kotlin.math.abs(parseTimestampSeconds(it) - secs) < minSecondsApart() }) continue
            if (analysed.any { kotlin.math.abs(parseTimestampSeconds(it) - secs) < minSecondsApart() }) continue
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
                                parseTimestampSeconds(moment.timestamp)) < minSecondsApart()
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

    /** Where the imported video is kept so playback survives a restart. */
    private fun localVideoFile(): File =
        File(File(filesDir, "videos").apply { mkdirs() }, "current.mp4")

    /** Copy the picked video into app storage; best effort, playback-only. */
    private fun cacheVideoLocally(uri: Uri) {
        try {
            val out = localVideoFile()
            contentResolver.openInputStream(uri)?.use { input ->
                java.io.FileOutputStream(out).use { fos -> input.copyTo(fos, 1 shl 16) }
            }
            Log.i("VideoRAG_Main", "cached video locally: ${out.length() / 1_000_000} MB")
        } catch (e: Throwable) {
            // Not fatal: indexing and search do not need it, only tapping a timestamp does.
            Log.w("VideoRAG_Main", "could not cache video for playback: ${e.message}")
        }
    }

    /**
     * Give the player the same shape as the video, so no black band is left over.
     *
     * VideoView keeps the source aspect ratio inside whatever box it is measured in. The
     * box was a fixed 200dp tall and match_parent wide, so a 16:9 clip shrank its own width
     * to fit the height and, being in a vertical LinearLayout, sat flush left - leaving a
     * black pillar down the right-hand side of the card.
     *
     * Deriving the height from the real video dimensions makes the video fill the card's
     * width exactly. Portrait clips are the exception: height is capped so they cannot take
     * over the screen, and in that case the leftover is centred rather than all on one side.
     */
    private fun fitPlayerToVideo(videoWidth: Int, videoHeight: Int) {
        if (videoWidth <= 0 || videoHeight <= 0) return
        val row = videoViewPlayback.parent as? View ?: return
        val available = row.width - row.paddingLeft - row.paddingRight
        if (available <= 0) return
        val maxHeight = (280 * resources.displayMetrics.density).toInt()
        val wanted = (available.toLong() * videoHeight / videoWidth).toInt()
        val lp = videoViewPlayback.layoutParams as LinearLayout.LayoutParams
        lp.height = wanted.coerceAtMost(maxHeight)
        lp.gravity = Gravity.CENTER_HORIZONTAL
        videoViewPlayback.layoutParams = lp
    }

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
        if (isFinishing) {
            VideoRAGService.stop(this)
            CoroutineScope(Dispatchers.IO).launch {
                try { orchestrator.releaseAll() } catch (e: Throwable) {
                    Log.w("VideoRAG_Main", "releaseAll error: ${e.message}")
                }
            }
        }
    }
}
