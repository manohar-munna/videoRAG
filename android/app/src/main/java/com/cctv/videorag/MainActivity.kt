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
    private var lastSelectedTimestampMs = 0
    private var isBusy = false

    /** Sampling rate. Fixed at 1 FPS now that the FPS pills are gone. */
    private val sampleFps = 1.0

    /** Prior turns, oldest first, fed back to the model so follow-ups have context. */
    private val conversation = mutableListOf<ConversationTurn>()

    // ── views ─────────────────────────────────────────────────────
    private lateinit var scrollView: NestedScrollView
    private lateinit var chatContainer: LinearLayout
    private lateinit var tvModelBadge: TextView
    private lateinit var tvIngestionInfo: TextView
    private lateinit var pbIngestion: ProgressBar
    private lateinit var btnPickVideo: Button
    private lateinit var btnClearAll: Button
    private lateinit var btnSelectModelFolder: Button
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
        orchestrator = MemoryOrchestrator(
            this,
            "${filesDir.absolutePath}/mobileclip_s2.onnx",
            "${filesDir.absolutePath}/qwen2_vl_2b/"
        )
        setupListeners()
        checkAndRequestStoragePermission()
        runTokenizerSelfTest()
        updateModelBadge()
        ChatView.addSystemNote(chatContainer, "Import a video, then ask questions about it.")
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
        btnClearAll.setOnClickListener { resetAllData() }

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

    /** Reports whether model FILES exist; deliberately does not load them on Main. */
    private fun updateModelBadge() {
        lifecycleScope.launch(Dispatchers.IO) {
            val ok = try { orchestrator.getFrameDescriber().isNativeGGUFAvailable() }
                     catch (_: Throwable) { false }
            withContext(Dispatchers.Main) {
                tvModelBadge.text = if (ok) "Model ready" else "No model"
            }
        }
    }

    /** The Kotlin CLIP BPE port must match Python's ids or retrieval silently degrades. */
    private fun runTokenizerSelfTest() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val tk = ClipTokenizer.fromAssets(this@MainActivity)
                val r = ClipTokenizerSelfTest.run(this@MainActivity, tk)
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

    private fun processSelectedVideoUri(uri: Uri) {
        currentVideoUri = uri
        val startedAt = System.currentTimeMillis()

        lifecycleScope.launch(Dispatchers.Default) {
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
            vectorStore.addMoment(
                IndexedMoment(
                    id = "${camera}_${timestamp}_${crop.label}",
                    camera = camera, timestamp = timestamp, epochTime = epochTime,
                    vector = v, cropRegion = crop.label, imagePath = imagePath,
                    description = frameDescription, jsonMetadata = frameJson.toString()
                )
            )
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

        lifecycleScope.launch(Dispatchers.Default) {
            val answer = try { answerQuestion(q) }
                         catch (e: Throwable) {
                             Log.e("VideoRAG_Query", "query failed", e)
                             "Something went wrong: ${e.message ?: e.javaClass.simpleName}"
                         }
            conversation.add(ConversationTurn(q, answer))
            withContext(Dispatchers.Main) {
                isBusy = false
                btnSearch.isEnabled = true
                ChatView.replacePending(chatContainer, answer) { seconds -> playVideoAt(seconds) }
                scrollToBottom()
            }
        }
    }

    /** Retrieve with CLIP (max-pooled across regions), then answer with the VLM. */
    private suspend fun answerQuestion(question: String): String {
        val embedder = orchestrator.getActiveEmbedder()

        // Each frame is indexed as 6 regions; collapse to the best region per frame so a
        // small object competes on its strongest crop rather than the whole scene.
        val bestPerFrame = HashMap<String, Pair<IndexedMoment, Float>>()
        val queryVector = embedder.embedText(question)
        for ((moment, score) in vectorStore.search(queryVector, topK = 60)) {
            val prev = bestPerFrame[moment.imagePath]
            if (prev == null || score > prev.second) bestPerFrame[moment.imagePath] = moment to score
        }
        val ranked = bestPerFrame.values.sortedByDescending { it.second }
        Log.i("VideoRAG_Query", "top=" + ranked.take(5).joinToString {
            "${it.first.timestamp}[${it.first.cropRegion}]=%.3f".format(it.second)
        })
        if (ranked.isEmpty()) return "I couldn't find anything matching that in this video."

        val vlm = orchestrator.getActiveVLM()
        return vlm.answerFromRetrievedContext(
            query = question,
            top5Moments = ranked.map { it.first },
            history = conversation.takeLast(3)
        )
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
