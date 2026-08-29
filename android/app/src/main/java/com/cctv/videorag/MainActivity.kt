package com.cctv.videorag

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import androidx.core.widget.NestedScrollView
import androidx.lifecycle.lifecycleScope
import com.cctv.videorag.indexing.*
import com.cctv.videorag.ingestion.*
import com.cctv.videorag.llm.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var orchestrator: MemoryOrchestrator
    private val vectorStore = MobileVectorStore()
    private val sqliteFts by lazy { SQLiteFtsHelper(this) }
    private val frameDecoder by lazy { VideoFrameDecoder(this) }

    private var lastFrameHash: Long? = null
    private var acceptedFramesCount = 0
    private var droppedFramesCount = 0
    private var selectedFps = 1.0 // Default: 1 frame per second
    // Gate ON by default. With it off, every sampled frame is embedded, captioned and
    // stored - on a long clip that is mostly duplicate footage of an empty scene.
    private var enableHashGate = true
    private var hashThreshold = MobileFrameFilter.DEFAULT_HAMMING_THRESHOLD
    private var currentVideoUri: Uri? = null
    private var lastSelectedTimestampMs: Int = 0
    private var ingestionStartTimeMs: Long = 0L

    // UI Elements
    private lateinit var scrollView: NestedScrollView
    private lateinit var tvStatus: TextView
    private lateinit var btnClearAll: Button
    private lateinit var btnSelectModelFolder: Button
    private lateinit var btnPickVideo: Button
    private lateinit var btnToggleUrl: Button
    private lateinit var btnViewAllJson: Button
    private lateinit var btnFps05: Button
    private lateinit var btnFps10: Button
    private lateinit var btnFps20: Button
    private lateinit var btnToggleHashGate: Button
    private lateinit var layoutUrlInput: LinearLayout
    private lateinit var etVideoUrl: EditText
    private lateinit var btnDownloadUrl: Button
    private lateinit var tvIngestionInfo: TextView
    private lateinit var pbIngestion: ProgressBar

    private lateinit var tvHashValue: TextView
    private lateinit var tvGateMetrics: TextView
    private lateinit var tvPyramidMetrics: TextView

    // 3-Column Metrics Grid
    private lateinit var tvMetricFrames: TextView
    private lateinit var tvMetricDropped: TextView
    private lateinit var tvMetricRegions: TextView
    private lateinit var tvMetricTime: TextView
    private lateinit var tvMetricTotalFrames: TextView

    private lateinit var etQuery: EditText
    private lateinit var btnSearch: Button
    private lateinit var tvExpandedQuery: TextView

    private lateinit var scrollStoryboard: View
    private lateinit var layoutStoryboardThumbnails: LinearLayout
    private lateinit var tvResults: TextView

    // Video Playback Card Views
    private lateinit var cardVideoPlayback: CardView
    private lateinit var tvPlayerTimestamp: TextView
    private lateinit var videoViewPlayback: VideoView
    private lateinit var btnClosePlayer: Button
    private lateinit var btnPlayPause: Button
    private lateinit var btnReplayTimestamp: Button

    // Activity Result Launcher for selecting local video
    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            currentVideoUri = it
            processSelectedVideoUri(it)
        }
    }

    // Activity Result Launcher for selecting GGUF model folder directly
    private val selectFolderLauncher = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
        uri?.let { treeUri ->
            try {
                contentResolver.takePersistableUriPermission(
                    treeUri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: Exception) {}

            val docId = android.provider.DocumentsContract.getTreeDocumentId(treeUri)
            val split = docId.split(":")
            val realPath = if (split.size > 1) {
                if ("primary".equals(split[0], ignoreCase = true)) {
                    "/storage/emulated/0/${split[1]}"
                } else {
                    "/storage/${split[0]}/${split[1]}"
                }
            } else {
                docId
            }

            Log.i("MainActivity", "User selected model folder: $realPath ($treeUri)")

            // customModelDirectory's setter calls loadVLM(), so this must not run on Main.
            lifecycleScope.launch(Dispatchers.IO) {
                val vlm = orchestrator.getFrameDescriber()
                vlm.customModelDirectory = realPath
                val ok = vlm.isNativeGGUFAvailable()
                withContext(Dispatchers.Main) {
                    updateModelBadge()
                    Toast.makeText(
                        this@MainActivity,
                        if (ok) "Model files found in $realPath" else "No .gguf files found in $realPath",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initOrchestrator()
        setupListeners()
        checkAndRequestStoragePermission()
        runTokenizerSelfTest()
    }

    override fun onResume() {
        super.onResume()
        updateModelBadge()
    }

    private fun checkAndRequestStoragePermission() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            if (!android.os.Environment.isExternalStorageManager()) {
                try {
                    val intent = android.content.Intent(android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (_: Exception) {
                    try {
                        val intent = android.content.Intent(android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                        startActivity(intent)
                    } catch (_: Exception) {}
                }
            }
        }
    }

    /**
     * Reports whether the model FILES are present. Deliberately does not load them:
     * this runs from onCreate and onResume, and it previously called getActiveVLM() on
     * Dispatchers.Main, mapping a ~1.7 GB model load onto the UI thread on every resume.
     * The badge also no longer claims the model is "Active" merely because files exist.
     */
    private fun updateModelBadge() {
        val tvModelBadge = findViewById<TextView>(R.id.tvModelBadge) ?: return
        lifecycleScope.launch(Dispatchers.IO) {
            val available = orchestrator.getFrameDescriber().isNativeGGUFAvailable()
            withContext(Dispatchers.Main) {
                tvModelBadge.text = if (available) "Model files found" else "No model files"
            }
        }
    }

    private fun initViews() {
        scrollView = findViewById(R.id.mainScrollView)
        tvStatus = findViewById(R.id.tvStatus)
        btnClearAll = findViewById(R.id.btnClearAll)
        btnSelectModelFolder = findViewById(R.id.btnSelectModelFolder)
        btnPickVideo = findViewById(R.id.btnPickVideo)
        btnToggleUrl = findViewById(R.id.btnToggleUrl)
        btnViewAllJson = findViewById(R.id.btnViewAllJson)
        btnFps05 = findViewById(R.id.btnFps05)
        btnFps10 = findViewById(R.id.btnFps10)
        btnFps20 = findViewById(R.id.btnFps20)
        btnToggleHashGate = findViewById(R.id.btnToggleHashGate)
        layoutUrlInput = findViewById(R.id.layoutUrlInput)
        etVideoUrl = findViewById(R.id.etVideoUrl)
        btnDownloadUrl = findViewById(R.id.btnDownloadUrl)
        tvIngestionInfo = findViewById(R.id.tvIngestionInfo)
        pbIngestion = findViewById(R.id.pbIngestion)

        tvHashValue = findViewById(R.id.tvHashValue)
        tvGateMetrics = findViewById(R.id.tvGateMetrics)
        tvPyramidMetrics = findViewById(R.id.tvPyramidMetrics)

        // Metrics Grid
        tvMetricFrames = findViewById(R.id.tvMetricFrames)
        tvMetricDropped = findViewById(R.id.tvMetricDropped)
        tvMetricRegions = findViewById(R.id.tvMetricRegions)
        tvMetricTime = findViewById(R.id.tvMetricTime)
        tvMetricTotalFrames = findViewById(R.id.tvMetricTotalFrames)

        etQuery = findViewById(R.id.etQuery)
        btnSearch = findViewById(R.id.btnSearch)
        tvExpandedQuery = findViewById(R.id.tvExpandedQuery)

        scrollStoryboard = findViewById(R.id.scrollStoryboard)
        layoutStoryboardThumbnails = findViewById(R.id.layoutStoryboardThumbnails)
        tvResults = findViewById(R.id.tvResults)

        // Video Player UI
        cardVideoPlayback = findViewById(R.id.cardVideoPlayback)
        tvPlayerTimestamp = findViewById(R.id.tvPlayerTimestamp)
        videoViewPlayback = findViewById(R.id.videoViewPlayback)
        btnClosePlayer = findViewById(R.id.btnClosePlayer)
        btnPlayPause = findViewById(R.id.btnPlayPause)
        btnReplayTimestamp = findViewById(R.id.btnReplayTimestamp)
    }

    private fun initOrchestrator() {
        val onnxPath = "${filesDir.absolutePath}/mobileclip_s2.onnx"
        val vlmPath = "${filesDir.absolutePath}/qwen2_vl_2b/"
        orchestrator = MemoryOrchestrator(this, onnxPath, vlmPath)
        updateModelBadge()
    }

    private fun setupListeners() {
        // Select Model Folder Directly via System File Picker
        btnSelectModelFolder.setOnClickListener {
            selectFolderLauncher.launch(null)
        }

        // Reset / Clear All
        btnClearAll.setOnClickListener {
            resetAllData()
            Toast.makeText(this, "All vector and SQLite FTS5 indices cleared.", Toast.LENGTH_SHORT).show()
        }

        // View All Indexed JSON Data in Modal Inspector
        btnViewAllJson.setOnClickListener {
            showAllJsonDialog()
        }

        // Pick Local Video File
        btnPickVideo.setOnClickListener {
            pickVideoLauncher.launch("video/*")
        }

        // Toggle URL Input Container
        btnToggleUrl.setOnClickListener {
            layoutUrlInput.visibility = if (layoutUrlInput.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        // Sampling FPS Selectors
        btnFps05.setOnClickListener { selectFps(0.5, btnFps05) }
        btnFps10.setOnClickListener { selectFps(1.0, btnFps10) }
        btnFps20.setOnClickListener { selectFps(2.0, btnFps20) }

        // dHash Gate Toggle (No Hashing vs Filtering)
        // Reflect the real default (gate ON) rather than whatever the layout hardcodes.
        applyHashGateButtonState()

        btnToggleHashGate.setOnClickListener {
            enableHashGate = !enableHashGate
            if (enableHashGate) {
                btnToggleHashGate.text = "Gate: $hashThreshold"
                btnToggleHashGate.setTextColor(ContextCompat.getColor(this, R.color.primary))
                Toast.makeText(this, "dHash gate ON - drops frames within $hashThreshold bits of the last keyframe", Toast.LENGTH_SHORT).show()
            } else {
                btnToggleHashGate.text = "Gate: OFF"
                btnToggleHashGate.setTextColor(ContextCompat.getColor(this, R.color.text_muted))
                Toast.makeText(this, "dHash gate OFF - every sampled frame is indexed", Toast.LENGTH_SHORT).show()
            }
        }

        // Download & Ingest Remote Video URL
        btnDownloadUrl.setOnClickListener {
            val url = etVideoUrl.text.toString().trim()
            if (url.isNotEmpty()) {
                downloadAndIngestVideoUrl(url)
            } else {
                Toast.makeText(this, "Please enter a valid video URL", Toast.LENGTH_SHORT).show()
            }
        }

        // Execute Hybrid Forensic RAG Query (Dense Vector + SQLite FTS5 RRF)
        btnSearch.setOnClickListener {
            val query = etQuery.text.toString().trim()
            if (query.isNotEmpty()) {
                performLocalVideoRAGQuery(query)
            } else {
                Toast.makeText(this, "Enter a search query to inspect footage", Toast.LENGTH_SHORT).show()
            }
        }

        // Video Player Controls
        btnClosePlayer.setOnClickListener {
            videoViewPlayback.stopPlayback()
            cardVideoPlayback.visibility = View.GONE
        }

        btnPlayPause.setOnClickListener {
            if (videoViewPlayback.isPlaying) {
                videoViewPlayback.pause()
                btnPlayPause.text = "▶ Play"
            } else {
                videoViewPlayback.start()
                btnPlayPause.text = "⏸ Pause"
            }
        }

        btnReplayTimestamp.setOnClickListener {
            videoViewPlayback.seekTo(lastSelectedTimestampMs)
            videoViewPlayback.start()
            btnPlayPause.text = "⏸ Pause"
        }

        videoViewPlayback.setOnPreparedListener { mp ->
            mp.isLooping = true
        }

        videoViewPlayback.setOnErrorListener { _, what, extra ->
            Log.w("VideoView", "Error playing video: what=$what, extra=$extra")
            Toast.makeText(this, "Seeking to timestamp $lastSelectedTimestampMs ms", Toast.LENGTH_SHORT).show()
            true
        }
    }

    /**
     * Confirm the Kotlin CLIP BPE port still matches Python's token ids.
     * A drift here is invisible at runtime - embeddings just land in the wrong place -
     * so it is checked explicitly rather than assumed.
     */
    private fun runTokenizerSelfTest() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val tk = ClipTokenizer.fromAssets(this@MainActivity)
                val r = ClipTokenizerSelfTest.run(this@MainActivity, tk)
                if (!r.ok) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@MainActivity,
                            "Tokenizer self-test FAILED (${r.failed}) - text search will be unreliable",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            } catch (e: Throwable) {
                Log.e("VideoRAG_Tokenizer", "self-test error", e)
            }
        }
    }

    /** Keep the gate button's label and colour in sync with [enableHashGate]. */
    private fun applyHashGateButtonState() {
        if (enableHashGate) {
            btnToggleHashGate.text = "Gate: $hashThreshold"
            btnToggleHashGate.setTextColor(ContextCompat.getColor(this, R.color.primary))
        } else {
            btnToggleHashGate.text = "Gate: OFF"
            btnToggleHashGate.setTextColor(ContextCompat.getColor(this, R.color.text_muted))
        }
    }

    private fun selectFps(fps: Double, selectedButton: Button) {
        selectedFps = fps
        btnFps05.setBackgroundResource(R.drawable.btn_pill_inactive)
        btnFps05.setTextColor(ContextCompat.getColor(this, R.color.text_main))

        btnFps10.setBackgroundResource(R.drawable.btn_pill_inactive)
        btnFps10.setTextColor(ContextCompat.getColor(this, R.color.text_main))

        btnFps20.setBackgroundResource(R.drawable.btn_pill_inactive)
        btnFps20.setTextColor(ContextCompat.getColor(this, R.color.text_main))

        selectedButton.setBackgroundResource(R.drawable.btn_pill_active)
        selectedButton.setTextColor(Color.WHITE)
    }

    private fun resetAllData() {
        vectorStore.clear()
        sqliteFts.clearAll()
        acceptedFramesCount = 0
        droppedFramesCount = 0
        lastFrameHash = null
        currentVideoUri = null

        videoViewPlayback.stopPlayback()
        cardVideoPlayback.visibility = View.GONE

        // Clean extracted frames directory
        try {
            val framesDir = File(filesDir, "extracted_frames")
            if (framesDir.exists()) {
                framesDir.deleteRecursively()
            }
        } catch (_: Exception) {}

        tvStatus.text = "Active Surveillance Index: 0 region vectors & 0 FTS5 rows"
        tvIngestionInfo.text = "Select a video file (1.0 FPS Dense Mode: 100% frames indexed)."
        tvHashValue.text = if (!enableHashGate) "Gate: OFF (every frame indexed)" else "Gate: threshold $hashThreshold | awaiting first frame"
        tvGateMetrics.text = "Accepted: 0 | Dropped: 0"
        tvPyramidMetrics.text = "Spatial Pyramid: 6 crops / frame | Vectors: 0"

        tvMetricFrames.text = "0"
        tvMetricDropped.text = "(0 dropped)"
        tvMetricRegions.text = "0"
        tvMetricTime.text = "00:00"
        tvMetricTotalFrames.text = "active status"

        tvExpandedQuery.text = "Query Expansion: Ready (attribute & semantic dynamic synthesis)"
        tvResults.text = "Awaiting query. Upload a video file to extract keyframes, then search for any visual moment."
        layoutStoryboardThumbnails.removeAllViews()
        scrollStoryboard.visibility = View.GONE
        pbIngestion.visibility = View.GONE
    }

    /**
     * Process user-selected local video file (MP4/MKV).
     */
    private fun processSelectedVideoUri(uri: Uri) {
        currentVideoUri = uri
        ingestionStartTimeMs = System.currentTimeMillis()

        lifecycleScope.launch(Dispatchers.Default) {
            vectorStore.clear()
            sqliteFts.clearAll()
            acceptedFramesCount = 0
            droppedFramesCount = 0
            lastFrameHash = null

            withContext(Dispatchers.Main) {
                videoViewPlayback.stopPlayback()
                cardVideoPlayback.visibility = View.GONE
                pbIngestion.visibility = View.VISIBLE
                pbIngestion.isIndeterminate = true
                tvIngestionInfo.text = "Processing: ${uri.lastPathSegment ?: "video.mp4"} (1.0 FPS Dense Mode)"
                layoutStoryboardThumbnails.removeAllViews()
                scrollStoryboard.visibility = View.GONE
                tvResults.text = "Extracting video keyframes at ${selectedFps} FPS and indexing into MobileCLIP & SQLite FTS5..."
            }

            try {
                frameDecoder.decodeVideoUri(
                    videoUri = uri,
                    cameraName = "cam_user",
                    sampleFps = selectedFps,
                    onProgress = { currentSec, totalSec, frameIndex ->
                        lifecycleScope.launch(Dispatchers.Main) {
                            pbIngestion.isIndeterminate = false
                            val progress = if (totalSec > 0) ((currentSec * 100) / totalSec).toInt() else 0
                            pbIngestion.progress = progress

                            val curMin = currentSec / 60
                            val curS = currentSec % 60
                            val totMin = totalSec / 60
                            val totS = totalSec % 60
                            val timeStr = String.format(Locale.US, "%02d:%02d / %02d:%02d", curMin, curS, totMin, totS)

                            tvIngestionInfo.text = "Decoded Frame #$frameIndex ($timeStr - $progress%) | Total Vectors: ${vectorStore.size} | FTS5 Rows: ${sqliteFts.size()}"

                            val elapsedSec = (System.currentTimeMillis() - ingestionStartTimeMs) / 1000
                            val elapsedMin = elapsedSec / 60
                            val elapsedS = elapsedSec % 60
                            tvMetricTime.text = String.format(Locale.US, "%02d:%02d", elapsedMin, elapsedS)
                            tvMetricTotalFrames.text = "($frameIndex keyframes)"
                        }
                    },
                    onKeyframeDecoded = { bitmap, timestamp, epochTime, imagePath ->
                        ingestAndIndexFrame(bitmap, "cam_user", timestamp, epochTime, imagePath)
                    }
                )

                withContext(Dispatchers.Main) {
                    pbIngestion.visibility = View.GONE
                    val modeLabel = if (!enableHashGate) "gate off" else "dHash gate, threshold $hashThreshold"
                    tvIngestionInfo.text = "Ingestion Complete! Indexed ${acceptedFramesCount} keyframes (${droppedFramesCount} dropped) into ${vectorStore.size} Dense Vectors & ${sqliteFts.size()} SQLite FTS5 Rows ($modeLabel)."
                    tvStatus.text = "Active Surveillance Index: ${vectorStore.size} vectors & ${sqliteFts.size()} FTS5 tokens in RAM"
                    tvResults.text = "Video indexing complete! Enter any search query above to search footage."
                    Toast.makeText(this@MainActivity, "Indexed ${acceptedFramesCount} frames (${vectorStore.size} vectors + SQLite FTS5)!", Toast.LENGTH_LONG).show()
                }
            } catch (e: Throwable) {
                Log.e("VideoRAG_Ingest", "Ingestion failed", e)
                withContext(Dispatchers.Main) {
                    pbIngestion.visibility = View.GONE
                    // Distinguish a missing/unloadable CLIP model from a genuine decode
                    // failure - "Decode error" for a model problem sent us hunting in the
                    // wrong place.
                    val isModel = e is OnDeviceEmbedder.ModelUnavailableException ||
                                  e.cause is OnDeviceEmbedder.ModelUnavailableException
                    val what = if (isModel) "CLIP model error" else "Decode error"
                    val detail = e.message ?: e.cause?.message ?: e.javaClass.simpleName
                    tvIngestionInfo.text = "$what: $detail"
                    tvResults.text = "$what\n\n$detail\n\n(${e.javaClass.name})"
                }
            }
        }
    }

    /**
     * Download remote video URL in background and ingest keyframes.
     */
    private fun downloadAndIngestVideoUrl(videoUrl: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            withContext(Dispatchers.Main) {
                pbIngestion.visibility = View.VISIBLE
                pbIngestion.isIndeterminate = false
                pbIngestion.progress = 0
                tvIngestionInfo.text = "Connecting and downloading video stream..."
            }

            try {
                val downloadedFile = VideoDownloader.downloadVideo(
                    context = this@MainActivity,
                    videoUrl = videoUrl,
                    onProgress = { percent, downloaded, total ->
                        lifecycleScope.launch(Dispatchers.Main) {
                            if (percent >= 0) {
                                pbIngestion.progress = percent
                                tvIngestionInfo.text = "Downloading: $percent% (${downloaded / 1024} KB / ${total / 1024} KB)"
                            } else {
                                pbIngestion.isIndeterminate = true
                                tvIngestionInfo.text = "Downloading: ${downloaded / 1024} KB"
                            }
                        }
                    }
                )

                withContext(Dispatchers.Main) {
                    tvIngestionInfo.text = "Download complete. Extracting keyframes..."
                }

                val localUri = Uri.fromFile(downloadedFile)
                currentVideoUri = localUri
                processSelectedVideoUri(localUri)

            } catch (e: Throwable) {
                Log.e("VideoRAG_Downloader", "Download failed", e)
                withContext(Dispatchers.Main) {
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "URL Download Failed: ${e.message}"
                    Toast.makeText(this@MainActivity, "Download failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /**
     * STAGE 1 & 2: dHash Keyframe Filtering + Qwen VLM Visual Description Generation + FAISS & SQLite Indexing.
     */
    private suspend fun ingestAndIndexFrame(bitmap: Bitmap, camera: String, timestamp: String, epochTime: Long, imagePath: String) {
        val currentHash = MobileFrameFilter.calculateDHash(bitmap)
        val hashHex = MobileFrameFilter.formatHashHex(currentHash)

        // Distance from the last KEPT keyframe. 64 means "no baseline yet".
        var hammingDist = lastFrameHash?.let { MobileFrameFilter.hammingDistance(it, currentHash) } ?: 64

        if (enableHashGate) {
            val keep = MobileFrameFilter.isKeyframe(lastFrameHash, currentHash, hashThreshold)
            if (!keep) {
                droppedFramesCount++
                updateTelemetryUI(hashHex, hammingDist, dropped = true)
                return   // note: lastFrameHash is NOT advanced, so the next frame is
                         // still compared against the last frame we actually kept
            }
        }

        lastFrameHash = currentHash
        acceptedFramesCount++
        updateTelemetryUI(hashHex, hammingDist, dropped = false)

        // Step 1: cheap per-frame stats. No VLM load here - see getFrameDescriber().
        val vlm = orchestrator.getFrameDescriber()
        val frameJson = vlm.describeFrameAsJson(bitmap, timestamp, acceptedFramesCount, imagePath)
        val frameDescription = frameJson.getString("visual_description")

        // Step 2: index the frame as 6 spatial regions in CLIP's image space.
        //
        // No text/image blending any more: the old code mixed a hash-bucket text vector
        // with a colour histogram at 0.6/0.4, which combined two unrelated coordinate
        // systems. Queries are embedded with the text tower and frames with the image
        // tower, which is what puts them in one comparable space.
        //
        // Regions matter for small objects. Measured on this footage, a bus covering ~3%
        // of the frame left every candidate within 0.02 cosine of the others and ranked
        // the right frame 4th-5th; per-region embeddings with max-pooling at query time
        // move it to 2nd.
        val embedder = orchestrator.getActiveEmbedder()
        val crops = SpatialCropper.generatePyramidCrops(bitmap)
        for (crop in crops) {
            val v = embedder.embedImage(crop.bitmap)
            vectorStore.addMoment(
                IndexedMoment(
                    id = "${camera}_${timestamp}_${crop.label}",
                    camera = camera,
                    timestamp = timestamp,
                    epochTime = epochTime,
                    vector = v,
                    cropRegion = crop.label,
                    imagePath = imagePath,
                    description = frameDescription,
                    jsonMetadata = frameJson.toString()
                )
            )
            if (crop.bitmap != bitmap) crop.bitmap.recycle()
        }

        // One lexical row per FRAME (the dense index holds the 6 per-region vectors).
        sqliteFts.insertMoment(
            momentId = "${camera}_${timestamp}",
            camera = camera,
            timestamp = timestamp,
            epochTime = epochTime,
            cropRegion = "frame",
            imagePath = imagePath,
            visualTokens = frameDescription
        )

        withContext(Dispatchers.Main) {
            tvStatus.text = "Active Surveillance Index: ${vectorStore.size} described moments in FAISS & SQLite"
            tvIngestionInfo.text = "Ingesting Frame #${acceptedFramesCount} ($timestamp) ➔ JSON Generated:\n${frameJson.toString(2)}"
        }
    }

    private suspend fun updateTelemetryUI(hashHex: String, hammingDist: Int, dropped: Boolean) {
        withContext(Dispatchers.Main) {
            if (!enableHashGate) {
                tvHashValue.text = "Gate: OFF (every frame indexed) | dHash: 0x$hashHex"
                tvGateMetrics.text = "Accepted: $acceptedFramesCount / $acceptedFramesCount (100% Ingested - 0 Dropped)"
            } else {
                val deltaStatus = if (dropped) "Δ=$hammingDist (Static Dropped ❌)" else "Δ=$hammingDist (Motion Keyframe Accepted ✅)"
                tvHashValue.text = "Last dHash: 0x$hashHex | $deltaStatus"

                val total = acceptedFramesCount + droppedFramesCount
                val dropRate = if (total > 0) (droppedFramesCount * 100.0f / total) else 0.0f
                tvGateMetrics.text = String.format(
                    Locale.US,
                    "Accepted: %d | Dropped Static: %d (Gate Drop Rate: %.1f%%)",
                    acceptedFramesCount, droppedFramesCount, dropRate
                )
            }
            tvPyramidMetrics.text = "Index Mode: Qwen VLM Described Frames | Vectors: ${vectorStore.size} | FTS Rows: ${sqliteFts.size()}"

            // Update 3-Column Metrics Grid
            tvMetricFrames.text = "$acceptedFramesCount"
            tvMetricDropped.text = "($droppedFramesCount dropped)"
            tvMetricRegions.text = "${vectorStore.size}"
        }
    }

    /**
     * STAGE 3 & 4: Retrieve Top 5 Described Moments & Synthesize Answer with Qwen VLM Context Window.
     */
    private fun performLocalVideoRAGQuery(userQuery: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                if (vectorStore.size == 0 && sqliteFts.size() == 0) {
                    withContext(Dispatchers.Main) {
                        tvResults.text = "Index is empty! Please upload a video first using 'Upload Video' to extract and index keyframes."
                        Toast.makeText(this@MainActivity, "No video indexed yet. Upload a video first.", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    tvResults.text = "Retrieving Top 5 moments from FAISS/SQLite & Synthesizing answer via Qwen VLM Context Window..."
                    layoutStoryboardThumbnails.removeAllViews()
                    scrollStoryboard.visibility = View.GONE
                }

                val expandedQueries = expandQueryNatively(userQuery)
                withContext(Dispatchers.Main) {
                    tvExpandedQuery.text = "Query Expansions: ${expandedQueries.joinToString(" • ")}"
                }

                val pathMetadata = HashMap<String, IndexedMoment>()
                for (m in vectorStore.getAllMoments()) {
                    pathMetadata[m.imagePath] = m
                }

                // Each frame is indexed as 6 regions, so a raw top-K would return several
                // crops of the same frame. Max-pool to the best-scoring region per frame:
                // the frame's score is its most relevant region, which is what makes a
                // small object competitive against frames that are similar overall.
                val embedder = orchestrator.getActiveEmbedder()
                val bestPerFrame = HashMap<String, Pair<IndexedMoment, Float>>()
                for (expandedQ in expandedQueries) {
                    val queryVector = embedder.embedText(expandedQ)
                    // pull deeper than we need, since 6 regions share each frame
                    for ((moment, score) in vectorStore.search(queryVector, topK = 40)) {
                        val prev = bestPerFrame[moment.imagePath]
                        if (prev == null || score > prev.second) {
                            bestPerFrame[moment.imagePath] = moment to score
                        }
                    }
                }
                val denseHits = bestPerFrame.values.sortedByDescending { it.second }.toMutableList()
                Log.i("VideoRAG_Query", "dense: ${denseHits.size} frames; top=" +
                    denseHits.take(5).joinToString { "${it.first.timestamp}[${it.first.cropRegion}]=%.3f".format(it.second) })

                val sparseHits = sqliteFts.searchSparse(userQuery, topK = 5)

                // Select Top 5 retrieved moments
                val fusedResults = HybridRetriever.fuseRRF(denseHits, sparseHits, pathMetadata, topK = 5)

                if (fusedResults.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        tvResults.text = "No matching moments found in the current video for: '$userQuery'."
                    }
                    return@launch
                }

                val top5Moments = fusedResults.map { it.moment }.sortedBy { it.timestamp }

                // Render Top 5 Thumbnails in UI with real descriptions
                withContext(Dispatchers.Main) {
                    renderHybridStoryboardThumbnails(top5Moments, fusedResults)
                }

                // Send Top 5 retrieved descriptions as Context Window to Qwen VLM
                val vlm = orchestrator.getActiveVLM()
                val finalAnswer = vlm.answerFromRetrievedContext(
                    query = userQuery,
                    top5Moments = top5Moments
                )

                withContext(Dispatchers.Main) {
                    Log.i("VideoRAG_Results", "Reasoning Complete: $finalAnswer")
                    tvResults.text = finalAnswer
                }

            } catch (e: Throwable) {
                Log.e("VideoRAG_Query", "Error executing query", e)
                withContext(Dispatchers.Main) {
                    tvResults.text = "Query execution error: ${e.localizedMessage ?: e.javaClass.simpleName}\n\nCheck Logcat for details."
                }
            }
        }
    }

    private fun renderHybridStoryboardThumbnails(
        moments: List<IndexedMoment>,
        fusedResults: List<HybridSearchResult>
    ) {
        layoutStoryboardThumbnails.removeAllViews()
        scrollStoryboard.visibility = View.VISIBLE

        val scoreMap = fusedResults.associateBy { it.moment.imagePath }

        for (moment in moments) {
            val result = scoreMap[moment.imagePath]

            val card = CardView(this).apply {
                radius = 12f
                setCardBackgroundColor(Color.WHITE)
                cardElevation = 2f
                useCompatPadding = true
                isClickable = true
                isFocusable = true

                val params = LinearLayout.LayoutParams(260, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    marginEnd = 12
                }
                layoutParams = params
            }

            val cardLayout = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(6, 6, 6, 8)
                val borderDrawable = GradientDrawable().apply {
                    setColor(Color.WHITE)
                    setStroke(1, Color.parseColor("#E2E8F0"))
                    cornerRadius = 12f
                }
                background = borderDrawable
            }

            val frameContainer = FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    150
                )
            }

            val imageView = ImageView(this).apply {
                layoutParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
                )
                scaleType = ImageView.ScaleType.CENTER_CROP
                val file = File(moment.imagePath)
                if (file.exists()) {
                    val bmp = BitmapFactory.decodeFile(file.absolutePath)
                    setImageBitmap(bmp)
                } else {
                    setBackgroundColor(Color.parseColor("#E2E8F0"))
                }
            }

            val playBadge = TextView(this).apply {
                text = "▶ Play"
                setTextColor(Color.WHITE)
                textSize = 9.5f
                setPadding(8, 3, 8, 3)
                setBackgroundColor(Color.parseColor("#D92563EB"))
                val badgeParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = 4
                    rightMargin = 4
                    gravity = android.view.Gravity.BOTTOM or android.view.Gravity.END
                }
                layoutParams = badgeParams
            }

            frameContainer.addView(imageView)
            frameContainer.addView(playBadge)

            // RRF scores are ~1/(60+rank); they are a ranking signal, not a
            // probability, so the old (rrfScore * 3000).coerceIn(15,99) presented an
            // arbitrary number as a confidence. Show the rank instead, which is what
            // the score actually encodes.
            val rank = fusedResults.indexOfFirst { it.moment.imagePath == moment.imagePath }
            val rankLabel = if (rank >= 0) "#${rank + 1}" else "-"
            val matchType = result?.matchType ?: "Hybrid"

            val tvTimestamp = TextView(this).apply {
                text = moment.timestamp
                setTextColor(Color.parseColor("#64748B"))
                textSize = 10f
                setPadding(2, 4, 0, 0)
            }

            val tvMatch = TextView(this).apply {
                text = "Rank $rankLabel · $matchType"
                setTextColor(Color.parseColor("#2563EB"))
                textSize = 10f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setPadding(2, 1, 0, 0)
            }

            val tvDesc = TextView(this).apply {
                text = if (moment.description.length > 55) moment.description.take(52) + "..." else moment.description
                setTextColor(Color.parseColor("#475569"))
                textSize = 9.0f
                setPadding(2, 1, 0, 2)
            }

            cardLayout.addView(frameContainer)
            cardLayout.addView(tvTimestamp)
            cardLayout.addView(tvMatch)
            cardLayout.addView(tvDesc)
            card.addView(cardLayout)

            // Click-to-Play Video at exact timestamp
            card.setOnClickListener {
                playVideoAtTimestamp(moment.timestamp)
            }

            layoutStoryboardThumbnails.addView(card)
        }
    }

    /**
     * Plays video in the embedded VideoView starting at the selected timestamp.
     */
    private fun playVideoAtTimestamp(timestamp: String) {
        val uri = currentVideoUri
        if (uri == null) {
            Toast.makeText(this, "Original video source not loaded.", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val parts = timestamp.split(":")
            val hh = if (parts.isNotEmpty()) parts[0].toIntOrNull() ?: 0 else 0
            val mm = if (parts.size > 1) parts[1].toIntOrNull() ?: 0 else 0
            val ss = if (parts.size > 2) parts[2].toIntOrNull() ?: 0 else 0
            val timeMs = (hh * 3600 + mm * 60 + ss) * 1000

            lastSelectedTimestampMs = timeMs

            cardVideoPlayback.visibility = View.VISIBLE
            tvPlayerTimestamp.text = "Playing from timestamp: $timestamp (${timeMs / 1000}s in video)"

            videoViewPlayback.setVideoURI(uri)
            videoViewPlayback.seekTo(timeMs)
            videoViewPlayback.start()
            btnPlayPause.text = "⏸ Pause"

            scrollView.post {
                scrollView.smoothScrollTo(0, cardVideoPlayback.top)
            }

            Toast.makeText(this, "Playing video at $timestamp", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("VideoPlay", "Failed to play video at $timestamp", e)
            Toast.makeText(this, "Error playing video: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Opens a dedicated modal inspector displaying all indexed keyframe JSON documents.
     */
    private fun showAllJsonDialog() {
        val allMoments = vectorStore.getAllMoments()
        if (allMoments.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("📋 No Indexed JSON Data")
                .setMessage("No video keyframes have been indexed yet.\n\nUpload a video first using '📁 Upload' to extract frames, generate visual descriptions, and index them into structured JSON.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val jsonArray = JSONArray()
        for (m in allMoments) {
            jsonArray.put(m.toJsonObject())
        }
        val formattedJson = jsonArray.toString(2)

        val dialogView = layoutInflater.inflate(R.layout.dialog_view_all_json, null)
        val tvDialogTitle = dialogView.findViewById<TextView>(R.id.tvDialogTitle)
        val tvJsonCount = dialogView.findViewById<TextView>(R.id.tvJsonCount)
        val tvJsonContent = dialogView.findViewById<TextView>(R.id.tvJsonContent)
        val btnCopyJson = dialogView.findViewById<Button>(R.id.btnCopyJson)
        val btnCloseJsonDialog = dialogView.findViewById<Button>(R.id.btnCloseJsonDialog)

        tvDialogTitle.text = "📋 Indexed Keyframe JSON"
        tvJsonCount.text = "Total Indexed Moments: ${allMoments.size} keyframes (${formattedJson.length} bytes)"
        tvJsonContent.text = formattedJson

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        dialog.window?.setBackgroundDrawable(android.graphics.drawable.ColorDrawable(android.graphics.Color.TRANSPARENT))

        btnCopyJson.setOnClickListener {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("Indexed Keyframes JSON", formattedJson)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(this, "Copied all JSON (${allMoments.size} frames) to clipboard! 📋", Toast.LENGTH_SHORT).show()
        }

        btnCloseJsonDialog.setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun expandQueryNatively(query: String): List<String> {
        val qLow = query.lowercase().trim()
        val expansions = mutableListOf(query)

        val words = qLow.split(Regex("\\s+")).filter { it.length > 2 }
        for (w in words) {
            expansions.add(w)
        }

        return expansions.distinct()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            videoViewPlayback.stopPlayback()
        } catch (_: Exception) {}
        lifecycleScope.launch(Dispatchers.IO) {
            orchestrator.releaseAll()
        }
    }
}
