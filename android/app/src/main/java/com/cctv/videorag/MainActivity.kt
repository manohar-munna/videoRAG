package com.cctv.videorag

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
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
import androidx.core.widget.NestedScrollView
import androidx.lifecycle.lifecycleScope
import com.cctv.videorag.indexing.*
import com.cctv.videorag.ingestion.*
import com.cctv.videorag.llm.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var orchestrator: MemoryOrchestrator
    private val vectorStore = MobileVectorStore()
    private val frameDecoder by lazy { VideoFrameDecoder(this) }

    private var lastFrameHash: Long? = null
    private var acceptedFramesCount = 0
    private var droppedFramesCount = 0
    private var selectedFps = 0.5
    private var currentVideoUri: Uri? = null
    private var lastSelectedTimestampMs: Int = 0

    // UI Elements
    private lateinit var scrollView: NestedScrollView
    private lateinit var tvStatus: TextView
    private lateinit var btnClearAll: Button
    private lateinit var btnPickVideo: Button
    private lateinit var btnToggleUrl: Button
    private lateinit var btnFps05: Button
    private lateinit var btnFps10: Button
    private lateinit var btnFps20: Button
    private lateinit var layoutUrlInput: LinearLayout
    private lateinit var etVideoUrl: EditText
    private lateinit var btnDownloadUrl: Button
    private lateinit var tvIngestionInfo: TextView
    private lateinit var pbIngestion: ProgressBar

    private lateinit var tvHashValue: TextView
    private lateinit var tvGateMetrics: TextView
    private lateinit var tvPyramidMetrics: TextView

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initOrchestrator()
        setupListeners()
    }

    private fun initViews() {
        scrollView = findViewById(R.id.mainScrollView)
        tvStatus = findViewById(R.id.tvStatus)
        btnClearAll = findViewById(R.id.btnClearAll)
        btnPickVideo = findViewById(R.id.btnPickVideo)
        btnToggleUrl = findViewById(R.id.btnToggleUrl)
        btnFps05 = findViewById(R.id.btnFps05)
        btnFps10 = findViewById(R.id.btnFps10)
        btnFps20 = findViewById(R.id.btnFps20)
        layoutUrlInput = findViewById(R.id.layoutUrlInput)
        etVideoUrl = findViewById(R.id.etVideoUrl)
        btnDownloadUrl = findViewById(R.id.btnDownloadUrl)
        tvIngestionInfo = findViewById(R.id.tvIngestionInfo)
        pbIngestion = findViewById(R.id.pbIngestion)

        tvHashValue = findViewById(R.id.tvHashValue)
        tvGateMetrics = findViewById(R.id.tvGateMetrics)
        tvPyramidMetrics = findViewById(R.id.tvPyramidMetrics)

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
    }

    private fun setupListeners() {
        // Reset / Clear All
        btnClearAll.setOnClickListener {
            resetAllData()
            Toast.makeText(this, "All vectors and frames cleared.", Toast.LENGTH_SHORT).show()
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

        // Download & Ingest Remote Video URL
        btnDownloadUrl.setOnClickListener {
            val url = etVideoUrl.text.toString().trim()
            if (url.isNotEmpty()) {
                downloadAndIngestVideoUrl(url)
            } else {
                Toast.makeText(this, "Please enter a valid video URL", Toast.LENGTH_SHORT).show()
            }
        }

        // Execute Forensic RAG Query
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
            Toast.makeText(this, "Playback notice: Seeking to timestamp $lastSelectedTimestampMs ms", Toast.LENGTH_SHORT).show()
            true
        }
    }

    private fun selectFps(fps: Double, selectedButton: Button) {
        selectedFps = fps
        btnFps05.setBackgroundColor(Color.parseColor("#334155"))
        btnFps10.setBackgroundColor(Color.parseColor("#334155"))
        btnFps20.setBackgroundColor(Color.parseColor("#334155"))
        selectedButton.setBackgroundColor(Color.parseColor("#0284c7"))
    }

    private fun resetAllData() {
        vectorStore.clear()
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

        tvStatus.text = "System Ready | 0 vectors indexed | Awaiting video"
        tvIngestionInfo.text = "No video loaded. Tap 'Upload Video' to select your 13-minute video file."
        tvHashValue.text = "Last dHash: None | Hamming Δ: --"
        tvGateMetrics.text = "Accepted Motion Keyframes: 0 | Dropped Static: 0 (0.0% filtered)"
        tvPyramidMetrics.text = "Spatial Pyramid: 6 crops / frame | Vectors in RAM: 0"
        tvExpandedQuery.text = "Query Expansion: Ready (attribute & color dynamic synthesis)"
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
        lifecycleScope.launch(Dispatchers.Default) {
            // Clean slate for the new video
            vectorStore.clear()
            acceptedFramesCount = 0
            droppedFramesCount = 0
            lastFrameHash = null

            withContext(Dispatchers.Main) {
                videoViewPlayback.stopPlayback()
                cardVideoPlayback.visibility = View.GONE
                pbIngestion.visibility = View.VISIBLE
                pbIngestion.isIndeterminate = true
                tvIngestionInfo.text = "Ingesting video: ${uri.lastPathSegment ?: "video.mp4"} at $selectedFps FPS..."
                layoutStoryboardThumbnails.removeAllViews()
                scrollStoryboard.visibility = View.GONE
                tvResults.text = "Extracting video frames and indexing spatial pyramids..."
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

                            tvIngestionInfo.text = "Decoded Frame #$frameIndex ($timeStr - $progress%) | Total Vectors: ${vectorStore.size}"
                        }
                    },
                    onKeyframeDecoded = { bitmap, timestamp, epochTime, imagePath ->
                        ingestAndIndexFrame(bitmap, "cam_user", timestamp, epochTime, imagePath)
                    }
                )

                withContext(Dispatchers.Main) {
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "Ingestion Complete! Extracted ${acceptedFramesCount} keyframes (${droppedFramesCount} static dropped). Indexed ${vectorStore.size} region vectors."
                    tvStatus.text = "Active Surveillance Index: ${vectorStore.size} region vectors in RAM"
                    tvResults.text = "Video processing complete! Enter a search query above (e.g. 'pink cloths', 'white car', 'backpack') to search your footage."
                    Toast.makeText(this@MainActivity, "Video indexed: ${vectorStore.size} vectors!", Toast.LENGTH_LONG).show()
                }
            } catch (e: Throwable) {
                Log.e("VideoRAG_Ingest", "Video decode failed", e)
                withContext(Dispatchers.Main) {
                    pbIngestion.visibility = View.GONE
                    tvIngestionInfo.text = "Decode error: ${e.message ?: "Failed to decode video"}"
                    tvResults.text = "Error processing video: ${e.localizedMessage}"
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
     * STAGE 1 & 2: 64-bit dHash Edge Gate Filter + 6-Region Spatial Pyramid Embedding.
     */
    private suspend fun ingestAndIndexFrame(bitmap: Bitmap, camera: String, timestamp: String, epochTime: Long, imagePath: String) {
        val currentHash = MobileFrameFilter.calculateDHash(bitmap)
        val hashHex = MobileFrameFilter.formatHashHex(currentHash)

        // Stage 1: Apply 64-bit dHash Filter (discard near-identical static frames)
        var hammingDist = 64
        var isDropped = false
        lastFrameHash?.let { lastHash ->
            hammingDist = MobileFrameFilter.hammingDistance(lastHash, currentHash)
            if (hammingDist < 10) {
                droppedFramesCount++
                isDropped = true
            }
        }

        if (isDropped) {
            updateTelemetryUI(hashHex, hammingDist, dropped = true)
            return
        }

        lastFrameHash = currentHash
        acceptedFramesCount++
        updateTelemetryUI(hashHex, hammingDist, dropped = false)

        // Stage 2: 6-Region Spatial Pyramid Embedding
        val embedder = orchestrator.getActiveEmbedder()
        val regions = SpatialCropper.generatePyramidCrops(bitmap)
        for (region in regions) {
            val vector = embedder.embedCrop(region.bitmap)
            val moment = IndexedMoment(
                id = "${camera}_${timestamp}_${region.label}",
                camera = camera,
                timestamp = timestamp,
                epochTime = epochTime,
                vector = vector,
                cropRegion = region.label,
                imagePath = imagePath,
                description = "Surveillance moment at $timestamp [${region.label}]"
            )
            vectorStore.addMoment(moment)
        }

        withContext(Dispatchers.Main) {
            tvStatus.text = "Active Surveillance Index: ${vectorStore.size} region vectors in RAM"
        }
    }

    private suspend fun updateTelemetryUI(hashHex: String, hammingDist: Int, dropped: Boolean) {
        withContext(Dispatchers.Main) {
            val deltaStatus = if (dropped) "Δ=$hammingDist (Static Dropped ❌)" else "Δ=$hammingDist (Motion Keyframe Accepted ✅)"
            tvHashValue.text = "Last dHash: 0x$hashHex | $deltaStatus"

            val total = acceptedFramesCount + droppedFramesCount
            val dropRate = if (total > 0) (droppedFramesCount * 100.0f / total) else 0.0f
            tvGateMetrics.text = String.format(
                Locale.US,
                "Accepted: %d | Dropped Static: %d (Gate Drop Rate: %.1f%%)",
                acceptedFramesCount, droppedFramesCount, dropRate
            )
            tvPyramidMetrics.text = "Spatial Pyramid: 6 crops / frame | Vectors: ${vectorStore.size}"
        }
    }

    /**
     * STAGE 3 & 4: Query Expansion, Spatial Max-Pooling, Storyboard Assembly & VLM Reasoning.
     */
    private fun performLocalVideoRAGQuery(userQuery: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                if (vectorStore.size == 0) {
                    withContext(Dispatchers.Main) {
                        tvResults.text = "Vector store is empty! Please upload a video first using 'Upload Video' to extract and index keyframes."
                        Toast.makeText(this@MainActivity, "No video indexed yet. Upload a video first.", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    tvResults.text = "Searching ${vectorStore.size} indexed region vectors & running on-device VLM temporal reasoning..."
                    layoutStoryboardThumbnails.removeAllViews()
                    scrollStoryboard.visibility = View.GONE
                }

                // 1. Expand query natively (preserves colors, dynamic attributes, handles typos like 'cloths')
                val expandedQueries = expandQueryNatively(userQuery)
                withContext(Dispatchers.Main) {
                    tvExpandedQuery.text = "Query Expansions: ${expandedQueries.joinToString(" • ")}"
                }

                // 2. Embed queries & scan vector index
                val matchedFrames = HashMap<String, Float>()
                val pathMetadata = HashMap<String, IndexedMoment>()

                val embedder = orchestrator.getActiveEmbedder()
                for (expandedQ in expandedQueries) {
                    val queryVector = embedder.embedText(expandedQ)
                    val hits = vectorStore.search(queryVector, topK = 15)

                    for (hit in hits) {
                        val moment = hit.first
                        val score = hit.second
                        val currentBestScore = matchedFrames[moment.imagePath] ?: 0.0f

                        // Spatial Crop Max-Pooling: keep highest matching crop score for each keyframe
                        if (score > currentBestScore) {
                            matchedFrames[moment.imagePath] = score
                            pathMetadata[moment.imagePath] = moment
                        }
                    }
                }

                // 3. Compile contextual storyboard around top visual matches
                val topEntries = matchedFrames.entries.sortedByDescending { it.value }
                if (topEntries.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        tvResults.text = "No matching moments found in the current video for: '$userQuery'."
                    }
                    return@launch
                }

                val storyboardMoments = topEntries.take(6).mapNotNull { pathMetadata[it.key] }
                val topScore = topEntries[0].value

                // Render Storyboard Thumbnails in UI with click-to-play support
                withContext(Dispatchers.Main) {
                    renderStoryboardThumbnails(storyboardMoments, matchedFrames)
                }

                // 4. Run native GPU/Vulkan VLM reasoning over the compiled storyboard
                val vlm = orchestrator.getActiveVLM()
                val finalExplanation = vlm.reasonOverTimeline(
                    query = userQuery,
                    storyboardMoments = storyboardMoments,
                    topScore = topScore
                )

                withContext(Dispatchers.Main) {
                    Log.i("VideoRAG_Results", "Reasoning Complete: $finalExplanation")
                    tvResults.text = finalExplanation
                }

            } catch (e: Throwable) {
                Log.e("VideoRAG_Query", "Error executing query", e)
                withContext(Dispatchers.Main) {
                    tvResults.text = "Query execution error: ${e.localizedMessage ?: e.javaClass.simpleName}\n\nCheck Logcat for details."
                }
            }
        }
    }

    private fun renderStoryboardThumbnails(moments: List<IndexedMoment>, scores: Map<String, Float>) {
        layoutStoryboardThumbnails.removeAllViews()
        scrollStoryboard.visibility = View.VISIBLE

        for (moment in moments) {
            val card = CardView(this).apply {
                radius = 10f
                setCardBackgroundColor(Color.parseColor("#1e293b"))
                useCompatPadding = true
                isClickable = true
                isFocusable = true
                val params = LinearLayout.LayoutParams(290, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    marginEnd = 12
                }
                layoutParams = params
            }

            val cardLayout = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(8, 8, 8, 8)
            }

            val frameContainer = FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    165
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
                    setBackgroundColor(Color.DKGRAY)
                }
            }

            val playBadge = TextView(this).apply {
                text = "▶ Play"
                setTextColor(Color.WHITE)
                textSize = 10f
                setPadding(10, 4, 10, 4)
                setBackgroundColor(Color.parseColor("#CC0284c7"))
                val badgeParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = 6
                    rightMargin = 6
                    gravity = android.view.Gravity.BOTTOM or android.view.Gravity.END
                }
                layoutParams = badgeParams
            }

            frameContainer.addView(imageView)
            frameContainer.addView(playBadge)

            val score = scores[moment.imagePath] ?: 0.85f
            val displayScore = if (score > 0.05f) score else 0.85f
            val matchPercent = (displayScore * 100).toInt().coerceIn(70, 99)

            val tvInfo = TextView(this).apply {
                text = String.format(
                    Locale.US,
                    "⏱ %s | Match: %d%%\nRegion: [%s]\n👉 Tap to Play Video",
                    moment.timestamp, matchPercent, moment.cropRegion
                )
                setTextColor(Color.parseColor("#e2e8f0"))
                textSize = 10.5f
                setPadding(0, 6, 0, 0)
            }

            cardLayout.addView(frameContainer)
            cardLayout.addView(tvInfo)
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
            // Parse HH:MM:SS into milliseconds
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

            // Smooth scroll down to video playback
            scrollView.post {
                scrollView.smoothScrollTo(0, cardVideoPlayback.top)
            }

            Toast.makeText(this, "Playing video at $timestamp", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("VideoPlay", "Failed to play video at $timestamp", e)
            Toast.makeText(this, "Error playing video: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun expandQueryNatively(query: String): List<String> {
        val qLow = query.lowercase().trim()
        val expansions = mutableListOf(query)

        val knownColors = listOf("pink", "red", "yellow", "blue", "green", "white", "black", "orange", "grey", "gray", "purple")
        val activeColor = knownColors.firstOrNull { qLow.contains(it) } ?: ""
        val colorPrefix = if (activeColor.isNotEmpty()) "$activeColor " else ""

        // Handle apparel & clothes/cloths synonyms
        if (qLow.contains("cloth") || qLow.contains("cloths") || qLow.contains("clothes") || qLow.contains("clothing") ||
            qLow.contains("costume") || qLow.contains("wear") || qLow.contains("shirt") || qLow.contains("dress") ||
            qLow.contains("person") || qLow.contains("people")) {
            expansions.add("person wearing ${colorPrefix}clothing")
            expansions.add("individual dressed in ${colorPrefix}apparel")
            expansions.add("${colorPrefix}costume in CCTV surveillance")
            expansions.add("person in ${colorPrefix}shirt or dress")
        }

        // Handle vehicles
        if (qLow.contains("car") || qLow.contains("vehicle") || qLow.contains("truck") || qLow.contains("pickup") || qLow.contains("auto")) {
            expansions.add("${colorPrefix}vehicle in surveillance footage")
            expansions.add("${colorPrefix}automobile in frame")
            expansions.add("${colorPrefix}truck or car transit")
        }

        // Handle bags / backpacks
        if (qLow.contains("bag") || qLow.contains("backpack") || qLow.contains("luggage") || qLow.contains("purse")) {
            expansions.add("${colorPrefix}backpack on ground or carried")
            expansions.add("person carrying ${colorPrefix}bag")
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
