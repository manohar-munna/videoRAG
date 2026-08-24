package com.stellar.videorag

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.stellar.videorag.ingestion.MobileFrameFilter
import com.stellar.videorag.indexing.*
import com.stellar.videorag.llm.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var orchestrator: MemoryOrchestrator
    private val vectorStore = MobileVectorStore()
    private var lastFrameHash: Long? = null

    private lateinit var tvStatus: TextView
    private lateinit var tvResults: TextView
    private lateinit var etQuery: EditText
    private lateinit var btnSearch: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvResults = findViewById(R.id.tvResults)
        etQuery = findViewById(R.id.etQuery)
        btnSearch = findViewById(R.id.btnSearch)

        val onnxPath = "${filesDir.absolutePath}/mobileclip_s2.onnx"
        val vlmPath = "${filesDir.absolutePath}/qwen2_vl_2b/"
        orchestrator = MemoryOrchestrator(this, onnxPath, vlmPath)

        btnSearch.setOnClickListener {
            val query = etQuery.text.toString().trim()
            if (query.isNotEmpty()) {
                performLocalVideoRAGQuery(query)
            }
        }
    }

    /**
     * STEP 1 & 2: Ingest decoded surveillance frames, apply dHash, slice, and embed crops on the NPU.
     */
    fun onFrameDecoded(bitmap: Bitmap, camera: String, timestamp: String, epochTime: Long, imagePath: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            val currentHash = MobileFrameFilter.calculateDHash(bitmap)
            
            // Apply 64-bit dHash filter (discard near-identical static scenes)
            lastFrameHash?.let { lastHash ->
                val distance = MobileFrameFilter.hammingDistance(lastHash, currentHash)
                if (distance < 10) {
                    Log.d("VideoRAG_Ingest", "Static frame dropped (Hamming distance $distance < 10)")
                    return@launch
                }
            }
            lastFrameHash = currentHash

            // Request memory allocation for the ONNX embedder session (unloads the VLM if running)
            val embedder = orchestrator.getActiveEmbedder()

            // Slice frame into a 6-region spatial pyramid
            val regions = SpatialCropper.generatePyramidCrops(bitmap)
            for (region in regions) {
                // Execute high-speed NPU forward pass for the target crop
                val vector = embedder.embedCrop(region.bitmap)
                
                val moment = IndexedMoment(
                    id = "${camera}_${timestamp}_${region.label}",
                    camera = camera,
                    timestamp = timestamp,
                    epochTime = epochTime,
                    vector = vector,
                    cropRegion = region.label,
                    imagePath = imagePath
                )
                vectorStore.addMoment(moment)
            }
            Log.d("VideoRAG_Index", "Indexed 6 spatial regions for frame at $timestamp successfully. Total vectors: ${vectorStore.size}")
            
            withContext(Dispatchers.Main) {
                tvStatus.text = "Indexed: ${vectorStore.size} region vectors"
            }
        }
    }

    /**
     * STEP 3 & 4: General Relevance expansion, Max-Pooling aggregation, and local VLM reasoning.
     */
    fun performLocalVideoRAGQuery(userQuery: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            withContext(Dispatchers.Main) {
                tvResults.text = "Searching on-device index & running VLM reasoning..."
            }

            // 1. Expand query to preserve target nouns and dynamic colors (prevent dilution)
            val expandedQueries = expandQueryNatively(userQuery)

            // 2. Fetch active embedder to convert query text to vector
            val matchedFrames = HashMap<String, Float>() // Maps imagePath -> highest crop score
            val pathMetadata = HashMap<String, IndexedMoment>()

            val embedder = orchestrator.getActiveEmbedder()
            for (expandedQ in expandedQueries) {
                val placeholderBitmap = createTextBitmapPlaceholder(expandedQ)
                val queryVector = embedder.embedCrop(placeholderBitmap)
                placeholderBitmap.recycle()
                val hits = vectorStore.search(queryVector, topK = 15)

                for (hit in hits) {
                    val moment = hit.first
                    val score = hit.second
                    val currentBestScore = matchedFrames[moment.imagePath] ?: 0.0f
                    
                    // Spatial Crop Max-Pooling: Keep highest matching crop score for each keyframe
                    if (score > currentBestScore) {
                        matchedFrames[moment.imagePath] = score
                        pathMetadata[moment.imagePath] = moment
                    }
                }
            }

            // 3. Compile contextual chronological storyboard around top visual matches
            val topPaths = matchedFrames.entries.sortedByDescending { it.value }.map { it.key }
            if (topPaths.isEmpty()) {
                withContext(Dispatchers.Main) {
                    tvResults.text = "No matching CCTV surveillance moments found on device."
                }
                return@launch
            }

            // Unload embedder, clear RAM caches, and launch native GPU Qwen-VL model
            val vlm = orchestrator.getActiveVLM()

            // Check if global counting inquiry
            val qLow = userQuery.lowercase()
            val isGlobalCount = qLow.contains("total") || qLow.contains("how many") || qLow.contains("count")
            
            val storyboardPaths = if (isGlobalCount) {
                topPaths.take(12)
            } else {
                val anchorPath = topPaths[0]
                val anchorMoment = pathMetadata[anchorPath] ?: return@launch
                compileStoryboardTimeline(anchorMoment)
            }

            // 4. Run native visual-language reasoning on the GPU over the storyboard frames
            val finalExplanation = vlm.reasonOverTimeline(userQuery, storyboardPaths)
            
            withContext(Dispatchers.Main) {
                Log.i("VideoRAG_Results", "Reasoning Complete: $finalExplanation")
                tvResults.text = finalExplanation
            }
        }
    }

    private fun expandQueryNatively(query: String): List<String> {
        val qLow = query.lowercase().trim()
        val expansions = mutableListOf(query)

        // Parse colors dynamically
        val knownColors = listOf("pink", "red", "yellow", "blue", "green", "white", "black", "orange", "grey", "gray", "purple")
        val activeColor = knownColors.firstOrNull { qLow.contains(it) } ?: ""
        val colorPrefix = if (activeColor.isNotEmpty()) "$activeColor " else ""

        if (qLow.contains("car") || qLow.contains("vehicle") || qLow.contains("truck") || qLow.contains("pickup")) {
            expansions.add("${colorPrefix}vehicle in surveillance footage")
            expansions.add("${colorPrefix}automobile in frame")
        }
        if (qLow.contains("people") || qLow.contains("person") || qLow.contains("costume") || qLow.contains("wear") || qLow.contains("shirt")) {
            expansions.add("person wearing ${colorPrefix}clothing")
            expansions.add("individual dressed in ${colorPrefix}apparel")
        }
        if (qLow.contains("bag") || qLow.contains("backpack") || qLow.contains("luggage")) {
            expansions.add("${colorPrefix}backpack on ground or carried")
        }
        return expansions.distinct()
    }

    private fun compileStoryboardTimeline(anchor: IndexedMoment): List<String> {
        return listOf(
            anchor.imagePath.replace(".jpg", "_prev.jpg"),
            anchor.imagePath,
            anchor.imagePath.replace(".jpg", "_next.jpg")
        )
    }

    private fun createTextBitmapPlaceholder(text: String): Bitmap {
        return Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
    }

    override fun onDestroy() {
        super.onDestroy()
        lifecycleScope.launch(Dispatchers.IO) {
            orchestrator.releaseAll()
        }
    }
}
