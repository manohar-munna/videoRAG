package com.cctv.videorag.llm

import android.content.Context
import android.os.Environment
import android.util.Log
import com.cctv.videorag.indexing.IndexedMoment
import java.io.File

class OnDeviceVLM(private val context: Context, private val defaultModelDirectory: String) {
    
    // External JNI hooks into native-lib.cpp
    private external fun nativeInitWithFiles(modelPath: String, mmprojPath: String, layersToOffload: Int): Long
    private external fun nativeInit(modelDir: String, layersToOffload: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, imagePaths: Array<String>): String
    private external fun nativeGetModelInfo(handle: Long): String
    private external fun nativeClose(handle: Long)

    private var nativeHandle: Long = 0
    private var activeModelDirectory: String? = null
    private var activeModelFileName: String? = null

    var customModelDirectory: String? = null
        set(value) {
            field = value
            if (nativeHandle != 0L) {
                unloadVLM()
            }
            loadVLM()
        }

    companion object {
        init {
            try {
                System.loadLibrary("llama_jni")
            } catch (e: UnsatisfiedLinkError) {
                Log.e("VideoRAG_VLM", "CRITICAL: Native library llama_jni failed to load: ${e.message}")
            }
        }
    }

    /**
     * Discovers exact GGUF model file and mmproj projector file in mobile storage.
     */
    fun findActiveModelFiles(): Pair<File, File?>? {
        val candidatePaths = mutableListOf<String>()
        customModelDirectory?.let { candidatePaths.add(it) }

        candidatePaths.addAll(
            listOf(
                "/storage/emulated/0/Download/qwen2_vl_2b",
                "/storage/emulated/0/Downloads/qwen2_vl_2b",
                "/storage/emulated/0/qwen2_vl_2b",
                File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "qwen2_vl_2b").absolutePath,
                "/sdcard/Download/qwen2_vl_2b",
                "/sdcard/Downloads/qwen2_vl_2b",
                "/sdcard/qwen2_vl_2b",
                File(Environment.getExternalStorageDirectory(), "Download/qwen2_vl_2b").absolutePath,
                File(context.filesDir, "qwen2_vl_2b").absolutePath,
                File(context.getExternalFilesDir(null), "qwen2_vl_2b").absolutePath,
                defaultModelDirectory
            )
        )

        // 1. Check direct candidate directories
        for (path in candidatePaths) {
            val dir = File(path)
            if (dir.exists() && dir.isDirectory) {
                val files = dir.listFiles() ?: emptyArray()
                val modelFile = files.firstOrNull {
                    it.extension.lowercase() == "gguf" &&
                    !it.name.contains("mmproj", ignoreCase = true) &&
                    it.length() > 50_000_000L
                }
                if (modelFile != null) {
                    val mmprojFile = files.firstOrNull {
                        it.extension.lowercase() == "gguf" &&
                        it.name.contains("mmproj", ignoreCase = true)
                    }
                    activeModelDirectory = dir.absolutePath
                    activeModelFileName = modelFile.name
                    Log.i("VideoRAG_VLM", "Discovered Model: ${modelFile.absolutePath}, Projector: ${mmprojFile?.absolutePath}")
                    return Pair(modelFile, mmprojFile)
                }
            }
        }

        // 2. Scan parent Download/ directory for any subfolder with GGUF
        val downloadParents = listOf(
            "/storage/emulated/0/Download",
            "/storage/emulated/0/Downloads",
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath
        )
        for (p in downloadParents) {
            val parentDir = File(p)
            if (parentDir.exists() && parentDir.isDirectory) {
                val subDirs = parentDir.listFiles() ?: emptyArray()
                for (sub in subDirs) {
                    if (sub.isDirectory) {
                        val subFiles = sub.listFiles() ?: emptyArray()
                        val model = subFiles.firstOrNull {
                            it.extension.lowercase() == "gguf" &&
                            !it.name.contains("mmproj", ignoreCase = true) &&
                            it.length() > 50_000_000L
                        }
                        if (model != null) {
                            val mmproj = subFiles.firstOrNull {
                                it.extension.lowercase() == "gguf" &&
                                it.name.contains("mmproj", ignoreCase = true)
                            }
                            activeModelDirectory = sub.absolutePath
                            activeModelFileName = model.name
                            return Pair(model, mmproj)
                        }
                    }
                }
            }
        }

        return null
    }

    fun isNativeGGUFAvailable(): Boolean {
        return findActiveModelFiles() != null
    }

    fun loadVLM() {
        if (nativeHandle == 0L) {
            val files = findActiveModelFiles()
            if (files != null) {
                val modelPath = files.first.absolutePath
                val mmprojPath = files.second?.absolutePath ?: ""
                Log.d("VideoRAG_VLM", "Directly initializing native VLM with files: $modelPath (mmproj: $mmprojPath)...")
                try {
                    nativeHandle = nativeInitWithFiles(modelPath, mmprojPath, 99)
                    if (nativeHandle == 0L) {
                        activeModelDirectory?.let { nativeHandle = nativeInit(it, 99) }
                    }
                    if (nativeHandle > 0L) {
                        Log.i("VideoRAG_VLM", "Native VLM successfully loaded! handle=$nativeHandle")
                    } else {
                        Log.e("VideoRAG_VLM", "Native VLM returned handle 0L.")
                    }
                } catch (e: Throwable) {
                    Log.e("VideoRAG_VLM", "JNI init failed: ${e.message}", e)
                    nativeHandle = 0L
                }
            } else {
                Log.e("VideoRAG_VLM", "No GGUF model files found in any storage candidate path.")
                nativeHandle = 0L
            }
        }
    }

    /**
     * Execute step-by-step structured forensic Chain-of-Thought reasoning over the timeline.
     */
    fun reasonOverTimeline(
        query: String,
        storyboardMoments: List<IndexedMoment>,
        topScore: Float = 0.0f
    ): String {
        if (storyboardMoments.isEmpty()) {
            return "No video storyboard keyframes available for analysis."
        }

        val sortedMoments = storyboardMoments.sortedBy { it.timestamp }
        val startMoment = sortedMoments.first()
        val endMoment = sortedMoments.last()
        val startTs = startMoment.timestamp
        val endTs = endMoment.timestamp
        val cropRegion = startMoment.cropRegion
        val storyboardPaths = sortedMoments.map { it.imagePath }

        val prompt = """
            You are an elite, highly precise on-device forensic surveillance AI.
            Analyze the chronological sequence of CCTV frames provided.
            
            User Query Target: "$query"
            • Timeline Start: $startTs
            • Timeline End: $endTs
            • Target Sector: $cropRegion
            
            Strictly structure your analysis using this format:
            🔍 FORENSIC SURVEILLANCE REPORT
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            • Target: [Describe what is being searched for]
            • Timeline: [$startTs ➔ $endTs]
            
            🎬 CHRONOLOGICAL KEYFRAME ANALYSIS:
            - [$startTs]: [Detailed situational description of what is happening in this frame, identifying colors, entities, and specific regions (e.g. top_left, right lane)]
            - [$endTs]: [Describe changes or motion relative to the previous frame]
            
            📋 FINAL VERDICT:
            [Provide a definitive, precise verification statement confirming if the query was successfully grounded in the timeline, noting direction of travel and final location.]
        """.trimIndent()

        // Ensure VLM is loaded from storage
        loadVLM()

        // EXPLICIT VERIFICATION: No silent mock fallback!
        if (nativeHandle <= 0L) {
            val primaryPath = customModelDirectory ?: activeModelDirectory ?: "/storage/emulated/0/Download/qwen2_vl_2b"
            val dir = File(primaryPath)
            
            val statusMessage: String
            if (!dir.exists()) {
                statusMessage = "Folder does not exist at: $primaryPath"
            } else {
                val files = dir.listFiles()
                if (files == null) {
                    statusMessage = "Android Scoped Storage is restricting access to this folder.\n👉 Tap '📂 Model Folder' button above to select your qwen2_vl_2b folder directly."
                } else if (files.isEmpty()) {
                    statusMessage = "Folder exists but is currently empty.\n👉 Ensure Qwen2.5-VL / Qwen2-VL .gguf and mmproj .gguf are placed inside Download/qwen2_vl_2b/."
                } else {
                    statusMessage = "Files found in folder: " + files.joinToString(", ") { "${it.name} (${it.length() / (1024 * 1024)} MB)" }
                }
            }

            return """
                ❌ Native On-Device VLM Error: Model weights not initialized!
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                
                The real Vision-Language Model (Qwen2.5-VL 3B / Qwen2-VL 2B) could not be loaded into memory.
                
                • Path Checked: $primaryPath
                • Status: $statusMessage
                
                📌 Action Required:
                1. Tap the "📂 Model Folder" button at the top to select your qwen2_vl_2b folder directly.
                2. Or place Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf and mmproj-*-F16.gguf in:
                   Internal storage/Download/qwen2_vl_2b/
                3. Re-open the app to initialize GPU/CPU tensors.
                
                (Silent heuristic mock fallback is disabled per diagnostic verification).
            """.trimIndent()
        }

        return try {
            val nativeRes = nativeGenerate(nativeHandle, prompt, storyboardPaths.toTypedArray())
            if (nativeRes.isNotEmpty() && !nativeRes.startsWith("Error")) {
                nativeRes
            } else {
                "❌ Native VLM Inference Error: $nativeRes"
            }
        } catch (e: Throwable) {
            "❌ Native VLM Execution Exception: ${e.message}"
        }
    }

    /**
     * Diagnostic report detailing active on-device VLM status.
     */
    fun getDiagnosticInfo(): String {
        return if (nativeHandle > 0L) {
            "🟢 Active: ${activeModelFileName ?: "Qwen2.5-VL / Qwen2-VL"} (Vulkan GPU, 4 Threads)"
        } else {
            "🔴 Model Missing: Sideload GGUF to Download/qwen2_vl_2b/"
        }
    }

    fun unloadVLM() {
        if (nativeHandle != 0L) {
            Log.d("VideoRAG_VLM", "Unloading VLM to free mobile system RAM...")
            try {
                nativeClose(nativeHandle)
            } catch (_: Throwable) {}
            nativeHandle = 0L
            System.gc()
        }
    }
}
