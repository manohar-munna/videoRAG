package com.cctv.videorag.llm

import android.content.Context
import android.os.Environment
import android.util.Log
import com.cctv.videorag.indexing.IndexedMoment
import java.io.File

class OnDeviceVLM(private val context: Context, private val defaultModelDirectory: String) {
    
    // External JNI hooks into native-lib.cpp
    private external fun nativeInit(modelDir: String, layersToOffload: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, imagePaths: Array<String>): String
    private external fun nativeGetModelInfo(handle: Long): String
    private external fun nativeClose(handle: Long)

    private var nativeHandle: Long = 0
    private var activeModelDirectory: String? = null
    private var activeModelFileName: String? = null

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
     * Discovers if Qwen2.5-VL 3B or Qwen2-VL 2B GGUF model files exist in known mobile storage directories.
     */
    fun findActiveModelDir(): String? {
        val candidatePaths = listOf(
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "qwen2_vl_2b").absolutePath,
            "/storage/emulated/0/Download/qwen2_vl_2b",
            "/storage/emulated/0/Downloads/qwen2_vl_2b",
            "/sdcard/Download/qwen2_vl_2b",
            "/sdcard/Downloads/qwen2_vl_2b",
            File(Environment.getExternalStorageDirectory(), "Download/qwen2_vl_2b").absolutePath,
            File(Environment.getExternalStorageDirectory(), "Downloads/qwen2_vl_2b").absolutePath,
            File(context.filesDir, "qwen2_vl_2b").absolutePath,
            defaultModelDirectory
        )

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
                    activeModelFileName = modelFile.name
                    Log.i("VideoRAG_VLM", "Discovered VLM GGUF weights: ${modelFile.absolutePath} (${modelFile.length() / (1024 * 1024)} MB)")
                    return dir.absolutePath
                }
            }
        }
        return null
    }

    fun isNativeGGUFAvailable(): Boolean {
        return findActiveModelDir() != null
    }

    fun loadVLM() {
        if (nativeHandle == 0L) {
            val modelDir = findActiveModelDir()
            if (modelDir != null) {
                activeModelDirectory = modelDir
                Log.d("VideoRAG_VLM", "Initializing native on-device VLM (4 CPU threads, GPU offload) from $modelDir...")
                try {
                    nativeHandle = nativeInit(modelDir, 99)
                    if (nativeHandle == 0L) {
                        Log.e("VideoRAG_VLM", "Native VLM initialization returned 0L (model files missing or invalid).")
                    } else {
                        Log.i("VideoRAG_VLM", "Native VLM loaded successfully with handle=$nativeHandle")
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
     * Execute step-by-step forensic reasoning over the timeline of compiled storyboard images.
     * EXPLICIT DIAGNOSTIC: If the real VLM model is missing or failed, it returns an explicit error message.
     */
    fun reasonOverTimeline(
        query: String,
        storyboardMoments: List<IndexedMoment>,
        topScore: Float = 0.0f
    ): String {
        if (storyboardMoments.isEmpty()) {
            return "No video storyboard keyframes available for analysis."
        }

        // Chronologically sorted sequence
        val sortedMoments = storyboardMoments.sortedBy { it.timestamp }
        val anchorMoment = sortedMoments.first()
        val anchorTimestamp = anchorMoment.timestamp
        val cropRegion = anchorMoment.cropRegion
        val storyboardPaths = sortedMoments.map { it.imagePath }

        val prompt = """
            You are an on-device forensic surveillance security analyst.
            Analyze the following chronological sequence of CCTV frames.
            User Query Target: "$query". Primary Timestamp Anchor: $anchorTimestamp. Primary Region: $cropRegion.
            Describe in detail what is happening in each keyframe and across the whole scene.
            Confirm findings with exact timestamp markers: [CONFIRMED_AT: $anchorTimestamp].
        """.trimIndent()

        // Ensure VLM is loaded from storage
        loadVLM()

        // EXPLICIT VERIFICATION: No silent mock fallback!
        if (nativeHandle <= 0L) {
            val scannedDir = findActiveModelDir() ?: "/storage/emulated/0/Download/qwen2_vl_2b"
            val dir = File(scannedDir)
            val filesInDir = if (dir.exists()) {
                dir.listFiles()?.joinToString(", ") { "${it.name} (${it.length() / (1024 * 1024)} MB)" } ?: "Empty folder"
            } else {
                "Folder does not exist"
            }

            return """
                ❌ Native On-Device VLM Error: Model weights not initialized!
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                
                The real Vision-Language Model (Qwen2.5-VL 3B / Qwen2-VL 2B) could not be loaded into memory.
                
                • Expected Path: $scannedDir
                • Storage Contents: $filesInDir
                
                📌 To fix this:
                1. Place Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf (or Qwen2-VL-2B) and mmproj-*-F16.gguf into:
                   Internal storage/Download/qwen2_vl_2b/
                2. Re-open the app to initialize GPU/CPU tensors.
                
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
            "🟢 Active: ${activeModelFileName ?: "Qwen2.5-VL / Qwen2-VL"} (Native 4 Threads, GPU offload)"
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
