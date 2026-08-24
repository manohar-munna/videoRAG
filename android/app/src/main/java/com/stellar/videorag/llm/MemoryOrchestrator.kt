package com.stellar.videorag.llm

import android.content.Context
import com.stellar.videorag.indexing.OnDeviceEmbedder
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class MemoryOrchestrator(
    private val context: Context,
    private val onnxModelPath: String,
    private val vlmModelPath: String
) {
    private val lock = Mutex()
    private var embedder: OnDeviceEmbedder? = null
    private var vlm: OnDeviceVLM? = null

    /**
     * Load the ONNX MobileCLIP model into active memory to perform live frame indexing.
     * Safely unloads the VLM if active to maintain <2.5GB RAM ceiling.
     */
    suspend fun getActiveEmbedder(): OnDeviceEmbedder = lock.withLock {
        vlm?.unloadVLM()
        vlm = null
        
        if (embedder == null) {
            embedder = OnDeviceEmbedder(onnxModelPath)
        }
        return embedder!!
    }

    /**
     * Unloads the ONNX embedder and allocates memory to load Qwen2-VL 2B
     * for query-time temporal storyboard reasoning.
     */
    suspend fun getActiveVLM(): OnDeviceVLM = lock.withLock {
        embedder?.close()
        embedder = null
        System.gc() // Reclaim heap before loading large models
        
        if (vlm == null) {
            vlm = OnDeviceVLM(context, vlmModelPath)
            vlm!!.loadVLM()
        }
        return vlm!!
    }
    
    suspend fun releaseAll() = lock.withLock {
        embedder?.close()
        embedder = null
        vlm?.unloadVLM()
        vlm = null
        System.gc()
    }
}
