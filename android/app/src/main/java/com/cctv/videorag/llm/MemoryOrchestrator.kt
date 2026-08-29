package com.cctv.videorag.llm

import android.content.Context
import com.cctv.videorag.indexing.OnDeviceEmbedder
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
     * The CLIP embedder. Does NOT unload the VLM.
     *
     * The mutual-exclusion design dates from when ingestion ran the VLM on every frame.
     * It no longer does, and every query now needs both models in sequence - embed the
     * question, retrieve, then generate - so swapping them cost a full model reload per
     * query AND discarded the native vision-encode cache each time. Measured on a Vivo
     * I2304 over three questions: the VLM context was created 3 times and released 2,
     * and the cache reported "0 hit, 5 miss" on every query because it never survived.
     *
     * Both resident is ~2.1 GB (CLIP fp32 ~0.4 GB + Qwen2-VL Q4 with a Q8 projector
     * ~1.7 GB). releaseAll() still frees everything when the screen is done with them.
     */
    suspend fun getActiveEmbedder(): OnDeviceEmbedder = lock.withLock {
        if (embedder == null) {
            // Throws ModelUnavailableException if the CLIP towers are missing; callers
            // surface that instead of silently falling back to meaningless vectors.
            embedder = OnDeviceEmbedder.create(context)
        }
        return embedder!!
    }

    /**
     * A VLM handle for frame description during ingestion, which is colour-histogram
     * work and needs no native model. Returning this without calling loadVLM() is what
     * keeps ingestion off the load/unload treadmill: previously every single frame ran
     * getActiveVLM() then getActiveEmbedder(), and the mutex unloaded one to load the
     * other each time - roughly 780 model swaps for a 13-minute clip at 1 FPS.
     */
    suspend fun getFrameDescriber(): OnDeviceVLM = lock.withLock {
        if (vlm == null) {
            vlm = OnDeviceVLM(context, vlmModelPath)   // deliberately not loaded
        }
        return vlm!!
    }

    /**
     * The loaded VLM. Keeps the embedder resident too - see getActiveEmbedder() for why
     * the previous swap-on-every-call behaviour was both slow and cache-defeating.
     */
    suspend fun getActiveVLM(): OnDeviceVLM = lock.withLock {
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
