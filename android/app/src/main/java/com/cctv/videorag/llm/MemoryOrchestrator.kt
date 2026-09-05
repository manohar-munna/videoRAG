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
        if (vlm == null) vlm = OnDeviceVLM(context, vlmModelPath)
        // Load unconditionally, not just when the instance is new. getFrameDescriber()
        // deliberately hands back an UNLOADED instance for ingestion, so by the time a
        // query arrives `vlm` is usually non-null but has no native handle. Guarding the
        // load behind `vlm == null` therefore skipped it entirely and every answer came
        // back "On-device model not loaded" while the weights were present and readable.
        // loadVLM() itself no-ops when the handle is already open, so this is idempotent.
        vlm!!.loadVLM()
        return vlm!!
    }
    
    /**
     * Free the CLIP towers while keeping the VLM and its vision-encode cache intact.
     *
     * A query needs CLIP for exactly one embedText() call, then never again - retrieval
     * is plain Kotlin over vectors already in memory. Leaving the towers resident costs
     * ~400 MB of fp32 weights (143 MB image + 254 MB text) through the whole generation
     * phase, which is precisely when memory is tightest.
     *
     * That is not a theoretical cost. Measured on the API-34 emulator (4 GB) mid-query:
     *   MemAvailable 495 MB, app TOTAL PSS 2.94 GB but RSS only 1.30 GB,
     *   SWAP PSS 1.68 GB, majflt 19,507
     * The process was faulting model weights back from swap on every generated token, so
     * latency measured swap bandwidth rather than compute.
     *
     * Deliberately NOT symmetric with the VLM: getActiveEmbedder() must stay cheap to
     * re-enter because ingestion calls it per frame. This is for the query path only,
     * after the question vector exists. Reloading the towers next query costs ~2 s.
     */
    suspend fun releaseEmbedder() = lock.withLock {
        embedder?.close()
        embedder = null
    }

    suspend fun releaseAll() = lock.withLock {
        embedder?.close()
        embedder = null
        vlm?.unloadVLM()
        vlm = null
        System.gc()
    }

    fun abortVLM() {
        vlm?.abort()
    }

    fun isVLMAborted(): Boolean = vlm?.isAborted ?: false
}
