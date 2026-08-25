package com.stellar.videorag.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.sqrt

class OnDeviceEmbedder(modelPath: String) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions()
        try {
            // Activate hardware NPU acceleration via NNAPI execution provider
            options.addNnapi()
        } catch (_: Exception) {
            // Fallback to optimized CPU threads
            options.setIntraOpNumThreads(4)
        }
        session = env.createSession(modelPath, options)
    }

    /**
     * Embed image crop into 512-D unit-normalized vector.
     */
    fun embedCrop(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val tensorData = preprocessImage(resized)
        if (resized != bitmap) {
            resized.recycle()
        }
        
        val shape = longArrayOf(1, 3, 224, 224)
        val inputName = session.inputNames.firstOrNull() ?: "input"
        
        val rawVec: FloatArray = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val output = results.get(0).value as Array<FloatArray>
                output[0]
            }
        }
        return normalize(rawVec)
    }

    private fun normalize(v: FloatArray): FloatArray {
        var sumSq = 0.0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq)
        if (norm == 0.0f) return v
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / norm
        return out
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        val totalPixels = 224 * 224
        val planarData = FloatArray(3 * totalPixels)

        // MobileCLIP pre-processing constants
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.2757771f)

        for (i in 0 until totalPixels) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255.0f
            val g = ((p shr 8) and 0xFF) / 255.0f
            val b = (p and 0xFF) / 255.0f

            planarData[i] = (r - mean[0]) / std[0]
            planarData[totalPixels + i] = (g - mean[1]) / std[1]
            planarData[2 * totalPixels + i] = (b - mean[2]) / std[2]
        }
        return planarData
    }

    fun close() {
        try {
            session.close()
        } catch (_: Exception) {}
    }
}
