package com.cctv.videorag.indexing

import org.json.JSONObject
import kotlin.math.sqrt

data class IndexedMoment(
    val id: String,
    val camera: String,
    val timestamp: String,
    val epochTime: Long,
    val vector: FloatArray,
    val cropRegion: String,
    val imagePath: String,
    val description: String = "",
    val jsonMetadata: String = ""
) {
    fun toJsonObject(): JSONObject {
        return if (jsonMetadata.isNotEmpty()) {
            try {
                JSONObject(jsonMetadata)
            } catch (_: Exception) {
                fallbackJson()
            }
        } else {
            fallbackJson()
        }
    }

    private fun fallbackJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("camera", camera)
            put("timestamp", timestamp)
            put("description", description)
            put("image_path", imagePath)
        }
    }
}

class MobileVectorStore {
    private val registry = ArrayList<IndexedMoment>()

    val size: Int
        get() = synchronized(registry) { registry.size }

    fun addMoment(moment: IndexedMoment) {
        synchronized(registry) {
            registry.add(moment)
        }
    }

    fun getAllMoments(): List<IndexedMoment> {
        synchronized(registry) {
            return ArrayList(registry)
        }
    }

    /**
     * Scan candidate vectors and rank matches using standard Cosine Similarity.
     */
    fun search(queryVector: FloatArray, topK: Int = 10, cameraFilter: String? = null): List<Pair<IndexedMoment, Float>> {
        val scoredMatches = ArrayList<Pair<IndexedMoment, Float>>()
        synchronized(registry) {
            for (moment in registry) {
                if (cameraFilter != null && cameraFilter.isNotEmpty() && moment.camera != cameraFilter) {
                    continue
                }
                val score = cosineSimilarity(queryVector, moment.vector)
                scoredMatches.add(Pair(moment, score))
            }
        }
        return scoredMatches.sortedByDescending { it.second }.take(topK)
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0.0f
        var normA = 0.0f
        var normB = 0.0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            normA += v1[i] * v1[i]
            normB += v2[i] * v2[i]
        }
        return if (normA == 0.0f || normB == 0.0f) 0.0f else dot / (sqrt(normA) * sqrt(normB))
    }

    fun clear() {
        synchronized(registry) {
            registry.clear()
        }
    }
}
