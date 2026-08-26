package com.cctv.videorag.indexing

import kotlin.math.max

data class HybridSearchResult(
    val moment: IndexedMoment,
    val rrfScore: Float,
    val denseScore: Float,
    val sparseScore: Float,
    val matchType: String
)

object HybridRetriever {

    private const val RRF_K = 60.0 // Standard constant for Reciprocal Rank Fusion

    /**
     * Merges Dense Vector Search (MobileCLIP) and Sparse Lexical Search (SQLite FTS5 BM25)
     * using Reciprocal Rank Fusion (RRF).
     *
     * RRF Formula: RRF(d) = 1.0 / (60 + Rank_Dense) + 1.0 / (60 + Rank_Sparse)
     */
    fun fuseRRF(
        denseHits: List<Pair<IndexedMoment, Float>>,
        sparseHits: List<Pair<String, Float>>,
        pathMetadata: Map<String, IndexedMoment>,
        topK: Int = 30
    ): List<HybridSearchResult> {
        val denseRankMap = HashMap<String, Int>()
        val denseScoreMap = HashMap<String, Float>()
        for ((rank, hit) in denseHits.withIndex()) {
            val path = hit.first.imagePath
            if (!denseRankMap.containsKey(path)) {
                denseRankMap[path] = rank + 1
                denseScoreMap[path] = hit.second
            }
        }

        val sparseRankMap = HashMap<String, Int>()
        val sparseScoreMap = HashMap<String, Float>()
        for ((rank, hit) in sparseHits.withIndex()) {
            val path = hit.first
            if (!sparseRankMap.containsKey(path)) {
                sparseRankMap[path] = rank + 1
                sparseScoreMap[path] = hit.second
            }
        }

        // Collect all unique candidate image paths
        val allPaths = HashSet<String>().apply {
            addAll(denseRankMap.keys)
            addAll(sparseRankMap.keys)
        }

        val fusedResults = mutableListOf<HybridSearchResult>()

        for (path in allPaths) {
            val moment = pathMetadata[path] ?: continue
            val denseRank = denseRankMap[path]
            val sparseRank = sparseRankMap[path]

            var rrfScore = 0.0
            var matchType = "Dense Vector"

            if (denseRank != null && sparseRank != null) {
                // Dual Match: Boosted by both Vector and Keyword
                rrfScore = (1.0 / (RRF_K + denseRank)) + (1.0 / (RRF_K + sparseRank))
                matchType = "Hybrid (Vector + FTS5)"
            } else if (denseRank != null) {
                rrfScore = 1.0 / (RRF_K + denseRank)
                matchType = "Dense Vector"
            } else if (sparseRank != null) {
                rrfScore = 1.0 / (RRF_K + sparseRank)
                matchType = "Sparse FTS5 Keyword"
            }

            fusedResults.add(
                HybridSearchResult(
                    moment = moment,
                    rrfScore = rrfScore.toFloat(),
                    denseScore = denseScoreMap[path] ?: 0.0f,
                    sparseScore = sparseScoreMap[path] ?: 0.0f,
                    matchType = matchType
                )
            )
        }

        return fusedResults.sortedByDescending { it.rrfScore }.take(topK)
    }
}
