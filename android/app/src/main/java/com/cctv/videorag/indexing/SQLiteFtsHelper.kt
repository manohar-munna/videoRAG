package com.cctv.videorag.indexing

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

class SQLiteFtsHelper(context: Context) : SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "videorag_fts.db"
        private const val DATABASE_VERSION = 1
        private const val TABLE_FTS = "video_fts"
        private const val TAG = "VideoRAG_FTS"
    }

    override fun onCreate(db: SQLiteDatabase) {
        val createSql = """
            CREATE VIRTUAL TABLE IF NOT EXISTS $TABLE_FTS USING fts5(
                moment_id UNINDEXED,
                camera UNINDEXED,
                timestamp UNINDEXED,
                epoch_time UNINDEXED,
                crop_region UNINDEXED,
                image_path UNINDEXED,
                visual_tokens,
                tokenize='porter unicode61'
            );
        """.trimIndent()
        db.execSQL(createSql)
        Log.i(TAG, "Created SQLite FTS5 virtual table: $TABLE_FTS")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_FTS")
        onCreate(db)
    }

    /**
     * Inserts extracted visual concept tokens into the FTS5 virtual table.
     */
    fun insertMoment(
        momentId: String,
        camera: String,
        timestamp: String,
        epochTime: Long,
        cropRegion: String,
        imagePath: String,
        visualTokens: String
    ) {
        try {
            val db = writableDatabase
            val values = ContentValues().apply {
                put("moment_id", momentId)
                put("camera", camera)
                put("timestamp", timestamp)
                put("epoch_time", epochTime.toString())
                put("crop_region", cropRegion)
                put("image_path", imagePath)
                put("visual_tokens", visualTokens)
            }
            db.insert(TABLE_FTS, null, values)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to insert moment into FTS5: ${e.message}")
        }
    }

    /**
     * Executes sparse BM25 lexical search over indexed visual tokens.
     * Returns a list of (imagePath, BM25 Score).
     */
    fun searchSparse(query: String, topK: Int = 30): List<Pair<String, Float>> {
        val results = mutableListOf<Pair<String, Float>>()
        val cleanQuery = sanitizeFtsQuery(query)
        if (cleanQuery.isEmpty()) return results

        try {
            val db = readableDatabase
            // FTS5 bm25() returns negative score where more negative = better match
            val sql = """
                SELECT image_path, bm25($TABLE_FTS) AS rank_score
                FROM $TABLE_FTS
                WHERE visual_tokens MATCH ?
                ORDER BY rank_score ASC
                LIMIT ?
            """.trimIndent()

            val cursor = db.rawQuery(sql, arrayOf(cleanQuery, topK.toString()))
            cursor.use { c ->
                val pathIdx = c.getColumnIndex("image_path")
                val rankIdx = c.getColumnIndex("rank_score")
                while (c.moveToNext()) {
                    val path = c.getString(pathIdx)
                    val rawBm25 = c.getFloat(rankIdx)
                    // Invert and normalize BM25 for positive ranking
                    val normalizedScore = (-rawBm25).coerceAtLeast(0.1f)
                    results.add(Pair(path, normalizedScore))
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "FTS5 search error on query '$cleanQuery': ${e.message}")
        }

        return results
    }

    /**
     * Formats natural language query into boolean FTS5 syntax (e.g. "red AND car").
     */
    private fun sanitizeFtsQuery(query: String): String {
        val tokens = query.lowercase().trim()
            .split(Regex("[^a-zA-Z0-9_]+"))
            .filter { it.length > 1 }

        if (tokens.isEmpty()) return ""
        // Connect terms with OR and AND matching for optimal recall & precision
        return if (tokens.size == 1) {
            tokens[0]
        } else {
            // Priority: all terms matching, or individual terms
            tokens.joinToString(" OR ")
        }
    }

    fun clearAll() {
        try {
            writableDatabase.execSQL("DELETE FROM $TABLE_FTS")
        } catch (_: Exception) {}
    }

    fun size(): Int {
        return try {
            val cursor = readableDatabase.rawQuery("SELECT count(*) FROM $TABLE_FTS", null)
            cursor.use {
                if (it.moveToFirst()) it.getInt(0) else 0
            }
        } catch (_: Exception) {
            0
        }
    }
}
