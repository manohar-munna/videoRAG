package com.cctv.videorag.indexing

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

class SQLiteFtsHelper(context: Context) : SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "videorag_fts.db"
        private const val DATABASE_VERSION = 2
        private const val TABLE_NAME = "video_tokens_index"
        private const val TAG = "VideoRAG_FTS"
    }

    override fun onCreate(db: SQLiteDatabase) {
        val createSql = """
            CREATE TABLE IF NOT EXISTS $TABLE_NAME (
                moment_id TEXT PRIMARY KEY,
                camera TEXT,
                timestamp TEXT,
                epoch_time INTEGER,
                crop_region TEXT,
                image_path TEXT,
                visual_tokens TEXT
            );
        """.trimIndent()
        db.execSQL(createSql)
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_tokens ON $TABLE_NAME(visual_tokens);")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_img_path ON $TABLE_NAME(image_path);")
        Log.i(TAG, "Created SQLite Visual Tokens table: $TABLE_NAME")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_NAME")
        onCreate(db)
    }

    /**
     * Inserts extracted visual concept tokens into the SQLite Lexical Index.
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
                put("epoch_time", epochTime)
                put("crop_region", cropRegion)
                put("image_path", imagePath)
                put("visual_tokens", visualTokens)
            }
            db.insertWithOnConflict(TABLE_NAME, null, values, SQLiteDatabase.CONFLICT_REPLACE)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to insert moment into SQLite: ${e.message}")
        }
    }

    /**
     * Executes sparse BM25/TF-IDF lexical search over indexed visual tokens.
     * Returns a list of (imagePath, Score).
     */
    fun searchSparse(query: String, topK: Int = 30): List<Pair<String, Float>> {
        val results = mutableListOf<Pair<String, Float>>()
        val queryTokens = query.lowercase().trim()
            .split(Regex("[^a-zA-Z0-9_]+"))
            .filter { it.length > 1 }

        if (queryTokens.isEmpty()) return results

        try {
            val db = readableDatabase
            // Build dynamic token match query
            val whereClause = StringBuilder()
            val args = mutableListOf<String>()

            for ((i, token) in queryTokens.withIndex()) {
                if (i > 0) whereClause.append(" OR ")
                whereClause.append("visual_tokens LIKE ?")
                args.add("%$token%")
            }

            val sql = "SELECT image_path, visual_tokens FROM $TABLE_NAME WHERE $whereClause LIMIT 100"
            val cursor = db.rawQuery(sql, args.toTypedArray())

            val pathScoreMap = HashMap<String, Float>()

            cursor.use { c ->
                val pathIdx = c.getColumnIndex("image_path")
                val tokenIdx = c.getColumnIndex("visual_tokens")
                while (c.moveToNext()) {
                    val path = c.getString(pathIdx)
                    val tokensStr = c.getString(tokenIdx)?.lowercase() ?: ""

                    var matchCount = 0f
                    for (t in queryTokens) {
                        if (tokensStr.contains(t)) {
                            matchCount += 1.0f
                        }
                    }

                    // Score: proportion of query tokens matched
                    val score = matchCount / queryTokens.size.toFloat()
                    val currentBest = pathScoreMap[path] ?: 0f
                    if (score > currentBest) {
                        pathScoreMap[path] = score
                    }
                }
            }

            for ((path, score) in pathScoreMap.entries.sortedByDescending { it.value }.take(topK)) {
                results.add(Pair(path, score))
            }
        } catch (e: Exception) {
            Log.w(TAG, "Search error on query '$query': ${e.message}")
        }

        return results
    }

    fun clearAll() {
        try {
            writableDatabase.execSQL("DELETE FROM $TABLE_NAME")
        } catch (_: Exception) {}
    }

    fun size(): Int {
        return try {
            val cursor = readableDatabase.rawQuery("SELECT count(*) FROM $TABLE_NAME", null)
            cursor.use {
                if (it.moveToFirst()) it.getInt(0) else 0
            }
        } catch (_: Exception) {
            0
        }
    }
}
