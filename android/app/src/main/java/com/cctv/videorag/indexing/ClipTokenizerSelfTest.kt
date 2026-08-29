package com.cctv.videorag.indexing

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Verifies the Kotlin BPE port against token ids captured from Python's open_clip.
 *
 * A tokenizer mismatch produces no exception and no visible symptom - the query
 * embedding simply lands in the wrong place and results quietly get worse. Since the
 * whole point of Phase 2 is to stop guessing about retrieval, this runs at startup in
 * debug builds and shouts if the port drifts from the reference implementation.
 */
object ClipTokenizerSelfTest {

    private const val TAG = "VideoRAG_Tokenizer"

    data class Result(val passed: Int, val failed: Int, val failures: List<String>) {
        val ok: Boolean get() = failed == 0
    }

    fun run(context: Context, tokenizer: ClipTokenizer): Result {
        var passed = 0
        var failed = 0
        val failures = mutableListOf<String>()

        try {
            val raw = context.assets.open("tokenizer_testcases.json")
                .bufferedReader().use { it.readText() }
            val cases = JSONObject(raw)
            val keys = cases.keys()
            while (keys.hasNext()) {
                val text = keys.next()
                val expectedArr: JSONArray = cases.getJSONArray(text)
                val expected = IntArray(expectedArr.length()) { expectedArr.getInt(it) }

                // compare only the non-padded prefix; padding is trivially zeros
                val actualFull = tokenizer.tokenize(text)
                val actual = actualFull.copyOfRange(0, minOf(expected.size, actualFull.size))

                if (actual.contentEquals(expected)) {
                    passed++
                } else {
                    failed++
                    val msg = "'$text'\n    expected ${expected.toList()}\n    actual   ${actual.toList()}"
                    failures.add(msg)
                    Log.e(TAG, "TOKENIZER MISMATCH: $msg")
                }
            }
        } catch (e: Exception) {
            failed++
            failures.add("self-test could not run: ${e.message}")
            Log.e(TAG, "tokenizer self-test failed to run", e)
        }

        if (failed == 0) Log.i(TAG, "tokenizer self-test: $passed/$passed cases match Python")
        else Log.e(TAG, "tokenizer self-test: $failed FAILED, $passed passed - retrieval will be wrong")
        return Result(passed, failed, failures)
    }
}
