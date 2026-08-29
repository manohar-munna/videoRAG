package com.cctv.videorag.indexing

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader

/**
 * CLIP byte-pair-encoding tokenizer, ported to match open_clip exactly.
 *
 * The text tower is trained on these exact token ids. If tokenization differs even
 * slightly the embedding lands somewhere unrelated in the 512-D space and retrieval
 * silently degrades - no exception, no log, just wrong results. That failure mode is
 * why ClipTokenizerTest checks against ids captured from the Python tokenizer rather
 * than trusting this by inspection.
 *
 * Pipeline (mirrors open_clip's SimpleTokenizer):
 *   clean/lowercase -> regex split -> UTF-8 bytes -> byte->unicode map
 *   -> BPE merges -> vocab lookup -> [SOT] ids [EOT] -> zero-pad to 77
 */
class ClipTokenizer(
    private val encoder: Map<String, Int>,
    private val bpeRanks: Map<Pair<String, String>, Int>
) {

    companion object {
        const val CONTEXT_LENGTH = 77
        private const val SOT = "<|startoftext|>"
        private const val EOT = "<|endoftext|>"
        private const val TAG = "VideoRAG_Tokenizer"

        // open_clip's pattern. Order matters: the contraction alternatives must precede
        // the letter class so "'t" is not swallowed as a letter run.
        private val PATTERN = Regex(
            """<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+""",
            RegexOption.IGNORE_CASE
        )
        private val WHITESPACE = Regex("""\s+""")

        /**
         * GPT-2 style reversible byte<->unicode map. Bytes that are not printable ASCII
         * are shifted into a private range so every byte becomes a single char that BPE
         * can operate on without ever seeing a raw control byte.
         */
        private val byteEncoder: Map<Int, Char> = buildMap {
            val bs = ArrayList<Int>()
            (33..126).forEach { bs.add(it) }
            (161..172).forEach { bs.add(it) }
            (174..255).forEach { bs.add(it) }
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0..255) {
                if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
            }
            for (i in bs.indices) put(bs[i], cs[i].toChar())
        }

        /** Load vocab + merges from APK assets (they are small and must never be missing). */
        fun fromAssets(
            context: Context,
            vocabAsset: String = "clip_vocab.json",
            mergesAsset: String = "clip_merges.txt"
        ): ClipTokenizer {
            val vocabJson = context.assets.open(vocabAsset).bufferedReader().use(BufferedReader::readText)
            val obj = JSONObject(vocabJson)
            val enc = HashMap<String, Int>(obj.length())
            val keys = obj.keys()
            while (keys.hasNext()) { val k = keys.next(); enc[k] = obj.getInt(k) }

            val ranks = HashMap<Pair<String, String>, Int>()
            context.assets.open(mergesAsset).bufferedReader().useLines { lines ->
                var rank = 0
                for (line in lines) {
                    val t = line.trim()
                    if (t.isEmpty() || t.startsWith("#version")) continue
                    val parts = t.split(' ')
                    if (parts.size == 2) ranks[parts[0] to parts[1]] = rank++
                }
            }
            Log.i(TAG, "CLIP tokenizer loaded: ${enc.size} vocab, ${ranks.size} merges")
            return ClipTokenizer(enc, ranks)
        }
    }

    private val bpeCache = HashMap<String, String>()

    /** Encode to exactly [CONTEXT_LENGTH] ids: [SOT] tokens [EOT] then zero padding. */
    fun tokenize(text: String): IntArray {
        val ids = IntArray(CONTEXT_LENGTH)                       // zero = pad
        val out = ArrayList<Int>(CONTEXT_LENGTH)
        out.add(encoder[SOT] ?: 49406)

        val cleaned = WHITESPACE.replace(text.trim(), " ").lowercase()
        for (m in PATTERN.findAll(cleaned)) {
            // to bytes, then into the reversible byte alphabet
            val mapped = buildString {
                for (b in m.value.toByteArray(Charsets.UTF_8)) {
                    append(byteEncoder[b.toInt() and 0xFF] ?: '?')
                }
            }
            for (piece in bpe(mapped).split(' ')) {
                if (piece.isEmpty()) continue
                val id = encoder[piece]
                if (id != null) out.add(id)
                else Log.w(TAG, "OOV bpe piece '$piece' - dropped")
                if (out.size >= CONTEXT_LENGTH - 1) break
            }
            if (out.size >= CONTEXT_LENGTH - 1) break
        }

        out.add(encoder[EOT] ?: 49407)
        for (i in out.indices) { if (i < CONTEXT_LENGTH) ids[i] = out[i] }
        return ids
    }

    /** Greedy BPE: repeatedly merge the adjacent pair with the lowest merge rank. */
    private fun bpe(token: String): String {
        bpeCache[token]?.let { return it }
        if (token.isEmpty()) return token

        // last symbol carries the end-of-word marker
        var word = ArrayList<String>(token.length)
        for (i in token.indices) {
            word.add(if (i == token.length - 1) token[i] + "</w>" else token[i].toString())
        }
        if (word.size == 1) {
            val r = word[0]; bpeCache[token] = r; return r
        }

        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until word.size - 1) {
                val r = bpeRanks[word[i] to word[i + 1]] ?: continue
                if (r < bestRank) { bestRank = r; bestIdx = i }
            }
            if (bestIdx < 0) break

            val merged = ArrayList<String>(word.size - 1)
            var i = 0
            while (i < word.size) {
                if (i == bestIdx) { merged.add(word[i] + word[i + 1]); i += 2 }
                else { merged.add(word[i]); i++ }
            }
            word = merged
            if (word.size == 1) break
        }

        val result = word.joinToString(" ")
        bpeCache[token] = result
        return result
    }
}
