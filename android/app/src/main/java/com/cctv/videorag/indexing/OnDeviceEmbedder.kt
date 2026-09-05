package com.cctv.videorag.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.cctv.videorag.ModelPaths
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Real dual-tower MobileCLIP-S2 embedder: images and text land in ONE 512-D space.
 *
 * What this replaces
 * ------------------
 * The previous implementation had no text tower at all. `embedText` hashed character
 * trigrams into buckets and `embedCrop` (when ONNX was absent, which it always was -
 * no .onnx file existed anywhere in the repo) built a colour histogram. Those are
 * unrelated coordinate systems, so the cosine similarity between a query and a frame
 * was arithmetic without meaning. MainActivity then blended them 0.6/0.4, which mixed
 * the two incompatible spaces into a third. Search was really substring matching over
 * captions the app had generated itself.
 *
 * Why there is no fallback
 * -----------------------
 * The old code silently degraded to histograms when the model was missing, which is
 * exactly why the defect survived so long: it always returned a plausible-looking
 * vector. Missing or broken models now throw, and the UI says so.
 *
 * Small objects
 * -------------
 * A global embedding of a wide surveillance frame is dominated by road and scenery.
 * Measured on this project's own footage, a yellow bus filling ~3% of the frame left
 * all candidate frames within 0.02 cosine of each other and ranked the correct frame
 * 4th-5th. Embedding SpatialCropper's 6 regions and max-pooling moves it to 2nd with
 * the top three hits all containing the bus. Callers should embed crops, not just the
 * full frame - see MobileVectorStore's per-region entries.
 */
class OnDeviceEmbedder private constructor(
    private val env: OrtEnvironment,
    private val imageSession: OrtSession,
    private val textSession: OrtSession,
    private val tokenizer: ClipTokenizer
) {

    class ModelUnavailableException(msg: String) : Exception(msg)

    companion object {
        private const val TAG = "VideoRAG_Embedder"

        /**
         * What the labeller can name. Deliberately small and scene-appropriate: every
         * entry costs a text-tower pass once per session, and a vocabulary full of things
         * that never appear only invites false positives - CLIP always returns its best
         * match, so the guard against nonsense is the score threshold plus keeping the
         * list to things this footage plausibly contains.
         */
        private val VOCAB = listOf(
            "car", "white truck", "truck", "bus", "van", "taxi", "motorcycle", "bicycle",
            "person", "group of people", "pedestrian crossing", "traffic light",
            "road", "highway", "parking lot", "building", "tree", "sky",
            "camera crew", "film camera", "dog", "traffic sign", "number plate",
            "police car", "ambulance", "construction equipment"
        )
        const val DIM = 512
        const val IMAGE_SIZE = 256          // MobileCLIP-S2's native input resolution

        /**
         * Whether to offer the image/text graphs to the XNNPACK execution provider.
         *
         * Off, on measurement rather than on principle. The EP is present in
         * onnxruntime-android 1.17.1 and the session builds - the log says "XNNPACK" -
         * but ORT also reports "some nodes were not assigned to the preferred execution
         * providers", so only part of MobileCLIP-S2 is taken and the per-image cost on an
         * SM8550 did not move: ~0.32 s before, ~0.35 s after.
         *
         * It was also the only change present when a six-crop batch started returning a
         * single row, and a whole video indexed at 33 vectors for 33 keyframes instead of
         * 198. embedImages() now detects and survives that, but an accelerator that buys
         * nothing and defeats batching is not worth shipping on.
         *
         * Left wired up rather than deleted: flip this to true to re-measure, ideally on
         * another SoC, and watch for "batched run returned N rows" in the log. What has
         * NOT been tested is this same path with ORT's own thread count left at 4 - the
         * first attempt pinned it to 1 per ORT's guidance, which likely penalised the
         * majority of the graph that XNNPACK declined.
         */
        private const val USE_XNNPACK = false

        // fp32 first, int8 only as a fallback.
        //
        // int8 dynamic quantisation rewrites convolutions to ConvInteger, and
        // onnxruntime-android ships a reduced kernel set that has no ConvInteger
        // implementation:
        //   ORT_NOT_IMPLEMENTED - Could not find an implementation for ConvInteger(10)
        //   node '/visual/trunk/stem/stem.0/conv_scale/conv/Conv_quant'
        // MobileCLIP's image tower is convolution-heavy, so int8 is simply not loadable
        // on this runtime. fp32 costs ~400 MB across both towers but uses only Conv and
        // MatMul, which are universally supported.
        private val IMAGE_MODELS = listOf("mobileclip_image.onnx", "mobileclip_image.int8.onnx")
        private val TEXT_MODELS  = listOf("mobileclip_text.onnx",  "mobileclip_text.int8.onnx")

        // open_clip's normalisation constants for this checkpoint.
        private val MEAN = floatArrayOf(0f, 0f, 0f)
        private val STD  = floatArrayOf(1f, 1f, 1f)

        fun create(context: Context): OnDeviceEmbedder {
            val imgFile = IMAGE_MODELS.firstNotNullOfOrNull { n ->
                ModelPaths.find(context) { it.name == n }
            }
            val txtFile = TEXT_MODELS.firstNotNullOfOrNull { n ->
                ModelPaths.find(context) { it.name == n }
            }
            if (imgFile == null || txtFile == null) {
                throw ModelUnavailableException(
                    "CLIP towers not found (${IMAGE_MODELS.first()}, ${TEXT_MODELS.first()}).\n" +
                    ModelPaths.instructions(context)
                )
            }
            val env = OrtEnvironment.getEnvironment()

            // XNNPACK: available, measured, and OFF. See USE_XNNPACK.
            //
            // Try XNNPACK first, fall back to ORT's own CPU kernels.
            //
            // Ingest is dominated by this tower, not by decoding: six spatial crops per
            // keyframe at ~0.30 s each on an SM8550, so a 47-second clip is 264 forward
            // passes and a 13-minute one is 876. XNNPACK carries hand-written ARM kernels
            // for fp32 convolution that ORT's default provider does not match, and returns
            // the same numbers - which makes it the one speed-up available here that cannot
            // move retrieval quality at all.
            //
            // Failure is an expected outcome, not an exception. XNNPACK is a build flag and
            // is absent from some onnxruntime-android packages, and the int8 fallback towers
            // (ConvInteger / MatMulInteger / DynamicQuantizeLinear) are operators it does not
            // map - which is also why NNAPI stays off entirely. So a session is attempted
            // with it and retried without, and the log says which one is live.
            fun session(f: File): OrtSession {
                fun options(useXnnpack: Boolean): OrtSession.SessionOptions {
                    val o = OrtSession.SessionOptions()
                    if (useXnnpack) o.addXnnpack(mapOf("intra_op_num_threads" to "4"))
                    // 4 either way. ORT's guidance is to drop its own pool to 1 under
                    // XNNPACK, but that assumes XNNPACK takes the graph. It does not take
                    // this one - session creation logs "some nodes were not assigned to
                    // the preferred execution providers" - so the majority still runs on
                    // ORT's CPU kernels, and pinning them to one thread costs more than
                    // the offloaded nodes save.
                    o.setIntraOpNumThreads(4)
                    return o
                }
                if (USE_XNNPACK) {
                    try {
                        val sess = env.createSession(f.absolutePath, options(true))
                        Log.i(TAG, "${f.name}: XNNPACK")
                        return sess
                    } catch (e: Throwable) {
                        Log.i(TAG, "${f.name}: XNNPACK unavailable (${e.message}); default CPU kernels")
                    }
                }
                return try {
                    env.createSession(f.absolutePath, options(false))
                } catch (e: Throwable) {
                    throw ModelUnavailableException(
                        "ONNX Runtime could not open ${f.name}: ${e.message}")
                }
            }
            val s1 = session(imgFile)
            val s2 = session(txtFile)
            Log.i(TAG, "MobileCLIP towers loaded (image=${imgFile.length()/1_000_000}MB, " +
                       "text=${txtFile.length()/1_000_000}MB)")
            return OnDeviceEmbedder(env, s1, s2, ClipTokenizer.fromAssets(context))
        }

    }

    /**
     * Embed several images in ONE session run.
     *
     * Ingestion is CLIP-bound, not decode-bound. A 13-minute clip keeps ~466 keyframes
     * and embeds 6 spatial crops of each, so 2796 forward passes dominate it; replacing
     * per-frame seeking with a sequential MediaCodec pass (1b9abec) only moved the total
     * 1109 s -> 1019 s, which is what showed the decoder was never the constraint.
     *
     * The exported tower has a dynamic batch axis - pixel_values is ['batch',3,256,256] -
     * so a frame's six crops go through as one run instead of six. Identical arithmetic
     * and identical vectors; it just stops paying per-call overhead 2796 times and lets
     * the runtime spread its four threads across the batch rather than within one small
     * graph.
     */
    fun embedImages(bitmaps: List<Bitmap>): List<FloatArray> {
        if (bitmaps.isEmpty()) return emptyList()
        if (bitmaps.size == 1) return listOf(embedImage(bitmaps[0]))

        val n = bitmaps.size
        val per = 3 * IMAGE_SIZE * IMAGE_SIZE
        val all = FloatArray(n * per)
        for ((i, bmp) in bitmaps.withIndex()) {
            val scaled = Bitmap.createScaledBitmap(bmp, IMAGE_SIZE, IMAGE_SIZE, true)
            preprocess(scaled).copyInto(all, i * per)
            if (scaled != bmp) scaled.recycle()
        }
        val shape = longArrayOf(n.toLong(), 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        val rows = try {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(all), shape).use { t ->
                imageSession.run(mapOf(imageSession.inputNames.first() to t)).use { r ->
                    @Suppress("UNCHECKED_CAST")
                    (r.get(0).value as Array<FloatArray>).map { normalise(it) }
                }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "batched image run failed (${e.message}); embedding one at a time")
            emptyList()
        }

        // Never return fewer vectors than images.
        //
        // The caller zips crops against this list, and zip() truncates to the shorter of
        // the two - so a short return silently drops spatial crops instead of failing. It
        // happened: an execution provider that fixes the batch axis to 1 returned a single
        // row for a six-image batch, and a whole video indexed at one vector per keyframe
        // rather than six. Retrieval still worked, just without any of the small-object
        // recall the crops exist to provide, and nothing in the logs said so.
        if (rows.size == n) return rows
        if (rows.isNotEmpty()) {
            Log.w(TAG, "batched run returned ${rows.size} rows for $n images; falling back")
        }
        return bitmaps.map { embedImage(it) }
    }

    /** Embed one image (or crop) into a unit-length 512-D vector. */
    fun embedImage(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val chw = preprocess(scaled)
        if (scaled != bitmap) scaled.recycle()

        val shape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { t ->
            imageSession.run(mapOf(imageSession.inputNames.first() to t)).use { r ->
                return firstRow(r.get(0).value)
            }
        }
    }

    /**
     * Turn a user question into the kind of text CLIP was actually trained on.
     *
     * CLIP's text tower learns from alt-text captions - "a photo of a white delivery
     * van" - never from questions. Feeding it "what vehicles are visible" puts most of
     * the sentence's weight on interrogative words that no image caption contains, and
     * the resulting vector barely discriminates. Measured on the 13-minute clip, that
     * query scored its top five frames 0.222 / 0.219 / 0.212 / 0.210 / 0.209: a spread
     * of 0.013 across completely different scenes, so frame choice was effectively
     * arbitrary and the VLM was asked about vehicles while looking at a dog.
     *
     * Stripping the interrogative frame and wrapping the remaining content words in the
     * standard "a photo of ..." template puts the query back in caption space. This is
     * the same prompt template OpenAI used for zero-shot CLIP classification.
     */
    private fun toCaption(query: String): String {
        val stop = setOf(
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            "is", "are", "was", "were", "be", "been", "being", "am",
            "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "show", "me", "find", "any", "there", "here", "you", "see", "seen",
            "visible", "appear", "appears", "appeared", "look", "looks", "please",
            "tell", "about", "of", "in", "on", "at", "the", "a", "an", "i", "we"
        )
        val words = query.lowercase()
            .replace(Regex("[^a-z0-9 ]"), " ")
            .split(" ")
            .filter { it.isNotBlank() && it !in stop }
        // Everything was a stop word ("what is happening?") - fall back to the raw
        // question rather than embedding an empty string.
        if (words.isEmpty()) return query
        return "a photo of " + words.joinToString(" ")
    }

    /**
     * Caption variants to search with, best-of. Cheap insurance against one phrasing
     * landing badly in CLIP space.
     *
     * toCaption() keeps every content word, which is right when they all describe
     * something visible ("people wearing pink costumes" scored 0.229) and wrong when the
     * question asks about an attribute the tower cannot read. "what is written on the
     * van" became "a photo of written van": "written" contributes nothing CLIP can match
     * and dilutes the one word that matters. It scored 0.184 and was rejected as absent,
     * even though the van is plainly in the footage.
     *
     * Searching the bare subject as well and keeping the better score fixes that without
     * having to guess which words are the useful ones. Two text embeddings cost about
     * 10 ms in total, against a query that spends minutes in the vision model.
     */
    fun embedTextVariants(text: String): List<FloatArray> {
        val caption = toCaption(text)
        val out = mutableListOf(embedCaption(caption))
        val head = headNoun(text)
        if (head != null) {
            out.add(embedCaption("a photo of a $head"))
            Log.i(TAG, "query \"$text\" -> \"$caption\" | \"a photo of a $head\"")
        } else {
            Log.i(TAG, "query \"$text\" -> \"$caption\" (no head-noun variant)")
        }
        return out
    }

    /** Words that begin a trailing phrase; the subject sits before them, not after. */
    private val TRAILING_PREPS = setOf(
        "in", "on", "at", "with", "near", "under", "over", "behind", "beside",
        "by", "from", "next", "inside", "outside", "across", "beneath", "against"
    )

    private val QUESTION_OR_AUX_WORDS = setOf(
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "can", "could", "will", "would", "shall", "should",
        "show", "me", "find", "any", "there", "here", "you", "see", "seen",
        "visible", "appear", "appears", "appeared", "look", "looks", "please",
        "tell", "the", "a", "an", "i", "we", "they", "it"
    )

    /**
     * The subject of a descriptive phrase, or null when the phrase has none worth
     * searching separately.
     *
     * The variant this feeds is max-pooled into the ranking, so whatever it names decides
     * which frames the model is shown. Taking the last word - the previous rule - got that
     * wrong in both directions.
     *
     * Too short: "yellow car" reduced to "car". On traffic footage that matches every
     * frame, so retrieval ranked by "is there a car" and handed the model five frames of
     * ordinary cars for a colour that never appears. The model then obliged: a green taxi
     * was reported as "a yellow car is on the road", and two blue number plates became
     * yellow ones. A two-word query is all signal - the modifier IS the question - so
     * there is no recall to protect and no variant is added.
     *
     * Wrong end: "a red double decker bus in the snow" reduced to "snow", which the
     * comment on MIN_RELEVANCE already records as scoring above the gate against ordinary
     * daylight frames. Cutting at the first preposition gives "bus", which is the subject
     * the recall variant was always meant to find.
     */
    private fun headNoun(text: String): String? {
        val rawWords = text.lowercase()
            .replace(Regex("[^a-z0-9 ]"), " ")
            .split(Regex("""\s+"""))
            .filter { it.isNotBlank() }
        val contentWords = rawWords.filter { it !in QUESTION_OR_AUX_WORDS && it !in TRAILING_PREPS }
        val cut = rawWords.indexOfFirst { it in TRAILING_PREPS }
        if (cut > 0) {
            val prePrepWords = rawWords.subList(0, cut)
                .filter { it !in QUESTION_OR_AUX_WORDS && it !in TRAILING_PREPS }
            val head = prePrepWords.lastOrNull()
            return head?.takeIf { it.length >= 2 && it != text.lowercase().trim() }
        }
        if (contentWords.size < 3) return null
        val head = contentWords.lastOrNull()
        return head?.takeIf { it.length >= 2 && it != text.lowercase().trim() }
    }

    /** Embed a natural-language query into the SAME 512-D space. */
    fun embedText(text: String): FloatArray = embedCaption(toCaption(text))

    private fun embedCaption(caption: String): FloatArray {
        val ids = tokenizer.tokenize(caption)
        val longs = LongArray(ids.size) { ids[it].toLong() }   // towers expect int64
        val shape = longArrayOf(1, ids.size.toLong())
        OnnxTensor.createTensor(env, LongBuffer.wrap(longs), shape).use { t ->
            textSession.run(mapOf(textSession.inputNames.first() to t)).use { r ->
                return firstRow(r.get(0).value)
            }
        }
    }

    /**
     * Zero-shot object labels for a frame, from vectors that already exist.
     *
     * The expensive half of "what is in this frame" is already paid: indexing embeds six
     * crops per keyframe, and that is 99.99% of the arithmetic. Naming what it saw costs
     * one dot product per label per crop - about 245k multiply-adds for the vocabulary
     * below, against roughly 5 GFLOPs for a single image forward pass. So this is free in
     * any sense that matters, and it needs no second model, no extra download and no
     * decoding.
     *
     * The alternative was running the VLM at ingest, which the ingest path deliberately
     * avoids ("the VLM is not loaded during ingestion"): at ~8 s a frame that is minutes
     * of extra work per video, versus microseconds here.
     *
     * These are CLIP similarities, not detections. There is no box, no count, and a label
     * is only as good as the crop it came from - so treat them as retrieval hints, and do
     * not put them where the VLM will read them as fact.
     */
    fun labelsFor(vectors: List<FloatArray>, topK: Int = 3, minScore: Float = 0.20f): List<String> {
        if (vectors.isEmpty()) return emptyList()
        val vocab = labelVectors() ?: return emptyList()
        // Max-pool each label over the crops: an object filling one crop should score on
        // that crop even when it is a speck in the global view. Same reason the crops
        // exist at all.
        val best = FloatArray(VOCAB.size) { -1f }
        for (v in vectors) {
            for (i in VOCAB.indices) {
                val d = dot(v, vocab[i])
                if (d > best[i]) best[i] = d
            }
        }
        return VOCAB.indices
            .filter { best[it] >= minScore }
            .sortedByDescending { best[it] }
            .take(topK)
            .map { VOCAB[it] }
    }

    /** Label embeddings, computed once per session and then reused. */
    private var labelCache: Array<FloatArray>? = null

    private fun labelVectors(): Array<FloatArray>? {
        labelCache?.let { return it }
        return try {
            val t0 = System.currentTimeMillis()
            val v = Array(VOCAB.size) { embedCaption("a photo of a ${VOCAB[it]}") }
            Log.i(TAG, "label vocabulary embedded: ${VOCAB.size} labels in " +
                       "${System.currentTimeMillis() - t0} ms (once per session)")
            labelCache = v
            v
        } catch (e: Throwable) {
            Log.w(TAG, "could not embed label vocabulary: ${e.message}")
            null
        }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) s += a[i] * b[i]
        return s
    }

    /** CHW float tensor in [0,1], then per-channel normalisation. */
    private fun preprocess(bmp: Bitmap): FloatArray {
        val n = IMAGE_SIZE * IMAGE_SIZE
        val px = IntArray(n)
        bmp.getPixels(px, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        val out = FloatArray(3 * n)
        for (i in 0 until n) {
            val p = px[i]
            out[i]         = ((((p shr 16) and 0xFF) / 255f) - MEAN[0]) / STD[0]
            out[n + i]     = ((((p shr 8)  and 0xFF) / 255f) - MEAN[1]) / STD[1]
            out[2 * n + i] = (((p and 0xFF) / 255f) - MEAN[2]) / STD[2]
        }
        return out
    }

    @Suppress("UNCHECKED_CAST")
    private fun firstRow(raw: Any?): FloatArray {
        val v = when (raw) {
            is Array<*> -> raw[0] as FloatArray
            is FloatArray -> raw
            else -> throw IllegalStateException("unexpected ONNX output ${raw?.javaClass}")
        }
        return normalise(v)
    }

    /**
     * The graph L2-normalises already; renormalise defensively so quantisation rounding
     * cannot leave vectors slightly off the unit sphere. MobileVectorStore depends on
     * this - every stored vector measures exactly 1.000000 - so a cosine is a dot product.
     */
    private fun normalise(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        val n = sqrt(s)
        if (n <= 1e-6f) return v
        return FloatArray(v.size) { v[it] / n }
    }

    fun close() {
        try { imageSession.close() } catch (_: Exception) {}
        try { textSession.close() } catch (_: Exception) {}
    }
}
