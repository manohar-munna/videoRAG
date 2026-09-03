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
        const val DIM = 512
        const val IMAGE_SIZE = 256          // MobileCLIP-S2's native input resolution

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

            // CPU only, deliberately. These graphs are int8 dynamically quantised
            // (ConvInteger / MatMulInteger / DynamicQuantizeLinear); NNAPI has no
            // mapping for those operators, so enabling it makes session creation or the
            // first run fail rather than accelerating anything.
            fun session(f: File): OrtSession {
                val o = OrtSession.SessionOptions()
                o.setIntraOpNumThreads(4)
                return try {
                    env.createSession(f.absolutePath, o)
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
        OnnxTensor.createTensor(env, FloatBuffer.wrap(all), shape).use { t ->
            imageSession.run(mapOf(imageSession.inputNames.first() to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val rows = r.get(0).value as Array<FloatArray>
                return rows.map { normalise(it) }
            }
        }
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
        // Head-noun heuristic: English puts it last in these phrasings ("... on the VAN",
        // "... wearing pink COSTUMES"), so the trailing content word is the subject often
        // enough to be worth one extra embedding.
        val head = caption.removePrefix("a photo of ").trim().split(" ").lastOrNull()
        if (!head.isNullOrBlank() && head != caption.removePrefix("a photo of ").trim()) {
            out.add(embedCaption("a photo of a $head"))
            Log.i(TAG, "query \"$text\" -> \"$caption\" | \"a photo of a $head\"")
        } else {
            Log.i(TAG, "query \"$text\" -> \"$caption\"")
        }
        return out
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
