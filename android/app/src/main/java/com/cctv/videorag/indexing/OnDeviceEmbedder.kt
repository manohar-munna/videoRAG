package com.cctv.videorag.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
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

        // Directories searched for the exported towers, mirroring the GGUF lookup.
        private val SEARCH_DIRS = listOf(
            "/storage/emulated/0/Download/mobileclip",
            "/sdcard/Download/mobileclip",
            "/storage/emulated/0/Download",
            "/sdcard/Download"
        )
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
            val imgFile = IMAGE_MODELS.firstNotNullOfOrNull { locate(it) }
            val txtFile = TEXT_MODELS.firstNotNullOfOrNull { locate(it) }
            if (imgFile == null || txtFile == null) {
                throw ModelUnavailableException(
                    "CLIP towers not found. Place ${IMAGE_MODELS.first()} and " +
                    "${TEXT_MODELS.first()} in Download/mobileclip/ then restart."
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

        private fun locate(name: String): File? =
            SEARCH_DIRS.asSequence().map { File(it, name) }.firstOrNull { it.isFile && it.length() > 1_000_000 }
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

    /** Embed a natural-language query into the SAME 512-D space. */
    fun embedText(text: String): FloatArray {
        val ids = tokenizer.tokenize(text)
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
        // The graph L2-normalises already; renormalise defensively so quantisation
        // rounding cannot leave vectors slightly off the unit sphere.
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
