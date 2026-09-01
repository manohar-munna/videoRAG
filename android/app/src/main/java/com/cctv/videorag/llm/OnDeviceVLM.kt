package com.cctv.videorag.llm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Environment
import android.util.Base64
import android.util.Log
import com.cctv.videorag.indexing.IndexedMoment
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import kotlin.math.abs

class OnDeviceVLM(private val context: Context, private val defaultModelDirectory: String) {

    // External JNI hooks into native-lib.cpp
    private external fun nativeInitWithFiles(modelPath: String, mmprojPath: String, layersToOffload: Int): Long
    private external fun nativeInit(modelDir: String, layersToOffload: Int): Long
    private external fun nativeGenerate(handle: Long, prompt: String, imagePaths: Array<String>): String
    private external fun nativeGetModelInfo(handle: Long): String
    private external fun nativeClose(handle: Long)

    private var nativeHandle: Long = 0
    private var activeModelDirectory: String? = null
    private var activeModelFileName: String? = null
    var customServerUrl: String? = null

    var customModelDirectory: String? = null
        set(value) {
            field = value
            if (nativeHandle != 0L) {
                unloadVLM()
            }
            loadVLM()
        }

    companion object {
        /**
         * How many retrieved keyframes go to the model per query.
         *
         * Each frame costs ~17 s (13.4 s encode + 3.7 s decode) at ~299 tokens on an
         * SD8Gen2, so this trades latency for recall. At 2 a query answered "no yellow
         * bus" because ranks 1-2 were 00:00:00 and 00:00:13 while the bus was at
         * 00:00:29 - correct for the frames it saw, wrong for the video. 5 gives
         * retrieval room to be imperfect without the answer being wrong.
         */
        const val MAX_FRAMES_TO_ANALYSE = 5

        init {
            try {
                System.loadLibrary("llama_jni")
            } catch (e: UnsatisfiedLinkError) {
                Log.e("VideoRAG_VLM", "CRITICAL: Native library llama_jni failed to load: ${e.message}")
            }
        }
    }

    /**
     * Discovers exact GGUF model file and mmproj projector file in mobile storage.
     */
    fun findActiveModelFiles(): Pair<File, File?>? {
        // Canonical location first: the app's own external dir needs no permission and
        // cannot be revoked. The guessed /sdcard/Download paths below only work when
        // MANAGE_EXTERNAL_STORAGE happens to be granted - it is not by default, is lost
        // on reinstall, and its absence is what produced "model not loaded" while the
        // weights sat readable on disk.
        val canonical = com.cctv.videorag.ModelPaths.modelsDir(context)
        val ggufHere = (canonical.listFiles() ?: emptyArray()).filter {
            it.isFile && it.extension.equals("gguf", true) && it.canRead()
        }
        val modelHere = ggufHere.firstOrNull {
            !it.name.contains("mmproj", true) && it.length() > 50_000_000L
        }
        if (modelHere != null) {
            activeModelDirectory = canonical.absolutePath
            activeModelFileName = modelHere.name
            return Pair(modelHere, pickProjector(ggufHere.toTypedArray()))
        }

        val candidatePaths = mutableListOf<String>()
        customModelDirectory?.let { candidatePaths.add(it) }

        candidatePaths.addAll(
            listOf(
                "/storage/emulated/0/Download/qwen2_vl_2b",
                "/storage/emulated/0/Downloads/qwen2_vl_2b",
                "/storage/emulated/0/qwen2_vl_2b",
                File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "qwen2_vl_2b").absolutePath,
                "/sdcard/Download/qwen2_vl_2b",
                "/sdcard/Downloads/qwen2_vl_2b",
                "/sdcard/qwen2_vl_2b",
                File(Environment.getExternalStorageDirectory(), "Download/qwen2_vl_2b").absolutePath,
                File(context.filesDir, "qwen2_vl_2b").absolutePath,
                File(context.getExternalFilesDir(null), "qwen2_vl_2b").absolutePath,
                defaultModelDirectory
            )
        )

        for (path in candidatePaths) {
            val dir = File(path)
            if (dir.exists() && dir.isDirectory) {
                val files = dir.listFiles() ?: emptyArray()
                val modelFile = files.firstOrNull {
                    it.extension.lowercase() == "gguf" &&
                    !it.name.contains("mmproj", ignoreCase = true) &&
                    it.length() > 50_000_000L
                }
                if (modelFile != null) {
                    val mmprojFile = pickProjector(files)
                    activeModelDirectory = dir.absolutePath
                    activeModelFileName = modelFile.name
                    return Pair(modelFile, mmprojFile)
                }
            }
        }

        val downloadParents = listOf(
            "/storage/emulated/0/Download",
            "/storage/emulated/0/Downloads",
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).absolutePath
        )
        for (p in downloadParents) {
            val parentDir = File(p)
            if (parentDir.exists() && parentDir.isDirectory) {
                val subDirs = parentDir.listFiles() ?: emptyArray()
                for (sub in subDirs) {
                    if (sub.isDirectory) {
                        val subFiles = sub.listFiles() ?: emptyArray()
                        val model = subFiles.firstOrNull {
                            it.extension.lowercase() == "gguf" &&
                            !it.name.contains("mmproj", ignoreCase = true) &&
                            it.length() > 50_000_000L
                        }
                        if (model != null) {
                            val mmproj = pickProjector(subFiles)
                            activeModelDirectory = sub.absolutePath
                            activeModelFileName = model.name
                            return Pair(model, mmproj)
                        }
                    }
                }
            }
        }

        return null
    }

    /**
     * Choose the multimodal projector, preferring a quantised one.
     *
     * Measured on a Snapdragon 8 Gen 2 (11.5 GB RAM, ~2.6 GB actually available):
     * the f16 projector is 1.33 GB and, alongside the 986 MB model, cannot stay
     * resident — it is re-read from storage every call and never converged below
     * 72 s per frame. The Q8_0 projector is 710 MB, fits, holds a steady ~20.3 s,
     * and showed no measurable loss of vision quality (it still reads fine text
     * such as vehicle livery and on-screen overlays).
     *
     * listFiles() ordering is arbitrary, so without this the f16 file wins roughly
     * half the time and performance silently collapses.
     */
    private fun pickProjector(files: Array<File>): File? {
        val projectors = files.filter {
            it.extension.lowercase() == "gguf" && it.name.contains("mmproj", ignoreCase = true)
        }
        if (projectors.isEmpty()) return null
        return projectors.firstOrNull { it.name.contains("q8", ignoreCase = true) }
            ?: projectors.firstOrNull { it.name.contains("q4", ignoreCase = true) }
            ?: projectors.minByOrNull { it.length() }   // fall back to the smallest
    }

    fun isNativeGGUFAvailable(): Boolean {
        return findActiveModelFiles() != null
    }

    fun loadVLM() {
        if (nativeHandle == 0L) {
            val files = findActiveModelFiles()
            if (files != null) {
                val modelPath = files.first.absolutePath
                val mmprojPath = files.second?.absolutePath ?: ""
                Log.d("VideoRAG_VLM", "Initializing native VLM with files: $modelPath (mmproj: $mmprojPath)...")
                try {
                    nativeHandle = nativeInitWithFiles(modelPath, mmprojPath, 99)
                    if (nativeHandle == 0L) {
                        activeModelDirectory?.let { nativeHandle = nativeInit(it, 99) }
                    }
                    if (nativeHandle != 0L) {
                        Log.i("VideoRAG_VLM", "Native VLM successfully loaded! handle=$nativeHandle")
                    } else {
                        Log.e("VideoRAG_VLM", "Native VLM returned handle 0L.")
                    }
                } catch (e: Throwable) {
                    Log.e("VideoRAG_VLM", "JNI init failed: ${e.message}", e)
                    nativeHandle = 0L
                }
            } else {
                Log.e("VideoRAG_VLM", "No GGUF model files found in any storage candidate path.")
                nativeHandle = 0L
            }
        }
    }

    /**
     * INGESTION STEP: Real on-device visual analysis of the keyframe bitmap.
     * Extracts actual colors, sectors, and entities present in the frame.
     */
    /**
     * INGESTION STEP: record what can be measured cheaply about a keyframe.
     *
     * Deliberately does NOT run the VLM. Encoding one frame costs ~20 s on a
     * Snapdragon 8 Gen 2, so captioning during ingestion would take hours for a
     * short clip. Frames are captioned lazily at query time instead, over only the
     * handful that retrieval actually surfaces.
     *
     * This previously asserted object categories from colour thresholds alone —
     * a yellow pixel cluster became a "transit_bus" in the "inner_left_lane" — and
     * those inventions then fed both the search index and the answer text. It now
     * records only what a histogram can honestly support: colours and lighting.
     */
    /**
     * Normalised rect for a SpatialCropper region label, as (x, y, w, h) fractions.
     *
     * Mirrors SpatialCropper.generatePyramidCrops() exactly - it builds each crop at 60%
     * of the frame in both axes and anchors it to a corner, the centre, or the whole
     * frame. Kept as a pure function of the stored label so no re-ingest is needed; the
     * geometry was always implied by crop_region, it simply was never used at query time.
     */
    private fun regionRect(label: String): FloatArray = when (label) {
        "top_left"     -> floatArrayOf(0.0f, 0.0f, 0.6f, 0.6f)
        "top_right"    -> floatArrayOf(0.4f, 0.0f, 0.6f, 0.6f)
        "bottom_left"  -> floatArrayOf(0.0f, 0.4f, 0.6f, 0.6f)
        "bottom_right" -> floatArrayOf(0.4f, 0.4f, 0.6f, 0.6f)
        "center"       -> floatArrayOf(0.2f, 0.2f, 0.6f, 0.6f)
        else           -> floatArrayOf(0.0f, 0.0f, 1.0f, 1.0f)   // "global"
    }

    /**
     * Path to an image showing [moment]'s matched region at the frame's full size.
     *
     * Returns the original frame untouched for "global" hits and whenever anything fails
     * to decode - a slightly worse framing is always preferable to dropping the evidence.
     */
    fun regionCropPath(moment: IndexedMoment): String {
        val src = moment.imagePath
        val r = regionRect(moment.cropRegion)
        if (r[2] >= 1f && r[3] >= 1f) return src
        return try {
            val dirEarly = File(context.cacheDir, "query_crops")
            val cached = File(dirEarly, "${moment.id.replace(Regex("[^A-Za-z0-9_]"), "_")}.jpg")
            // The chat strip re-requests the same crops to show the user exactly what the
            // model was given, so serve the existing file rather than decoding twice.
            if (cached.isFile && cached.length() > 0L) return cached.absolutePath
            val full = BitmapFactory.decodeFile(src) ?: return src
            val x = (full.width  * r[0]).toInt().coerceIn(0, full.width  - 1)
            val y = (full.height * r[1]).toInt().coerceIn(0, full.height - 1)
            val w = (full.width  * r[2]).toInt().coerceAtLeast(1).coerceAtMost(full.width  - x)
            val h = (full.height * r[3]).toInt().coerceAtLeast(1).coerceAtMost(full.height - y)
            val crop = Bitmap.createBitmap(full, x, y, w, h)
            // Back to the frame's own dimensions so the vision encoder is handed exactly
            // as many pixels - and so exactly as many tokens - as it was before.
            val scaled = Bitmap.createScaledBitmap(crop, full.width, full.height, true)
            val dir = File(context.cacheDir, "query_crops").apply { mkdirs() }
            val out = File(dir, "${moment.id.replace(Regex("[^A-Za-z0-9_]"), "_")}.jpg")
            java.io.FileOutputStream(out).use { scaled.compress(Bitmap.CompressFormat.JPEG, 90, it) }
            if (crop != full) crop.recycle()
            if (scaled != crop) scaled.recycle()
            full.recycle()
            Log.i("VideoRAG_VLM", "crop ${moment.timestamp}[${moment.cropRegion}] -> ${out.name}")
            out.absolutePath
        } catch (e: Throwable) {
            Log.w("VideoRAG_VLM", "region crop failed for $src: ${e.message}")
            src
        }
    }

    fun describeFrameAsJson(bitmap: Bitmap, timestamp: String, frameIndex: Int, imagePath: String): JSONObject {
        val stats = analyzeFramePixels(bitmap)

        val colorsArray = JSONArray()
        for (c in stats.dominantColors) colorsArray.put(c)

        val colorText = if (stats.dominantColors.isEmpty()) "no dominant colour"
                        else stats.dominantColors.joinToString(", ")

        return JSONObject().apply {
            put("frame_index", frameIndex)
            put("timestamp", timestamp)
            put("image_path", imagePath)
            put("dominant_colors", colorsArray)
            put("lighting", stats.brightnessCategory)
            put("analysis", "colour histogram only; not yet inspected by the vision model")
            put("visual_description",
                "Keyframe at $timestamp. Dominant colours: $colorText. " +
                "Lighting: ${stats.brightnessCategory}.")
        }
    }

    /**
     * Backward compatibility helper returning text string.
     */
    fun describeFrame(bitmap: Bitmap, timestamp: String): String {
        val json = describeFrameAsJson(bitmap, timestamp, 1, "")
        return json.getString("visual_description")
    }

    /**
     * QUERY STEP: Evaluates the user query against the retrieved keyframe evidence.
     * Produces a truthful, evidence-grounded response (Positive Match vs Negative Not-Found).
     */
    /**
     * QUERY STEP: send the retrieved keyframe images and the user's question to the
     * on-device VLM and return what it actually reports.
     *
     * This previously ran a lexical gate over the generated descriptions and, when no
     * query token matched, returned a hardcoded "NEGATIVE FINDING" block asserting the
     * footage showed a "highway traffic corridor consisting of transit buses" — for any
     * video, without consulting the model. The VLM is now always asked, and it is given
     * the frames themselves rather than a summary of them, so the answer is grounded in
     * pixels instead of in our own generated text.
     */
    /**
     * QUERY STEP: send the retrieved keyframe images and the question to the
     * on-device VLM and return what it actually reports.
     *
     * This previously ran a lexical gate over our own generated descriptions and,
     * when no query token matched, returned a hardcoded "NEGATIVE FINDING" block
     * asserting the footage showed a "highway traffic corridor consisting of transit
     * buses" - for any video, without ever consulting the model. The VLM is now always
     * asked, and is given the frames themselves rather than a summary of them, so the
     * answer is grounded in pixels instead of in text we generated.
     */
    fun answerFromRetrievedContext(
        query: String,
        top5Moments: List<IndexedMoment>,
        history: List<ConversationTurn> = emptyList()
    ): String {
        if (top5Moments.isEmpty()) {
            return "No indexed keyframes to search. Ingest a video first."
        }

        // Send only the few best-ranked frames. Encode cost is per frame and dominates
        // the query: at ~20 s/frame, five frames is a 100 s wait before the first token,
        // and at the 720p frames the decoder used to store it was ~87 s each. The list
        // arrives ranked, so take the top MAX_FRAMES_TO_ANALYSE and then order them in
        // time for the model.
        val ranked = top5Moments.take(MAX_FRAMES_TO_ANALYSE)
        val sorted = ranked.sortedBy { it.timestamp }
        val startTs = sorted.first().timestamp
        val endTs = sorted.last().timestamp
        // Send the region that actually matched, not the whole frame.
        //
        // Every frame is indexed as six spatial regions and retrieval reports which one
        // won. Measured over this project's own index, the winner is a sub-region rather
        // than `global` in 83% of hits, and the crop vectors are far from the frame's own
        // vector - mean cosine 0.766, minimum 0.544 - so that choice carries real
        // information about where the subject is. Passing imagePath threw it away and
        // handed the model the entire scene every time.
        //
        // Cropping to the winning region and rescaling back to the frame's own size keeps
        // the token count identical, because token count follows pixel dimensions. What
        // changes is how much of that budget lands on the subject: a 60%x60% region is 36%
        // of the frame's area, so the subject occupies ~2.8x the pixels it did before.
        // On a 299-token frame that is roughly 9 tokens of subject becoming ~25.
        //
        // This is the grounding problem that image_min_tokens=1024 was meant to solve, at
        // no latency cost - that route measured 3 min -> 15 min per question and was
        // reverted in ca58db1.
        val imagePaths = sorted.filter { File(it.imagePath).exists() }
                               .map { regionCropPath(it) }

        if (nativeHandle == 0L) {
            return "On-device model not loaded, so no visual analysis was performed. " +
                   "Retrieved ${sorted.size} keyframes spanning [$startTs - $endTs].\n\n" +
                   com.cctv.videorag.ModelPaths.instructions(context)
        }
        if (imagePaths.isEmpty()) {
            return "Retrieved ${sorted.size} keyframes for [$startTs - $endTs], but their " +
                   "image files are missing from storage, so they could not be analysed."
        }

        // One media marker per frame; mtmd substitutes the encoded image at each
        // marker, keeping frames in chronological order inside the prompt.
        val marker = "<__media__>"
        val frameList = sorted
            .filter { File(it.imagePath).exists() }
            .joinToString(separator = "\n") { "Frame at ${it.timestamp}:\n" + marker }

        // Qwen2-VL has native visual grounding and will answer a bare noun phrase like
        // "yellow bus" in detection mode, emitting <|object_ref_start|>...<|box_start|>
        // (606,182),(709,325)<|box_end|> instead of prose. Correct, but not an answer a
        // person can read. Frame the task as prose Q&A and rule coordinates out explicitly.
        val shown = sorted.filter { File(it.imagePath).exists() }
        val frameCount = shown.size
        // Prefill the assistant turn with the first timestamp so the model continues an
        // established pattern instead of choosing a format. Instructions alone produced,
        // in turn: a copied worked example, literal <placeholders>, "Answer:" on every
        // line, and finally a two-word reply with no timestamps at all.
        val firstStamp = shown.firstOrNull()?.timestamp ?: "00:00:00"

        // The frames are retrieval hits, not consecutive video: they can be minutes
        // apart. Without saying so the model narrates them as continuous motion and
        // invents movement between samples that were never observed - a real hazard when
        // the output is meant to be surveillance evidence.
        val system = "You are a CCTV analyst. You are shown a few still keyframes taken " +
                     "from a longer recording. They are in time order but NOT consecutive - " +
                     "minutes may pass between them, so never describe motion or travel " +
                     "between two frames, only what each one shows. Report by subject: name " +
                     "a thing once and list every timestamp it appears at. Answer only from " +
                     "what is visible; if the subject is not there, say so plainly rather " +
                     "than guessing. Reply in plain English sentences. Never output bounding " +
                     "boxes, coordinates, or object-reference tags."

        // Replay recent turns so follow-ups resolve against what was already said.
        // Answers are truncated: the point is to carry the established facts, not to
        // spend the context window re-reading full prose.
        val priorContext = if (history.isEmpty()) "" else buildString {
            append("\nEarlier in this conversation:\n")
            for (t in history) {
                append("Q: ").append(t.question).append("\n")
                append("A: ").append(t.answer.substringBefore("\n---").trim().take(220)).append("\n")
            }
        }

        val prompt = """<|im_start|>system
$system<|im_end|>
<|im_start|>user
These are keyframes from a surveillance video, in time order.

$frameList
$priorContext
Question: $query

Write one line for each of the $frameCount frames, starting with its timestamp, describing what that frame shows in relation to the question. Note clothing colour, vehicle markings and any text you can read. Say plainly when the thing asked about is not in a frame.<|im_end|>
<|im_start|>assistant
$firstStamp -"""

        val answer = try {
            nativeGenerate(nativeHandle, prompt, imagePaths.toTypedArray())
        } catch (e: Throwable) {
            Log.e("VideoRAG_VLM", "nativeGenerate failed: ${e.message}", e)
            return "Visual analysis failed: ${e.message ?: e.javaClass.simpleName}"
        }

        if (answer.isBlank() || answer.startsWith("Error")) {
            return "The on-device model returned no usable output. $answer".trim()
        }

        val footer = "Analysed ${imagePaths.size} keyframes spanning [$startTs - $endTs] " +
                     "using ${activeModelFileName ?: "the on-device model"}."
        // the prefilled "HH:MM:SS -" is part of the answer, so put it back on the front
        val full = "$firstStamp -" + answer.trimEnd()
        return groupBySubject(humanise(full)) + "\n\n---\n" + footer
    }
    /**
     * Collapse repeated per-frame descriptions into one line per subject.
     *
     * Asking the model to group timestamps itself failed repeatedly: given a worked
     * example it copied the example, given placeholders it emitted the placeholders,
     * and given a two-step "describe then answer" it prefixed every line with "Answer:".
     * A 2B model reliably does one thing here - describe a single image - so the model
     * describes each frame and the aggregation happens in code, where it is exact.
     *
     * "00:03:42 - White truck parked by a crowd."
     * "00:07:20 - White truck parked by a crowd."   ->  one line, both timestamps
     */
    private fun groupBySubject(text: String): String {
        val lineRx = Regex("""^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–:]\s*(.+?)\s*$""")
        val order = LinkedHashMap<String, MutableList<String>>()   // normalised -> timestamps
        val display = HashMap<String, String>()                    // normalised -> first wording
        val passthrough = mutableListOf<String>()

        for (raw in text.lines()) {
            val m = lineRx.find(raw)
            if (m == null) {
                if (raw.isNotBlank()) passthrough.add(raw.trim())
                continue
            }
            val ts = m.groupValues[1]
            val desc = m.groupValues[2].trim().trimEnd('.')
            val key = desc.lowercase().replace(Regex("[^a-z0-9 ]"), "").replace(Regex("\\s+"), " ")
            if (key.isEmpty()) continue
            order.getOrPut(key) { mutableListOf() }.add(ts)
            display.putIfAbsent(key, desc)
        }
        if (order.isEmpty()) return text

        val out = StringBuilder()
        for ((key, times) in order) {
            val what = display[key] ?: continue
            out.append(what.replaceFirstChar { it.uppercase() })
            out.append(if (times.size == 1) " at ${times[0]}." else " at ${times.dropLast(1).joinToString(", ")} and ${times.last()}.")
            out.append('\n')
        }
        for (p in passthrough) out.append(p).append('\n')
        return out.toString().trim()
    }

    /**
     * Turn Qwen2-VL grounding output into a readable sentence.
     *
     * Even when asked for prose the model sometimes answers a short noun-phrase query in
     * detection mode:
     *     <|object_ref_start|>yellow bus<|object_ref_end|><|box_start|>(606,182),(709,325)<|box_end|>
     * That is a correct, useful answer - the box is in normalised 0-1000 coordinates and
     * did land on the bus - so translate it rather than showing raw tags or discarding it.
     */
    private fun humanise(raw: String): String {
        val refRx = Regex("""<\|object_ref_start\|>(.*?)<\|object_ref_end\|>""")
        val boxRx = Regex("""<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>""")
        if (!refRx.containsMatchIn(raw) && !boxRx.containsMatchIn(raw)) return raw

        val labels = refRx.findAll(raw).map { it.groupValues[1].trim() }.filter { it.isNotEmpty() }.toList()
        val boxes = boxRx.findAll(raw).map {
            val (x1, y1, x2, y2) = it.destructured
            // 0-1000 normalised -> percentage of frame, plus a rough position in words
            val cx = (x1.toInt() + x2.toInt()) / 2
            val cy = (y1.toInt() + y2.toInt()) / 2
            val h = when { cx < 333 -> "left"; cx < 667 -> "centre"; else -> "right" }
            val v = when { cy < 333 -> "upper"; cy < 667 -> "middle"; else -> "lower" }
            "$v $h of the frame"
        }.toList()

        val subject = labels.firstOrNull() ?: "the subject"
        val where = boxes.firstOrNull()
        val sb = StringBuilder()
        sb.append(if (where != null) "Yes - $subject is visible in the $where."
                  else "Yes - $subject is visible.")
        if (labels.size > 1) sb.append(" Also detected: ${labels.drop(1).joinToString(", ")}.")

        // keep any prose the model produced alongside the tags
        val leftover = raw.replace(refRx, "").replace(boxRx, "").trim()
        if (leftover.isNotEmpty()) sb.append("\n\n").append(leftover)
        return sb.toString()
    }

    private data class FramePixelStats(
        val dominantColors: List<String>,
        val brightnessCategory: String,
        val hasYellow: Boolean,
        val yellowInLowerLeft: Boolean,
        val hasRed: Boolean,
        val hasBlue: Boolean,
        val hasGreen: Boolean,
        val hasWhite: Boolean,
        val hasDark: Boolean
    )

    private fun analyzeFramePixels(bmp: Bitmap): FramePixelStats {
        val stepX = maxOf(1, bmp.width / 16)
        val stepY = maxOf(1, bmp.height / 16)

        var totalR = 0L
        var totalG = 0L
        var totalB = 0L
        var sampleCount = 0

        var yellowCount = 0
        var yellowLowerLeftCount = 0
        var redCount = 0
        var greenCount = 0
        var blueCount = 0
        var whiteCount = 0
        var darkCount = 0

        val midY = bmp.height / 2
        val midX = bmp.width / 2

        for (y in 0 until bmp.height step stepY) {
            for (x in 0 until bmp.width step stepX) {
                val pixel = bmp.getPixel(x, y)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                totalR += r
                totalG += g
                totalB += b
                sampleCount++

                val hsv = FloatArray(3)
                Color.RGBToHSV(r, g, b, hsv)
                val hue = hsv[0]
                val sat = hsv[1]
                val value = hsv[2]

                // Yellow / Amber
                if ((hue in 26.0f..65.0f && sat > 0.22f && value > 0.35f) || (r > 130 && g > 130 && b < r * 0.70f)) {
                    yellowCount++
                    if (y > midY && x < midX) {
                        yellowLowerLeftCount++
                    }
                }
                // Red / Crimson
                if (((hue in 0.0f..25.0f || hue in 330.0f..360.0f) && sat > 0.22f && value > 0.20f) || (r > 90 && r > g * 1.3f && r > b * 1.3f)) {
                    redCount++
                }
                // Green / Teal
                if ((hue in 68.0f..175.0f && sat > 0.20f && value > 0.25f) || (g > 80 && g > r * 1.15f && g > b * 1.05f)) {
                    greenCount++
                }
                // Blue / Navy
                if ((hue in 176.0f..255.0f && sat > 0.25f && value > 0.25f) || (b > 80 && b > r * 1.20f && b > g * 1.10f)) {
                    blueCount++
                }
                // White / Silver
                if (value > 0.80f && sat < 0.15f) {
                    whiteCount++
                }
                // Dark / Black
                if (value < 0.20f) {
                    darkCount++
                }
            }
        }

        val colors = mutableListOf<String>()
        val total = maxOf(1, sampleCount).toFloat()

        val hasYellow = (yellowCount / total > 0.008f)
        val yellowInLowerLeft = (yellowLowerLeftCount / total > 0.004f)
        val hasRed = (redCount / total > 0.008f)
        val hasGreen = (greenCount / total > 0.010f)
        val hasBlue = (blueCount / total > 0.010f)
        val hasWhite = (whiteCount / total > 0.04f)
        val hasDark = (darkCount / total > 0.04f)

        if (hasYellow) colors.add("Yellow")
        if (hasBlue) colors.add("Blue")
        if (hasRed) colors.add("Red")
        if (hasGreen) colors.add("Green")
        if (hasWhite) colors.add("White")
        if (hasDark) colors.add("Black")

        val avgLuminance = if (sampleCount > 0) ((totalR + totalG + totalB) / (3 * sampleCount)).toInt() else 128
        val lighting = when {
            avgLuminance > 160 -> "High Key Daylight"
            avgLuminance < 80 -> "Low Light Corridor"
            else -> "Balanced Surveillance Lighting"
        }

        return FramePixelStats(
            dominantColors = colors,
            brightnessCategory = lighting,
            hasYellow = hasYellow,
            yellowInLowerLeft = yellowInLowerLeft,
            hasRed = hasRed,
            hasBlue = hasBlue,
            hasGreen = hasGreen,
            hasWhite = hasWhite,
            hasDark = hasDark
        )
    }

    fun getDiagnosticInfo(): String {
        return if (nativeHandle != 0L) {
            // Reports the real state: the build is CPU-only, GGML_VULKAN is not enabled.
            "Loaded: ${activeModelFileName ?: "on-device model"} (CPU, 5 threads)"
        } else {
            "No model loaded - place GGUF files in Download/qwen2_vl_2b"
        }
    }

    fun unloadVLM() {
        if (nativeHandle != 0L) {
            Log.d("VideoRAG_VLM", "Unloading VLM to free mobile system RAM...")
            try {
                nativeClose(nativeHandle)
            } catch (_: Throwable) {}
            nativeHandle = 0L
            System.gc()
        }
    }
}
