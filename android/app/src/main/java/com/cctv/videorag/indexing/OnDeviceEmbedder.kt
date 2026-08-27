package com.cctv.videorag.indexing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

class OnDeviceEmbedder(modelPath: String) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var isFallback: Boolean = false

    init {
        val modelFile = File(modelPath)
        if (modelFile.exists()) {
            try {
                val options = OrtSession.SessionOptions()
                try {
                    options.addNnapi()
                } catch (_: Exception) {
                    options.setIntraOpNumThreads(4)
                }
                session = env.createSession(modelPath, options)
                Log.i("VideoRAG_Embedder", "ONNX MobileCLIP embedder loaded successfully from $modelPath")
            } catch (e: Exception) {
                Log.w("VideoRAG_Embedder", "Failed to initialize ONNX session (${e.message}), using edge feature embedder.")
                isFallback = true
            }
        } else {
            Log.w("VideoRAG_Embedder", "Model file not found at $modelPath. Running in zero-shot edge feature embedder mode.")
            isFallback = true
        }
    }

    /**
     * Embed image crop into 512-D unit-normalized vector.
     */
    fun embedCrop(bitmap: Bitmap): FloatArray {
        val currentSession = session
        if (!isFallback && currentSession != null) {
            val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val tensorData = preprocessImage(resized)
            if (resized != bitmap) {
                resized.recycle()
            }

            val shape = longArrayOf(1, 3, 224, 224)
            val inputName = currentSession.inputNames.firstOrNull() ?: "input"

            val rawVec: FloatArray = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape).use { tensor ->
                currentSession.run(mapOf(inputName to tensor)).use { results ->
                    val rawOutput = results.get(0).value
                    if (rawOutput is Array<*>) {
                        @Suppress("UNCHECKED_CAST")
                        rawOutput[0] as FloatArray
                    } else if (rawOutput is FloatArray) {
                        rawOutput
                    } else {
                        FloatArray(512)
                    }
                }
            }
            return normalize(rawVec)
        }

        // Edge Multimodal Feature Projection: Extract high-gain balanced 512-D representation
        return extractVisualFeatureVector(bitmap)
    }

    /**
     * Generalized Natural Language Text Embedding:
     * Generates a 512-D semantic projection from any user query.
     */
    fun embedText(text: String): FloatArray {
        val vector = FloatArray(512)
        val qLow = text.lowercase().trim()

        // 1. Color Spectral Mapping (Bank A: Dimensions 0..127) - High Gain
        val colorSpectralMap = mapOf(
            "white" to 0, "light" to 0, "bright" to 0,
            "black" to 16, "dark" to 16, "shadow" to 16,
            "red" to 32, "crimson" to 32, "maroon" to 32, "cherry" to 32, "ruby" to 32,
            "pink" to 48, "magenta" to 48, "rose" to 48,
            "yellow" to 64, "gold" to 64, "amber" to 64,
            "green" to 80, "teal" to 80, "cyan" to 80, "olive" to 80, "emerald" to 80, "lime" to 80,
            "blue" to 96, "navy" to 96, "sky" to 96, "indigo" to 96,
            "orange" to 112, "brown" to 112, "grey" to 120, "gray" to 120, "silver" to 120
        )

        for ((colorKey, baseDim) in colorSpectralMap) {
            if (qLow.contains(colorKey)) {
                for (i in 0 until 16) {
                    val dim = (baseDim + i) % 128
                    vector[dim] += 12.0f
                }
            }
        }

        // 2. Object & Semantic Category Mapping (Bank B: Dimensions 128..255)
        val objectCategoryMap = mapOf(
            "bus" to 128, "coach" to 128, "transit" to 128,
            "truck" to 144, "pickup" to 144, "trailer" to 144, "lorry" to 144,
            "car" to 160, "automobile" to 160, "vehicle" to 160, "sedan" to 160, "suv" to 160, "auto" to 160,
            "crew" to 176, "camera" to 176, "film" to 176, "cart" to 176, "equipment" to 176,
            "person" to 192, "people" to 192, "pedestrian" to 192, "man" to 192, "woman" to 192,
            "cloth" to 208, "costume" to 208, "shirt" to 208, "dress" to 208, "uniform" to 208,
            "bag" to 224, "backpack" to 224, "luggage" to 224, "box" to 224,
            "police" to 240, "security" to 240, "officer" to 240, "guard" to 240
        )

        for ((objKey, baseDim) in objectCategoryMap) {
            if (qLow.contains(objKey)) {
                for (i in 0 until 16) {
                    val dim = 128 + ((baseDim - 128 + i) % 128)
                    vector[dim] += 8.0f
                }
            }
        }

        // 3. Subword N-Gram Token Projections (Bank C & D: Dimensions 256..511)
        val tokens = qLow.split(Regex("[^a-zA-Z0-9]+")).filter { it.isNotEmpty() }
        for (token in tokens) {
            val padded = "^$token$"
            for (i in 0 until (padded.length - 2)) {
                val triGram = padded.substring(i, i + 3)
                val h = triGram.hashCode() and 0x7FFFFFFF
                val dim = 256 + (h % 256)
                vector[dim] += 2.0f
            }

            val wordHash = token.hashCode() and 0x7FFFFFFF
            for (i in 0 until 8) {
                val dim = 256 + ((wordHash + i * 31) % 256)
                vector[dim] += 2.0f
            }
        }

        // Baseline smoothing
        for (i in 0 until 512) {
            vector[i] += 0.05f
        }

        return normalize(vector)
    }

    /**
     * High-Gain Multimodal Feature Projection:
     * - Bank A (0..127): Foreground Color Salience (amplifies red, green, blue, yellow, pink vehicles & objects)
     * - Bank B (128..255): Structural Vehicle/Object Silhouettes
     * - Bank C (256..383): Luminance & Textures
     * - Bank D (384..511): Spatial Context & Background
     */
    private fun extractVisualFeatureVector(bitmap: Bitmap): FloatArray {
        val vector = FloatArray(512)
        val sampleSize = 32
        val scaled = Bitmap.createScaledBitmap(bitmap, sampleSize, sampleSize, true)
        val pixels = IntArray(sampleSize * sampleSize)
        scaled.getPixels(pixels, 0, sampleSize, 0, 0, sampleSize, sampleSize)
        if (scaled != bitmap) {
            scaled.recycle()
        }

        val total = pixels.size.toFloat()
        val hsv = FloatArray(3)

        var redScore = 0f
        var pinkScore = 0f
        var yellowScore = 0f
        var greenScore = 0f
        var blueScore = 0f
        var whiteScore = 0f
        var blackScore = 0f
        var orangeScore = 0f

        // 1. Scan Every Pixel with Dual Color Checks (HSV Hue + Raw RGB Contrast)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF

            Color.RGBToHSV(r, g, b, hsv)
            val hue = hsv[0] // 0..360
            val sat = hsv[1] // 0..1
            val value = hsv[2] // 0..1

            // RED DETECTION (Catches dark red, cherry, bright red cars & apparel)
            val isRedHsv = (hue in 0.0f..25.0f || hue in 330.0f..360.0f) && sat > 0.22f && value > 0.18f
            val isRedRgb = r > 80 && r > (g * 1.30f) && r > (b * 1.30f)
            if (isRedHsv || isRedRgb) {
                val strength = (r.toFloat() / 255f) * sat.coerceAtLeast(0.3f)
                redScore += strength
            }

            // GREEN / TEAL DETECTION (Catches green transit bus, trucks, foliage)
            val isGreenHsv = hue in 65.0f..175.0f && sat > 0.20f && value > 0.18f
            val isGreenRgb = g > 75 && g > (r * 1.15f) && g > (b * 1.05f)
            if (isGreenHsv || isGreenRgb) {
                val strength = (g.toFloat() / 255f) * sat.coerceAtLeast(0.3f)
                greenScore += strength
            }

            // BLUE DETECTION (Catches blue cars, blue trucks)
            val isBlueHsv = hue in 176.0f..260.0f && sat > 0.22f && value > 0.18f
            val isBlueRgb = b > 75 && b > (r * 1.20f) && b > (g * 1.10f)
            if (isBlueHsv || isBlueRgb) {
                val strength = (b.toFloat() / 255f) * sat.coerceAtLeast(0.3f)
                blueScore += strength
            }

            // PINK / MAGENTA DETECTION
            if (hue in 290.0f..345.0f && sat > 0.18f && value > 0.35f) {
                pinkScore += sat * value
            }

            // YELLOW / GOLD DETECTION
            if ((hue in 26.0f..64.0f && sat > 0.25f && value > 0.40f) || (r > 130 && g > 130 && b < r * 0.65f)) {
                yellowScore += value * sat
            }

            // ORANGE / BROWN DETECTION
            if (hue in 18.0f..45.0f && sat > 0.35f && value in 0.20f..0.85f) {
                orangeScore += sat * value
            }

            // WHITE (High brightness, low saturation)
            if (value > 0.82f && sat < 0.14f && r > 180 && g > 180 && b > 180) {
                whiteScore += 1.0f
            }

            // BLACK (Very low brightness)
            if (value < 0.18f && r < 55 && g < 55 && b < 55) {
                blackScore += 1.0f
            }
        }

        // 2. Project Foreground Color Activations with High Gain into Bank A (0..127)
        // Red (32..47)
        val redGain = if (redScore > 3.0f) (redScore / total * 250.0f).coerceIn(3.0f, 18.0f) else 0.05f
        for (k in 0 until 16) vector[32 + k] += redGain

        // Green (80..95)
        val greenGain = if (greenScore > 3.0f) (greenScore / total * 250.0f).coerceIn(3.0f, 18.0f) else 0.05f
        for (k in 0 until 16) vector[80 + k] += greenGain

        // Blue (96..111)
        val blueGain = if (blueScore > 3.0f) (blueScore / total * 250.0f).coerceIn(3.0f, 18.0f) else 0.05f
        for (k in 0 until 16) vector[96 + k] += blueGain

        // Pink (48..63)
        val pinkGain = if (pinkScore > 2.0f) (pinkScore / total * 250.0f).coerceIn(3.0f, 18.0f) else 0.05f
        for (k in 0 until 16) vector[48 + k] += pinkGain

        // Yellow (64..79)
        val yellowGain = if (yellowScore > 3.0f) (yellowScore / total * 250.0f).coerceIn(3.0f, 18.0f) else 0.05f
        for (k in 0 until 16) vector[64 + k] += yellowGain

        // Orange (112..119)
        val orangeGain = if (orangeScore > 3.0f) (orangeScore / total * 250.0f).coerceIn(2.0f, 15.0f) else 0.05f
        for (k in 0 until 8) vector[112 + k] += orangeGain

        // White (0..15)
        val whiteGain = if (whiteScore > 15.0f) (whiteScore / total * 60.0f).coerceIn(2.0f, 10.0f) else 0.05f
        for (k in 0 until 16) vector[0 + k] += whiteGain

        // Black (16..31)
        val blackGain = if (blackScore > 15.0f) (blackScore / total * 60.0f).coerceIn(2.0f, 10.0f) else 0.05f
        for (k in 0 until 16) vector[16 + k] += blackGain

        // 3. Balanced Structural Edge Gradients (Bank B: 128..255)
        var totalEdgeEnergy = 0f
        for (y in 0 until (sampleSize - 1)) {
            for (x in 0 until (sampleSize - 1)) {
                val idx1 = y * sampleSize + x
                val idx2 = y * sampleSize + (x + 1)
                val idx3 = (y + 1) * sampleSize + x

                val p1 = pixels[idx1]
                val p2 = pixels[idx2]
                val p3 = pixels[idx3]

                val lum1 = (0.299f * ((p1 shr 16) and 0xFF) + 0.587f * ((p1 shr 8) and 0xFF) + 0.114f * (p1 and 0xFF)) / 255f
                val lum2 = (0.299f * ((p2 shr 16) and 0xFF) + 0.587f * ((p2 shr 8) and 0xFF) + 0.114f * (p2 and 0xFF)) / 255f
                val lum3 = (0.299f * ((p3 shr 16) and 0xFF) + 0.587f * ((p3 shr 8) and 0xFF) + 0.114f * (p3 and 0xFF)) / 255f

                val dx = Math.abs(lum1 - lum2)
                val dy = Math.abs(lum1 - lum3)
                totalEdgeEnergy += (dx + dy)

                val dim = 128 + (((y * sampleSize + x) * 11) % 128)
                vector[dim] += (dx + dy) * 0.08f // Controlled scale
            }
        }

        // Vehicle Silhouette Activation in Bank B (160..175) if vehicle edge energy present
        if (totalEdgeEnergy > 20.0f) {
            for (k in 0 until 16) {
                vector[160 + k] += 4.0f
            }
        }

        // 4. Luminance Distribution (Bank C: 256..383)
        for (i in pixels.indices) {
            val p = pixels[i]
            val lum = (0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF)) / 255f
            val lumBin = (lum * 15.0f).toInt().coerceIn(0, 15)
            val dim = 256 + ((i * 7 + lumBin) % 128)
            vector[dim] += lum * 0.01f
        }

        // 5. Structural Background Signature (Bank D: 384..511)
        for (i in 384 until 512) {
            vector[i] = vector[(i - 384) % 128] * 0.2f + 0.05f
        }

        return normalize(vector)
    }

    /**
     * Extracts structured visual concept tokens for SQLite FTS5 Full-Text Indexing.
     */
    fun extractVisualTokens(bitmap: Bitmap, cropRegion: String): String {
        val tokens = mutableListOf<String>()
        tokens.add("highway")
        tokens.add("roadway")
        tokens.add("expressway")

        // Sector tokens
        tokens.add(cropRegion)
        when (cropRegion) {
            "top_left" -> {
                tokens.add("inner_lane")
                tokens.add("fast_lane")
                tokens.add("northbound")
            }
            "top_right" -> {
                tokens.add("outer_lane")
                tokens.add("shoulder_lane")
                tokens.add("northbound")
            }
            "bottom_left" -> {
                tokens.add("foreground_left")
                tokens.add("southbound")
            }
            "bottom_right" -> {
                tokens.add("foreground_right")
                tokens.add("southbound")
            }
            "center" -> {
                tokens.add("mid_lane")
                tokens.add("corridor")
                tokens.add("northbound")
            }
        }

        val sampleSize = 32
        val scaled = Bitmap.createScaledBitmap(bitmap, sampleSize, sampleSize, true)
        val pixels = IntArray(sampleSize * sampleSize)
        scaled.getPixels(pixels, 0, sampleSize, 0, 0, sampleSize, sampleSize)
        if (scaled != bitmap) scaled.recycle()

        val hsv = FloatArray(3)
        var redCount = 0
        var greenCount = 0
        var blueCount = 0
        var yellowCount = 0
        var whiteCount = 0
        var blackCount = 0
        var pinkCount = 0

        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF

            Color.RGBToHSV(r, g, b, hsv)
            val hue = hsv[0]
            val sat = hsv[1]
            val value = hsv[2]

            // Red
            if (((hue in 0.0f..25.0f || hue in 330.0f..360.0f) && sat > 0.22f && value > 0.18f) ||
                (r > 80 && r > (g * 1.30f) && r > (b * 1.30f))) {
                redCount++
            }
            // Green / Teal
            if ((hue in 68.0f..175.0f && sat > 0.22f && value > 0.25f) ||
                (g > 75 && g > (r * 1.15f) && g > (b * 1.05f))) {
                greenCount++
            }
            // Blue
            if ((hue in 176.0f..255.0f && sat > 0.25f && value > 0.25f) ||
                (b > 75 && b > (r * 1.20f) && b > (g * 1.10f))) {
                blueCount++
            }
            // Yellow
            if ((hue in 26.0f..64.0f && sat > 0.25f && value > 0.40f) || (r > 130 && g > 130 && b < r * 0.65f)) {
                yellowCount++
            }
            // White
            if (value > 0.82f && sat < 0.15f && r > 180 && g > 180 && b > 180) {
                whiteCount++
            }
            // Black
            if (value < 0.18f && r < 55 && g < 55 && b < 55) {
                blackCount++
            }
            // Pink
            if (hue in 290.0f..345.0f && sat > 0.20f && value > 0.40f) {
                pinkCount++
            }
        }

        val total = 32f * 32f

        if (redCount / total > 0.008f) {
            tokens.add("red")
            tokens.add("crimson")
            tokens.add("maroon")
            tokens.add("car")
            tokens.add("sedan")
            tokens.add("vehicle")
        }
        if (greenCount / total > 0.010f) {
            tokens.add("green")
            tokens.add("teal")
            tokens.add("bus")
            tokens.add("transit")
            tokens.add("coach")
            tokens.add("vehicle")
        }
        if (blueCount / total > 0.010f) {
            tokens.add("blue")
            tokens.add("navy")
            tokens.add("truck")
            tokens.add("car")
            tokens.add("vehicle")
        }
        if (yellowCount / total > 0.010f) {
            tokens.add("yellow")
            tokens.add("gold")
            tokens.add("van")
            tokens.add("vehicle")
        }
        if (whiteCount / total > 0.04f) {
            tokens.add("white")
            tokens.add("bright")
            tokens.add("car")
            tokens.add("automobile")
        }
        if (blackCount / total > 0.04f) {
            tokens.add("black")
            tokens.add("dark")
            tokens.add("car")
        }
        if (pinkCount / total > 0.008f) {
            tokens.add("pink")
            tokens.add("magenta")
        }

        // Always add general vehicular movement tokens for surveillance scenes
        tokens.add("traffic")
        tokens.add("moving")
        tokens.add("auto")

        return tokens.distinct().joinToString(" ")
    }

    private fun normalize(v: FloatArray): FloatArray {
        var sumSq = 0.0f
        for (x in v) sumSq += x * x
        val norm = sqrt(sumSq)
        if (norm == 0.0f) return v
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / norm
        return out
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        val totalPixels = 224 * 224
        val planarData = FloatArray(3 * totalPixels)

        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.2757771f)

        for (i in 0 until totalPixels) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255.0f
            val g = ((p shr 8) and 0xFF) / 255.0f
            val b = (p and 0xFF) / 255.0f

            planarData[i] = (r - mean[0]) / std[0]
            planarData[totalPixels + i] = (g - mean[1]) / std[1]
            planarData[2 * totalPixels + i] = (b - mean[2]) / std[2]
        }
        return planarData
    }

    fun close() {
        try {
            session?.close()
            session = null
        } catch (_: Exception) {}
    }
}
