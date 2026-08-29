package com.cctv.videorag.ingestion

import android.graphics.Bitmap
import java.lang.Long.bitCount
import kotlin.math.abs

object MobileFrameFilter {

    /**
     * Minimum Hamming distance from the last KEPT keyframe for a frame to count as a
     * new scene. Matches the desktop pipeline's default (hash_filter.py, threshold=10)
     * and the range documented in the README for outdoor/traffic scenes.
     *
     * The gate previously dropped only at a distance of 1, i.e. only frames that were
     * near bit-identical, so on real footage it effectively never fired and every
     * sampled frame was indexed.
     */
    const val DEFAULT_HAMMING_THRESHOLD = 10

    /**
     * True if [currentHash] differs enough from the last kept keyframe to be worth
     * indexing. The first frame is always a keyframe.
     *
     * Compare against the last KEPT frame, not the immediately preceding sampled one:
     * comparing to the previous frame lets a slow pan drift arbitrarily far while each
     * step stays under threshold.
     */
    fun isKeyframe(
        lastKeptHash: Long?,
        currentHash: Long,
        threshold: Int = DEFAULT_HAMMING_THRESHOLD
    ): Boolean {
        if (lastKeptHash == null) return true
        return hammingDistance(lastKeptHash, currentHash) >= threshold
    }

    /**
     * Compute a 64-bit perceptual difference hash (dHash) of a Bitmap.
     * High-speed execution (under 0.15ms on mobile CPU).
     */
    fun calculateDHash(bitmap: Bitmap): Long {
        val width = 9
        val height = 8
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)
        if (resized != bitmap) {
            resized.recycle()
        }

        val gray = IntArray(width * height)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            // Grayscale luminosity formula
            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        }

        var hash: Long = 0
        var bitIndex = 0
        for (y in 0 until 8) {
            for (x in 0 until 8) {
                val left = gray[y * width + x]
                val right = gray[y * width + (x + 1)]
                if (left > right) {
                    hash = hash or (1L shl bitIndex)
                }
                bitIndex++
            }
        }
        return hash
    }

    /**
     * Measure visual difference via Hamming Distance (number of differing bits).
     */
    fun hammingDistance(hash1: Long, hash2: Long): Int {
        return bitCount(hash1 xor hash2)
    }

    /**
     * High-Sensitivity Sub-block Motion Detector (for highway CCTV / wide surveillance scenes):
     * Computes pixel intensity delta across downsampled grid.
     */
    fun calculateMotionEnergy(bmp1: Bitmap, bmp2: Bitmap): Float {
        val w = 16
        val h = 16
        val s1 = Bitmap.createScaledBitmap(bmp1, w, h, true)
        val s2 = Bitmap.createScaledBitmap(bmp2, w, h, true)
        val p1 = IntArray(w * h)
        val p2 = IntArray(w * h)
        s1.getPixels(p1, 0, w, 0, 0, w, h)
        s2.getPixels(p2, 0, w, 0, 0, w, h)
        if (s1 != bmp1) s1.recycle()
        if (s2 != bmp2) s2.recycle()

        var diffSum = 0f
        for (i in p1.indices) {
            val r1 = (p1[i] shr 16) and 0xFF
            val r2 = (p2[i] shr 16) and 0xFF
            val g1 = (p1[i] shr 8) and 0xFF
            val g2 = (p2[i] shr 8) and 0xFF
            val b1 = p1[i] and 0xFF
            val b2 = p2[i] and 0xFF
            val lum1 = 0.299f * r1 + 0.587f * g1 + 0.114f * b1
            val lum2 = 0.299f * r2 + 0.587f * g2 + 0.114f * b2
            diffSum += abs(lum1 - lum2)
        }
        return diffSum / (w * h * 255f)
    }

    /**
     * Format 64-bit hash as 16-character hexadecimal string.
     */
    fun formatHashHex(hash: Long): String {
        return String.format("%016X", hash)
    }
}
