package com.cctv.videorag.indexing

import android.graphics.Bitmap

data class CropRegion(
    val label: String,
    val bitmap: Bitmap,
    val xNorm: Float, // Normalized coordinates for boundary tracking
    val yNorm: Float,
    val wNorm: Float,
    val hNorm: Float
)

object SpatialCropper {
    /**
     * Slices high-resolution surveillance frame into a 6-region spatial pyramid:
     * 1. Global full scene
     * 2. Top-Left quadrant
     * 3. Top-Right quadrant
     * 4. Bottom-Left quadrant
     * 5. Bottom-Right quadrant
     * 6. Center focus region
     */
    fun generatePyramidCrops(source: Bitmap): List<CropRegion> {
        val w = source.width
        val h = source.height
        val cropW = (w * 0.60f).toInt().coerceAtMost(w)
        val cropH = (h * 0.60f).toInt().coerceAtMost(h)

        return listOf(
            CropRegion("global", source, 0f, 0f, 1f, 1f),
            CropRegion("top_left", Bitmap.createBitmap(source, 0, 0, cropW, cropH), 0f, 0f, 0.6f, 0.6f),
            CropRegion("top_right", Bitmap.createBitmap(source, (w - cropW).coerceAtLeast(0), 0, cropW, cropH), 0.4f, 0f, 0.6f, 0.6f),
            CropRegion("bottom_left", Bitmap.createBitmap(source, 0, (h - cropH).coerceAtLeast(0), cropW, cropH), 0f, 0.4f, 0.6f, 0.6f),
            CropRegion("bottom_right", Bitmap.createBitmap(source, (w - cropW).coerceAtLeast(0), (h - cropH).coerceAtLeast(0), cropW, cropH), 0.4f, 0.4f, 0.6f, 0.6f),
            CropRegion("center", Bitmap.createBitmap(source, ((w - cropW) / 2).coerceAtLeast(0), ((h - cropH) / 2).coerceAtLeast(0), cropW, cropH), 0.2f, 0.2f, 0.6f, 0.6f)
        )
    }
}
