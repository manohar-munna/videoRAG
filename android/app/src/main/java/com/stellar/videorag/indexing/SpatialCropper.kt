package com.stellar.videorag.indexing

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
    fun generatePyramidCrops(source: Bitmap): List<CropRegion> {
        val w = source.width
        val h = source.height
        val cropW = (w * 0.60f).toInt()
        val cropH = (h * 0.60f).toInt()

        return listOf(
            CropRegion("global", source, 0f, 0f, 1f, 1f),
            CropRegion("top_left", Bitmap.createBitmap(source, 0, 0, cropW, cropH), 0f, 0f, 0.6f, 0.6f),
            CropRegion("top_right", Bitmap.createBitmap(source, w - cropW, 0, cropW, cropH), 0.4f, 0f, 0.6f, 0.6f),
            CropRegion("bottom_left", Bitmap.createBitmap(source, 0, h - cropH, cropW, cropH), 0f, 0.4f, 0.6f, 0.6f),
            CropRegion("bottom_right", Bitmap.createBitmap(source, w - cropW, h - cropH, cropW, cropH), 0.4f, 0.4f, 0.6f, 0.6f),
            CropRegion("center", Bitmap.createBitmap(source, (w - cropW) / 2, (h - cropH) / 2, cropW, cropH), 0.2f, 0.2f, 0.6f, 0.6f)
        )
    }
}
