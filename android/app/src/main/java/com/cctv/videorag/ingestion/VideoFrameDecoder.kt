package com.cctv.videorag.ingestion

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

class VideoFrameDecoder(private val context: Context) {

    /**
     * Decode video from Uri (local file / content picker) into downsampled keyframes at specified target frame rate.
     */
    suspend fun decodeVideoUri(
        videoUri: Uri,
        cameraName: String,
        sampleFps: Double = 0.5,
        onProgress: (currentSec: Long, totalSec: Long, frameIndex: Int) -> Unit,
        onKeyframeDecoded: suspend (Bitmap, String, Long, String) -> Unit
    ) = withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val totalSec = durationMs / 1000L
            val intervalMs = (1000.0 / sampleFps).toLong().coerceAtLeast(300L)

            val frameOutputDir = File(context.filesDir, "extracted_frames/$cameraName").apply { mkdirs() }

            var curTimeMs = 0L
            var frameIdx = 0
            while (curTimeMs < durationMs) {
                val timeUs = curTimeMs * 1000L
                val frameBitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if (frameBitmap != null) {
                    val secondsTotal = curTimeMs / 1000
                    val hh = secondsTotal / 3600
                    val mm = (secondsTotal % 3600) / 60
                    val ss = secondsTotal % 60
                    val ts = String.format(Locale.US, "%02d:%02d:%02d", hh, mm, ss)

                    val filename = String.format(Locale.US, "%s_%02d_%02d_%02d_%05d.jpg", cameraName, hh, mm, ss, frameIdx)
                    val outFile = File(frameOutputDir, filename)
                    FileOutputStream(outFile).use { fos ->
                        frameBitmap.compress(Bitmap.CompressFormat.JPEG, 85, fos)
                    }

                    onProgress(secondsTotal, totalSec, frameIdx)
                    onKeyframeDecoded(frameBitmap, ts, System.currentTimeMillis() - (durationMs - curTimeMs), outFile.absolutePath)
                    frameIdx++
                }
                curTimeMs += intervalMs
            }
        } catch (e: Exception) {
            Log.e("VideoFrameDecoder", "Error decoding video: ${e.message}", e)
            throw e
        } finally {
            try {
                retriever.release()
            } catch (_: Exception) {}
        }
    }

    /**
     * Create CameraX ImageAnalysis analyzer for real-time live CCTV feed decoding.
     */
    fun createLiveStreamAnalyzer(
        onLiveFrame: (Bitmap, String, Long) -> Unit
    ): ImageAnalysis.Analyzer {
        return ImageAnalysis.Analyzer { imageProxy ->
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                val now = System.currentTimeMillis()
                val seconds = (now / 1000) % 86400
                val hh = (seconds / 3600) % 24
                val mm = (seconds % 3600) / 60
                val ss = seconds % 60
                val ts = String.format(Locale.US, "%02d:%02d:%02d", hh, mm, ss)
                onLiveFrame(bitmap, ts, now)
            }
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val planeY = image.planes[0].buffer
        val planeU = image.planes[1].buffer
        val planeV = image.planes[2].buffer

        val sizeY = planeY.remaining()
        val sizeU = planeU.remaining()
        val sizeV = planeV.remaining()

        val nv21 = ByteArray(sizeY + sizeU + sizeV)
        planeY.get(nv21, 0, sizeY)
        planeV.get(nv21, sizeY, sizeV)
        planeU.get(nv21, sizeY + sizeV, sizeU)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 85, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
