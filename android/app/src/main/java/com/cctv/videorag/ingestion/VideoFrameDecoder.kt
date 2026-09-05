package com.cctv.videorag.ingestion

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

class VideoFrameDecoder(private val context: Context) {

    @Volatile
    var isCancelled = false
        private set

    fun cancel() {
        isCancelled = true
        Log.i("VideoFrameDecoder", "Video decoding cancellation requested")
    }

    companion object {
        /**
         * Longest edge for a stored keyframe.
         *
         * Frames were previously stored at the video's native resolution. Vision-token
         * count scales with that, and encode time scales with tokens: a 1280x720 frame
         * is ~1125 tokens and ~87 s on an SD8Gen2, versus ~264 tokens and ~20 s at
         * 640x360. Storing 720p also overflowed the model's context once several frames
         * were sent together. 640 keeps a 16:9 frame near ~264 tokens.
         */
        const val MAX_KEYFRAME_DIM = 640
    }

    /** Scale so the longest edge is at most [maxDim], preserving aspect ratio. */
    private fun downscale(src: Bitmap, maxDim: Int): Bitmap {
        val longest = maxOf(src.width, src.height)
        if (longest <= maxDim) return src
        val scale = maxDim.toFloat() / longest
        val w = (src.width * scale).toInt().coerceAtLeast(1)
        val h = (src.height * scale).toInt().coerceAtLeast(1)
        val out = Bitmap.createScaledBitmap(src, w, h, true)
        if (out != src) src.recycle()
        return out
    }

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
        isCancelled = false
        // Sequential decode first; the seek-per-frame path below is the fallback.
        //
        // getFrameAtTime(OPTION_CLOSEST) has to decode forward from the preceding sync
        // frame on EVERY call, so sampling 812 times over a 13-minute clip re-decodes
        // most of the file once per sample. Measured on a Vivo I2304: 1109 s to ingest.
        // Decoding the stream once and keeping every Nth frame does the same work in one
        // pass. Any failure - odd codec, DRM, unusual colour format - falls through to
        // the original path, which is slow but known to work everywhere.
        try {
            decodeVideoSequential(videoUri, cameraName, sampleFps, onProgress, onKeyframeDecoded)
            return@withContext
        } catch (e: Throwable) {
            if (isCancelled) return@withContext
            Log.w("VideoFrameDecoder", "sequential decode failed (${e.message}); " +
                  "falling back to per-frame seeking")
        }

        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val totalSec = durationMs / 1000L
            val intervalMs = (1000.0 / sampleFps).toLong().coerceAtLeast(300L)

            val frameOutputDir = File(context.filesDir, "extracted_frames/$cameraName").apply { mkdirs() }

            var curTimeMs = 0L
            var frameIdx = 0
            while (curTimeMs < durationMs && !isCancelled) {
                val timeUs = curTimeMs * 1000L
                // OPTION_CLOSEST, not OPTION_CLOSEST_SYNC.
                //
                // CLOSEST_SYNC snaps to the nearest codec sync frame rather than the
                // requested time, so on a clip with a multi-second GOP a 1 fps sweep
                // returns the same decoded I-frame over and over. Measured on the
                // 13-minute sample: 812 requests produced 156 distinct images, 80.8%
                // byte-identical, an effective rate of one frame every 5.2 s.
                //
                // The wasted work was never the point. Anything visible only between two
                // sync frames was not deduplicated, it was never decoded - so a vehicle
                // that passes through in three seconds could not be found at all. For a
                // tool whose whole purpose is "when did X appear", that is a correctness
                // bug, and it also made the dHash gate's statistics meaningless: of 680
                // frames it reported dropping, 656 were exact re-decodes.
                //
                // CLOSEST decodes forward from the preceding sync frame to the requested
                // timestamp, which costs more per call. If that proves too slow the fix
                // is a sequential MediaCodec/MediaExtractor pass rather than going back
                // to sampling the same picture repeatedly.
                val rawFrame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                val frameBitmap = rawFrame?.let { downscale(it, MAX_KEYFRAME_DIM) }
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
     * Decode the video once, linearly, keeping one frame per sampling interval.
     *
     * MediaExtractor feeds encoded samples to a MediaCodec decoder in ByteBuffer mode;
     * getOutputImage() hands back a YUV_420_888 Image for each decoded frame, and frames
     * are kept only when the presentation timestamp crosses the next sampling boundary.
     * Every frame is still decoded - that is unavoidable - but each is decoded exactly
     * once, instead of once per seek that happens to span it.
     */
    private suspend fun decodeVideoSequential(
        videoUri: Uri,
        cameraName: String,
        sampleFps: Double,
        onProgress: (currentSec: Long, totalSec: Long, frameIndex: Int) -> Unit,
        onKeyframeDecoded: suspend (Bitmap, String, Long, String) -> Unit
    ) {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, videoUri, null)

            var track = -1
            for (i in 0 until extractor.trackCount) {
                val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("video/")) { track = i; break }
            }
            require(track >= 0) { "no video track" }
            extractor.selectTrack(track)

            val format = extractor.getTrackFormat(track)
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION))
                format.getLong(MediaFormat.KEY_DURATION) else 0L
            val totalSec = durationUs / 1_000_000L

            // Ask for a layout getOutputImage() can describe. Decoders that ignore this
            // still work: the Image API reports whatever planes they actually produced.
            format.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            )

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)   // null surface = ByteBuffer mode
            codec.start()

            val frameOutputDir = File(context.filesDir, "extracted_frames/$cameraName").apply { mkdirs() }
            val intervalUs = (1_000_000.0 / sampleFps).toLong().coerceAtLeast(1L)
            var nextEmitUs = 0L
            var frameIdx = 0
            var sawInputEos = false
            var sawOutputEos = false
            val info = MediaCodec.BufferInfo()

            while (!sawOutputEos && !isCancelled) {
                if (!sawInputEos) {
                    val inIndex = codec.dequeueInputBuffer(10_000)
                    if (inIndex >= 0) {
                        val buf = codec.getInputBuffer(inIndex)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEos = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIndex = codec.dequeueOutputBuffer(info, 10_000)
                if (outIndex < 0) {
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEos = true
                    continue
                }
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEos = true

                val wanted = info.size > 0 && info.presentationTimeUs >= nextEmitUs
                if (wanted) {
                    // Skip whole intervals rather than emitting a burst, in case the
                    // stream has a gap longer than one sampling period.
                    while (nextEmitUs <= info.presentationTimeUs) nextEmitUs += intervalUs

                    val secondsTotal = info.presentationTimeUs / 1_000_000L
                    val hh = secondsTotal / 3600
                    val mm = (secondsTotal % 3600) / 60
                    val ss = secondsTotal % 60
                    val ts = String.format(Locale.US, "%02d:%02d:%02d", hh, mm, ss)
                    val filename = String.format(
                        Locale.US, "%s_%02d_%02d_%02d_%05d.jpg", cameraName, hh, mm, ss, frameIdx)
                    val outFile = File(frameOutputDir, filename)

                    val savedOk = try {
                        codec.getOutputImage(outIndex)?.use { image ->
                            imageToJpegFile(image, outFile, 85)
                            true
                        } ?: false
                    } catch (e: Throwable) {
                        Log.w("VideoFrameDecoder", "frame convert failed: ${e.message}")
                        false
                    }

                    if (savedOk && outFile.isFile && outFile.length() > 0L) {
                        val decoded = BitmapFactory.decodeFile(outFile.absolutePath)
                        val frameBitmap = if (decoded != null) downscale(decoded, MAX_KEYFRAME_DIM) else null
                        if (frameBitmap != null) {
                            onProgress(secondsTotal, totalSec, frameIdx)
                            onKeyframeDecoded(
                                frameBitmap, ts,
                                System.currentTimeMillis() - (durationUs / 1000 - secondsTotal * 1000),
                                outFile.absolutePath
                            )
                            frameIdx++
                        }
                    }
                }
                codec.releaseOutputBuffer(outIndex, false)
            }
            Log.i("VideoFrameDecoder", "sequential decode: emitted $frameIdx frames")
        } finally {
            try { codec?.stop() } catch (_: Throwable) {}
            try { codec?.release() } catch (_: Throwable) {}
            try { extractor.release() } catch (_: Throwable) {}
        }
    }

    /**
     * YUV_420_888 Image to JPEG file directly, via NV21 in a single compression pass.
     * Avoids generational loss and memory allocations from double compression.
     */
    private fun imageToJpegFile(image: Image, targetFile: File, quality: Int = 85) {
        val w = image.width
        val h = image.height
        val nv21 = ByteArray(w * h * 3 / 2)

        val yPlane = image.planes[0]
        val yBuf = yPlane.buffer
        var pos = 0
        if (yPlane.rowStride == w && yPlane.pixelStride == 1) {
            yBuf.get(nv21, 0, w * h)
            pos = w * h
        } else {
            for (row in 0 until h) {
                yBuf.position(row * yPlane.rowStride)
                if (yPlane.pixelStride == 1) {
                    yBuf.get(nv21, pos, w); pos += w
                } else {
                    for (col in 0 until w) {
                        nv21[pos++] = yBuf.get(row * yPlane.rowStride + col * yPlane.pixelStride)
                    }
                }
            }
        }

        // NV21 expects V then U, interleaved, at half resolution.
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                nv21[pos++] = vBuf.get(row * vPlane.rowStride + col * vPlane.pixelStride)
                nv21[pos++] = uBuf.get(row * uPlane.rowStride + col * uPlane.pixelStride)
            }
        }

        FileOutputStream(targetFile).use { fos ->
            YuvImage(nv21, ImageFormat.NV21, w, h, null)
                .compressToJpeg(Rect(0, 0, w, h), quality, fos)
        }
    }
}
