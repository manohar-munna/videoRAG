package com.cctv.videorag.ingestion

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object VideoDownloader {
    /**
     * Download video from HTTP/HTTPS URL with progress callback.
     */
    suspend fun downloadVideo(
        context: Context,
        videoUrl: String,
        onProgress: (percent: Int, downloadedBytes: Long, totalBytes: Long) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val url = URL(videoUrl)
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 15000
        connection.readTimeout = 30000
        connection.instanceFollowRedirects = true
        connection.connect()

        if (connection.responseCode !in 200..299) {
            throw IllegalStateException("Server returned HTTP ${connection.responseCode}: ${connection.responseMessage}")
        }

        val totalLength = connection.contentLengthLong
        val downloadsDir = File(context.cacheDir, "downloaded_videos").apply { mkdirs() }
        val filename = "video_${System.currentTimeMillis()}.mp4"
        val destinationFile = File(downloadsDir, filename)

        connection.inputStream.use { input ->
            FileOutputStream(destinationFile).use { output ->
                val buffer = ByteArray(8 * 1024)
                var totalBytesRead = 0L
                var bytesRead: Int

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead

                    val percent = if (totalLength > 0) {
                        ((totalBytesRead * 100) / totalLength).toInt()
                    } else {
                        -1
                    }
                    onProgress(percent, totalBytesRead, totalLength)
                }
            }
        }
        Log.i("VideoDownloader", "Successfully downloaded video to ${destinationFile.absolutePath} (${destinationFile.length()} bytes)")
        return@withContext destinationFile
    }
}
