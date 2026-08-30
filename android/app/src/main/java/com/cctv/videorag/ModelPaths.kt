package com.cctv.videorag

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Where model weights live. One answer, for every device.
 *
 * Weights used to be hunted for across a dozen guessed paths under /sdcard/Download,
 * which needs MANAGE_EXTERNAL_STORAGE. That permission is not granted by default, can
 * be revoked on reinstall, and is refused outright by some vendor builds - so the app
 * reported "model not loaded" while the files sat right there on disk, and the only
 * recovery was the Model Folder picker.
 *
 * The app's own external directory needs no permission, cannot be revoked, is visible
 * over adb and MTP for sideloading, and is removed when the app is uninstalled. It is
 * the canonical location; the old Download paths remain only as a fallback for weights
 * already sideloaded there on a device where the permission does happen to be granted.
 */
object ModelPaths {

    private const val TAG = "VideoRAG_Models"

    /** `/sdcard/Android/data/com.cctv.videorag/files/models` - created on first use. */
    fun modelsDir(context: Context): File {
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        return File(base, "models").apply { if (!exists()) mkdirs() }
    }

    /** Ordered search path: canonical location first, legacy sideload spots after. */
    fun searchDirs(context: Context): List<File> {
        val dirs = mutableListOf(modelsDir(context))
        dirs += File(context.filesDir, "models")
        // legacy locations, only readable with all-files access
        dirs += listOf(
            File("/storage/emulated/0/Download/qwen2_vl_2b"),
            File("/storage/emulated/0/Download/mobileclip"),
            File("/storage/emulated/0/Download"),
            File("/sdcard/Download/qwen2_vl_2b"),
            File("/sdcard/Download/mobileclip"),
            File("/sdcard/Download")
        )
        return dirs.filter { it.isDirectory }
    }

    /**
     * First readable file in the search path matching [predicate].
     * Readability is tested, not assumed: a directory can list on a device where the
     * app is not permitted to open what is inside it.
     */
    fun find(context: Context, predicate: (File) -> Boolean): File? {
        for (dir in searchDirs(context)) {
            val hit = (dir.listFiles() ?: continue).firstOrNull { it.isFile && predicate(it) && it.canRead() }
            if (hit != null) {
                Log.i(TAG, "found ${hit.name} in ${dir.absolutePath}")
                return hit
            }
        }
        return null
    }

    /** Path to show the user when something is missing. */
    fun instructions(context: Context): String =
        "Copy the model files to:\n${modelsDir(context).absolutePath}"
}
