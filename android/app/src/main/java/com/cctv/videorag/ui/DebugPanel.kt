package com.cctv.videorag.ui

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Typeface
import android.text.TextUtils
import android.util.TypedValue
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.cctv.videorag.R

/**
 * The diagnostics that used to occupy the main screen.
 *
 * The redesign removed the hash-telemetry block, metrics grid, JSON inspector and
 * sampling controls because they were debug surface competing with the conversation.
 * They are still genuinely useful when an answer looks wrong — knowing whether the
 * right frame was even retrieved is the difference between a retrieval bug and a model
 * mistake — so they live here instead, one tap away and out of the way.
 */
object DebugPanel {

    /** Snapshot of everything worth inspecting after a query. */
    data class State(
        val modelInfo: String,
        val embedderInfo: String,
        val tokenizerInfo: String,
        val keyframesKept: Int,
        val duplicatesDropped: Int,
        val vectorCount: Int,
        val ftsRows: Int,
        val sampleFps: Double,
        val gateEnabled: Boolean,
        val gateThreshold: Int,
        val lastQuery: String?,
        /** Ranked retrieval hits: timestamp, region, score. */
        val lastHits: List<Triple<String, String, Float>>,
        val framesSentToModel: List<String>,
        val droppedTimestamps: List<String>,
        val genStats: String,
        val lastLatencyMs: Long?,
        val indexedJson: String?
    )

    fun show(ctx: Context, s: State) {
        val root = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(ctx, 20f), dp(ctx, 16f), dp(ctx, 20f), dp(ctx, 8f))
        }

        section(ctx, root, "Pipeline")
        kv(ctx, root, "VLM", s.modelInfo)
        kv(ctx, root, "Embedder", s.embedderInfo)
        kv(ctx, root, "Tokenizer", s.tokenizerInfo)

        section(ctx, root, "Index")
        kv(ctx, root, "Keyframes kept", s.keyframesKept.toString())
        kv(ctx, root, "Duplicates dropped", s.duplicatesDropped.toString())
        val total = s.keyframesKept + s.duplicatesDropped
        kv(ctx, root, "Gate drop rate",
            if (total > 0) "%.1f%%".format(s.duplicatesDropped * 100.0 / total) else "—")
        kv(ctx, root, "Dense vectors", "${s.vectorCount}  (6 regions per frame)")
        kv(ctx, root, "FTS rows", s.ftsRows.toString())

        section(ctx, root, "Ingestion settings")
        kv(ctx, root, "Sampling", "${s.sampleFps} FPS")
        kv(ctx, root, "dHash gate", if (s.gateEnabled) "on, threshold ${s.gateThreshold}" else "off")

        section(ctx, root, "Last query")
        if (s.lastQuery == null) {
            kv(ctx, root, "—", "no query run yet")
        } else {
            kv(ctx, root, "Question", s.lastQuery)
            s.lastLatencyMs?.let { kv(ctx, root, "Latency", "${it / 1000}s") }
            if (s.genStats.isNotBlank())
                for (part in s.genStats.split(" ")) {
                    val kvp = part.split("=")
                    if (kvp.size == 2) kv(ctx, root, kvp[0], kvp[1])
                }
            kv(ctx, root, "Frames sent", s.framesSentToModel.joinToString(", ").ifEmpty { "—" })
            if (s.droppedTimestamps.isNotEmpty())
                kv(ctx, root, "Dropped (not shown to model)", s.droppedTimestamps.joinToString(", "))
            mono(ctx, root, "Retrieval ranking (max-pooled per frame)")
            if (s.lastHits.isEmpty()) mono(ctx, root, "  (none)")
            for ((i, h) in s.lastHits.withIndex()) {
                val sent = if (s.framesSentToModel.contains(h.first)) "sent" else "—"
                mono(ctx, root, "  %d. %s  [%s]  %.3f  %s".format(i + 1, h.first, h.second, h.third, sent))
            }
        }

        val scroll = ScrollView(ctx).apply { addView(root) }

        val dlg = AlertDialog.Builder(ctx)
            .setTitle("Diagnostics")
            .setView(scroll)
            .setPositiveButton("Close", null)
            .setNeutralButton("Copy") { _, _ ->
                val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("VideoRAG diagnostics", asText(s)))
            }
            .create()

        if (s.indexedJson != null) {
            dlg.setButton(AlertDialog.BUTTON_NEGATIVE, "Indexed JSON") { _, _ ->
                AlertDialog.Builder(ctx)
                    .setTitle("Indexed keyframes")
                    .setView(ScrollView(ctx).apply {
                        addView(TextView(ctx).apply {
                            text = s.indexedJson
                            typeface = Typeface.MONOSPACE
                            textSize = 10f
                            setPadding(dp(ctx, 16f), dp(ctx, 12f), dp(ctx, 16f), dp(ctx, 12f))
                            setTextIsSelectable(true)
                        })
                    })
                    .setPositiveButton("Close", null)
                    .show()
            }
        }
        dlg.show()
    }

    // ── rendering helpers ─────────────────────────────────────────

    private fun dp(ctx: Context, v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, ctx.resources.displayMetrics).toInt()

    private fun section(ctx: Context, parent: LinearLayout, title: String) {
        parent.addView(TextView(ctx).apply {
            text = title.uppercase()
            setTextColor(ctx.getColor(R.color.primary))
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
            letterSpacing = 0.08f
            setPadding(0, dp(ctx, 14f), 0, dp(ctx, 6f))
        })
    }

    private fun kv(ctx: Context, parent: LinearLayout, k: String, v: String) {
        parent.addView(LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(ctx, 2f), 0, dp(ctx, 2f))
            addView(TextView(ctx).apply {
                text = k
                setTextColor(ctx.getColor(R.color.text_muted))
                textSize = 12.5f
                layoutParams = LinearLayout.LayoutParams(dp(ctx, 118f), LinearLayout.LayoutParams.WRAP_CONTENT)
            })
            addView(TextView(ctx).apply {
                text = v
                setTextColor(ctx.getColor(R.color.text_main))
                textSize = 12.5f
                ellipsize = TextUtils.TruncateAt.MIDDLE
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
        })
    }

    private fun mono(ctx: Context, parent: LinearLayout, line: String) {
        parent.addView(TextView(ctx).apply {
            text = line
            typeface = Typeface.MONOSPACE
            setTextColor(ctx.getColor(R.color.text_main))
            textSize = 11.5f
            setPadding(0, dp(ctx, 1f), 0, dp(ctx, 1f))
        })
    }

    private fun asText(s: State): String = buildString {
        appendLine("VLM: ${s.modelInfo}")
        appendLine("Embedder: ${s.embedderInfo}")
        appendLine("Tokenizer: ${s.tokenizerInfo}")
        appendLine("Keyframes ${s.keyframesKept}, dropped ${s.duplicatesDropped}, vectors ${s.vectorCount}, fts ${s.ftsRows}")
        appendLine("Sampling ${s.sampleFps} FPS, gate ${if (s.gateEnabled) "on/${s.gateThreshold}" else "off"}")
        appendLine("Query: ${s.lastQuery ?: "-"}  latency ${s.lastLatencyMs ?: "-"}ms")
        appendLine("Sent: ${s.framesSentToModel.joinToString(", ")}")
        appendLine("Dropped: ${s.droppedTimestamps.joinToString(", ").ifEmpty { "-" }}")
        for ((i, h) in s.lastHits.withIndex())
            appendLine("  %d. %s [%s] %.3f".format(i + 1, h.first, h.second, h.third))
    }
}
