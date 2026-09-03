package com.cctv.videorag.ui

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import android.text.style.RelativeSizeSpan
import java.util.Locale
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.Spannable
import android.text.SpannableString
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.cctv.videorag.R
import java.io.File

/**
 * Renders the question/answer thread.
 *
 * Built with plain views rather than a RecyclerView: a conversation is a short,
 * append-only list, and this keeps the construction style consistent with the rest of
 * the screen without adding an adapter for a handful of rows.
 */
object ChatView {

    private const val PENDING_TAG = "pending"

    /** A keyframe that was actually sent to the model, shown as evidence under its answer. */
    data class FrameRef(val imagePath: String, val timestamp: String, val seconds: Int)

    /** A timestamp anywhere in an answer becomes a tappable seek link. */
    private val TIMESTAMP = Regex("""\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b""")

    private fun dp(ctx: Context, v: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, ctx.resources.displayMetrics).toInt()

    private fun bubbleBg(ctx: Context, fill: Int, stroke: Int?): GradientDrawable =
        GradientDrawable().apply {
            setColor(fill)
            cornerRadius = dp(ctx, 14f).toFloat()
            if (stroke != null) setStroke(dp(ctx, 1f), stroke)
        }

    /** The operator's question: right-aligned, solid blue. */
    /**
     * The question bubble. Returns the TextView so the caller can stamp the elapsed
     * time onto it once the answer lands - see setUserMessageTiming().
     */
    fun addUserMessage(container: LinearLayout, text: String): TextView {
        val ctx = container.context
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(ctx, 8f) }
        }
        val bubble = TextView(ctx).apply {
            this.text = text
            setTextColor(Color.WHITE)
            textSize = 14f
            background = bubbleBg(ctx, ctx.getColor(R.color.primary), null)
            setPadding(dp(ctx, 14f), dp(ctx, 10f), dp(ctx, 14f), dp(ctx, 10f))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { marginStart = dp(ctx, 48f) }   // leave a gutter so it reads as a reply
        }
        row.addView(bubble)
        container.addView(row)
        return bubble
    }

    /**
     * Append how long the answer took to the question bubble, dimmed and smaller so it
     * reads as an annotation rather than part of the question. Makes per-query cost
     * visible without opening the debug panel.
     */
    fun setUserMessageTiming(bubble: TextView, question: String, millis: Long) {
        val secs = millis / 1000.0
        val label = if (secs < 10) String.format(Locale.US, "  %.1fs", secs)
                    else String.format(Locale.US, "  %ds", (millis / 1000))
        val sp = SpannableString(question + label)
        val from = question.length
        sp.setSpan(RelativeSizeSpan(0.8f), from, sp.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        sp.setSpan(ForegroundColorSpan(Color.argb(170, 255, 255, 255)),
                   from, sp.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        bubble.text = sp
    }

    /**
     * The model's answer: left-aligned white card, with timestamps turned into links.
     * [onTimestamp] receives seconds into the video.
     */
    fun addAssistantMessage(
        container: LinearLayout,
        text: String,
        onTimestamp: ((Int) -> Unit)? = null,
        frames: List<FrameRef> = emptyList()
    ) {
        val ctx = container.context
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.START
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(ctx, 8f) }
        }
        val bubble = TextView(ctx).apply {
            setTextColor(ctx.getColor(R.color.text_main))
            textSize = 14f
            setLineSpacing(dp(ctx, 2f).toFloat(), 1f)
            background = bubbleBg(ctx, ctx.getColor(R.color.card_bg), ctx.getColor(R.color.card_border))
            setPadding(dp(ctx, 14f), dp(ctx, 10f), dp(ctx, 14f), dp(ctx, 10f))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { marginEnd = dp(ctx, 48f) }
            setTextIsSelectable(true)
        }
        bubble.text = linkifyTimestamps(ctx, text, onTimestamp)
        if (onTimestamp != null) bubble.movementMethod = LinkMovementMethod.getInstance()
        row.addView(bubble)
        container.addView(row)

        if (frames.isNotEmpty()) addFrameStrip(container, frames, onTimestamp)
    }

    /**
     * The keyframes actually handed to the model, under its answer.
     *
     * This is the evidence for the reply: retrieval picks a handful of frames out of the
     * whole video and the model only ever sees those, so showing them makes a wrong
     * answer diagnosable — you can see immediately whether the model misread a frame or
     * simply never received the right one. Tapping a thumbnail seeks the video, the same
     * as tapping a timestamp in the text.
     */
    private fun addFrameStrip(
        container: LinearLayout,
        frames: List<FrameRef>,
        onTimestamp: ((Int) -> Unit)?
    ) {
        val ctx = container.context

        container.addView(TextView(ctx).apply {
            text = "Frames analysed (${frames.size}) — tap to jump"
            setTextColor(ctx.getColor(R.color.text_light))
            textSize = 11f
            setPadding(dp(ctx, 4f), 0, 0, dp(ctx, 4f))
        })

        val scroller = HorizontalScrollView(ctx).apply {
            isHorizontalScrollBarEnabled = false
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(ctx, 10f) }
        }
        val strip = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }

        for (f in frames) {
            val cell = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                background = bubbleBg(ctx, ctx.getColor(R.color.card_bg), ctx.getColor(R.color.card_border))
                setPadding(dp(ctx, 4f), dp(ctx, 4f), dp(ctx, 4f), dp(ctx, 4f))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { marginEnd = dp(ctx, 6f) }
                isClickable = true
                setOnClickListener { onTimestamp?.invoke(f.seconds) }
            }
            cell.addView(ImageView(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(dp(ctx, 116f), dp(ctx, 66f))
                scaleType = ImageView.ScaleType.CENTER_CROP
                // decode small: these are thumbnails in a scrolling row, and a query can
                // add several at once
                val file = File(f.imagePath)
                if (file.exists()) {
                    val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
                    BitmapFactory.decodeFile(file.absolutePath, opts)?.let { setImageBitmap(it) }
                } else {
                    setBackgroundColor(ctx.getColor(R.color.card_border))
                }
            })
            cell.addView(TextView(ctx).apply {
                text = f.timestamp
                setTextColor(ctx.getColor(R.color.primary))
                textSize = 10.5f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
                setPadding(0, dp(ctx, 3f), 0, 0)
            })
            strip.addView(cell)
        }
        scroller.addView(strip)
        container.addView(scroller)
    }

    /** Placeholder shown while the model runs; replaced by [replacePending]. */
    fun addPending(container: LinearLayout, label: String = "Analysing keyframes…") {
        val ctx = container.context
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.START
            tag = PENDING_TAG
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(ctx, 8f) }
        }
        val inner = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = bubbleBg(ctx, ctx.getColor(R.color.card_bg), ctx.getColor(R.color.card_border))
            setPadding(dp(ctx, 14f), dp(ctx, 10f), dp(ctx, 14f), dp(ctx, 10f))
        }
        inner.addView(ProgressBar(ctx).apply {
            isIndeterminate = true
            layoutParams = LinearLayout.LayoutParams(dp(ctx, 16f), dp(ctx, 16f))
        })
        inner.addView(TextView(ctx).apply {
            text = label
            setTextColor(ctx.getColor(R.color.text_muted))
            textSize = 13f
            setPadding(dp(ctx, 10f), 0, 0, 0)
        })
        row.addView(inner)
        container.addView(row)
    }

    /** Swap the spinner for the finished answer. */
    fun replacePending(
        container: LinearLayout,
        text: String,
        onTimestamp: ((Int) -> Unit)? = null,
        frames: List<FrameRef> = emptyList()
    ) {
        for (i in container.childCount - 1 downTo 0) {
            if (container.getChildAt(i).tag == PENDING_TAG) { container.removeViewAt(i); break }
        }
        addAssistantMessage(container, text, onTimestamp, frames)
    }

    /** A quiet centred note (status, errors) that is not part of the dialogue. */
    fun addSystemNote(container: LinearLayout, text: String, isError: Boolean = false) {
        val ctx = container.context
        container.addView(TextView(ctx).apply {
            this.text = text
            setTextColor(ctx.getColor(if (isError) R.color.danger else R.color.text_muted))
            textSize = 12f
            gravity = Gravity.CENTER
            setPadding(dp(ctx, 16f), dp(ctx, 6f), dp(ctx, 16f), dp(ctx, 10f))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        })
    }

    fun clear(container: LinearLayout) = container.removeAllViews()

    /**
     * Make every HH:MM:SS / MM:SS occurrence a blue tappable span that seeks the player.
     * The model is asked to cite timestamps, so these are the app's main affordance for
     * jumping from an answer to the moment it describes.
     */
    private fun linkifyTimestamps(
        ctx: Context,
        text: String,
        onTimestamp: ((Int) -> Unit)?
    ): CharSequence {
        if (onTimestamp == null) return text
        val sp = SpannableString(text)
        for (m in TIMESTAMP.findAll(text)) {
            val h = m.groupValues[1].toIntOrNull() ?: 0
            val mi = m.groupValues[2].toIntOrNull() ?: 0
            val s = m.groupValues[3].toIntOrNull() ?: 0
            val total = h * 3600 + mi * 60 + s
            val start = m.range.first
            val end = m.range.last + 1
            sp.setSpan(object : ClickableSpan() {
                override fun onClick(widget: View) = onTimestamp(total)
            }, start, end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            sp.setSpan(ForegroundColorSpan(ctx.getColor(R.color.primary)),
                start, end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            sp.setSpan(StyleSpan(Typeface.BOLD),
                start, end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        return sp
    }
}
