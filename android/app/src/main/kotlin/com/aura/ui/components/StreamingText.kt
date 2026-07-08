package com.aura.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.keyframes
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle

/**
 * Compose the annotated string for a streaming response — appends
 * a colored cursor block (▍) at the end. Used by [StreamingText].
 */
fun buildStreamingAnnotatedString(
    text: String,
    cursorColor: Color,
    isStreaming: Boolean,
    colors: MarkdownColors,
): AnnotatedString = buildAnnotatedString {
    append(parseMarkdown(text, colors))
    if (isStreaming) {
        withStyle(SpanStyle(color = cursorColor)) {
            append(" ▍")
        }
    }
}

/**
 * Lightweight streaming text — the standard text rendering but with
 * a blinking cursor caret at the end. Mimics the look of Cursor /
 * ChatGPT / Claude. Falls back to plain markdown rendering when
 * the message is done streaming.
 */
@Composable
fun StreamingText(
    text: String,
    isStreaming: Boolean,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.bodyLarge,
    cursorColor: Color = MaterialTheme.colorScheme.primary,
) {
    // The infinite transition must live in a Composable context.
    //
    // Binary on/off blink at 530ms per cycle (53% on, 47% off — a
    // 50/50 split feels too nervous). Using keyframes with two
    // steps produces a hard cut between on and off, like a real
    // terminal cursor and the ChatGPT / Claude streaming
    // indicator. The previous tween + RepeatMode.Reverse
    // produced a smooth 0.3 → 1.0 alpha fade, which looks like
    // throbbing rather than typing.
    val infiniteTransition = rememberInfiniteTransition(label = "cursor")
    val cursorAlpha by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = keyframes<Float> {
                durationMillis = 530
                1f at 0
                1f at 280  // on for ~53% of cycle
                0f at 281  // hard cut to off
                0f at 529
                1f at 530  // hard cut back to on
            },
            repeatMode = RepeatMode.Restart,
        ),
        label = "alpha",
    )
    val effectiveCursor = cursorColor.copy(alpha = cursorAlpha)
    val colors = rememberMarkdownColors()
    val annotated = buildStreamingAnnotatedString(
        text = text,
        cursorColor = effectiveCursor,
        isStreaming = isStreaming,
        colors = colors,
    )
    Text(
        text = annotated,
        style = style,
        modifier = modifier,
    )
}