package com.aura.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
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
    val infiniteTransition = rememberInfiniteTransition(label = "cursor")
    val cursorAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 900, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
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