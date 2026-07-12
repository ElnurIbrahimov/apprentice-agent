package com.aura.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.keyframes
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aura.ui.theme.AuraTokens
import com.aura.ui.theme.JetBrainsMono

/**
 * Compose the annotated string for a streaming response — appends
 * a colored cursor block (▍) at the end. Uses [StreamingMarkdownState]
 * to suppress trailing unclosed markdown markers (e.g. `**` at
 * the end of a partial bold span) so the rendered text doesn't
 * flicker as the closing marker arrives in the next chunk.
 */
fun buildStreamingAnnotatedString(
    text: String,
    cursorColor: Color,
    isStreaming: Boolean,
    colors: MarkdownColors,
    state: StreamingMarkdownState = StreamingMarkdownState(),
    clickable: Boolean = true,
): AnnotatedString = buildAnnotatedString {
    append(state.render(text, colors, clickable))
    if (isStreaming) {
        withStyle(SpanStyle(color = cursorColor)) {
            append(" ▍")
        }
    }
}

/**
 * Approximate tokens/sec rate for the streaming badge.
 *
 * The Aura Web UI shows `42 tok/s` next to the streaming cursor.
 * We don't have the tokenizer the model uses, but a
 * 4-characters-per-token heuristic is the same one the Web uses
 * (it shows this in a title attribute: "tokens estimated from
 * chars / 4"). Average over the second-most-recent second so the
 * number doesn't jump wildly on every chunk — windows shorter
 * than 1s would oscillate between 0 and very high.
 */
internal fun estimateTokensPerSecond(
    charCount: Int,
    startTimeMs: Long,
    nowMs: Long,
): Int {
    val elapsedSec = (nowMs - startTimeMs).coerceAtLeast(1L) / 1000.0
    if (elapsedSec < 0.5) return 0
    val tokens = charCount / 4.0
    return (tokens / elapsedSec).toInt()
}

/**
 * Lightweight streaming text — the standard text rendering but with
 * a blinking cursor caret at the end. Mimics the look of Cursor /
 * ChatGPT / Claude. Falls back to plain markdown rendering when
 * the message is done streaming.
 *
 * During streaming, also shows a small `42 tok/s` badge in
 * JetBrains Mono to the right of the text. The badge is hidden
 * the instant streaming stops.
 */
@Composable
fun StreamingText(
    text: String,
    isStreaming: Boolean,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.bodyLarge,
    cursorColor: Color = AuraTokens.Dark.accentPurple,
) {
    // The infinite transition is only created while streaming. Once
    // streaming stops, we skip the transition entirely so the cursor
    // blink animator doesn't drain battery on every completed message
    // in the conversation list. The cursor color is only applied to
    // the annotated string when isStreaming is true (see
    // buildStreamingAnnotatedString), but the underlying
    // rememberInfiniteTransition would otherwise keep ticking.
    val colors = rememberMarkdownColors()
    val state = remember { StreamingMarkdownState() }
    val effectiveCursor = if (isStreaming) {
        val infiniteTransition = rememberInfiniteTransition(label = "cursor")
        val cursorAlpha by infiniteTransition.animateFloat(
            initialValue = 1f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = keyframes<Float> {
                    durationMillis = 530
                    1f at 0
                    1f at 280
                    0f at 281
                    0f at 529
                    1f at 530
                },
                repeatMode = RepeatMode.Restart,
            ),
            label = "alpha",
        )
        cursorColor.copy(alpha = cursorAlpha)
    } else {
        cursorColor
    }
    val annotated = buildStreamingAnnotatedString(
        text = text,
        cursorColor = effectiveCursor,
        isStreaming = isStreaming,
        colors = colors,
        state = state,
    )

    // Track elapsed time during streaming. We use a remember'd
    // startTime so the elapsed counter begins the moment the
    // first chunk arrives, and is reset on stream end.
    val startTimeMs = remember { mutableLongStateOf(0L) }
    val lastNowMs = remember { mutableLongStateOf(0L) }
    var tokensPerSec by remember { mutableStateOf(0) }
    LaunchedEffect(isStreaming) {
        if (isStreaming) {
            startTimeMs.longValue = System.currentTimeMillis()
            lastNowMs.longValue = startTimeMs.longValue
        } else {
            tokensPerSec = 0
        }
    }
    if (isStreaming) {
        LaunchedEffect(text) {
            val now = System.currentTimeMillis()
            tokensPerSec = estimateTokensPerSecond(text.length, startTimeMs.longValue, now)
            lastNowMs.longValue = now
        }
    }

    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = annotated,
            style = style,
            modifier = modifier,
        )
        if (isStreaming && tokensPerSec > 0) {
            Spacer(Modifier.width(8.dp))
            Surface(
                color = AuraTokens.Dark.surface2,
                shape = RoundedCornerShape(6.dp),
            ) {
                Text(
                    text = "$tokensPerSec tok/s",
                    fontFamily = JetBrainsMono,
                    fontSize = 10.sp,
                    color = AuraTokens.Dark.textSecondary,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                )
            }
        }
    }
}
