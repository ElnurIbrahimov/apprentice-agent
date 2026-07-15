package com.aura.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.ui.viewmodel.InFlightToolCall

import com.aura.ui.theme.AuraThemeTokens
/**
 * Three states a tool call can be in:
 *
 *  - [Running]: the agentic loop emitted [com.aura.agent.AgentEvent.ToolCallStart]
 *    but no matching [com.aura.agent.AgentEvent.ToolResult] has landed yet.
 *    The badge shows a pulsing cyan dot.
 *
 *  - [Done]: the tool call completed. The full result is rendered in
 *    the conversation's [com.aura.agent.Turn.toolTurns]; the badge
 *    here is a 1-line summary with a green checkmark.
 *
 *  - [Failed]: the tool returned an error. Shows a red X and the
 *    error message.
 */
sealed class ToolCallState {
    data class Running(val inFlight: InFlightToolCall) : ToolCallState()
    data class Done(val name: String, val args: String, val result: String) : ToolCallState()
    data class Failed(val name: String, val args: String, val error: String) : ToolCallState()
}

/**
 * Render a tool call as a small badge below the assistant message.
 * Tapping the badge expands an inline detail view (no popover — keeps
 * the chat scroll uninterrupted).
 */
@Composable
fun ToolCallBadge(
    state: ToolCallState,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    val (containerColor, contentColor, statusIcon, statusTint, summary) = when (state) {
        is ToolCallState.Running -> ToolCallBadgeStyle(
            container = AuraThemeTokens.colors.surface2.copy(alpha = 0.6f),
            content = AuraThemeTokens.colors.textSecondary,
            icon = RunningDot(),
            tint = AuraThemeTokens.colors.assistantAccent,
            summary = "${state.inFlight.name} — running…",
        )
        is ToolCallState.Done -> ToolCallBadgeStyle(
            container = AuraThemeTokens.colors.surface2.copy(alpha = 0.5f),
            content = AuraThemeTokens.colors.textPrimary,
            icon = Icons.Filled.Check,
            tint = AuraThemeTokens.colors.actionPrimary,
            summary = state.name + " — " + state.result.lineSequence().firstOrNull()?.take(120).orEmpty(),
        )
        is ToolCallState.Failed -> ToolCallBadgeStyle(
            container = AuraThemeTokens.colors.error.copy(alpha = 0.5f),
            content = AuraThemeTokens.colors.textPrimary,
            icon = Icons.Filled.Close,
            tint = AuraThemeTokens.colors.error,
            summary = "${state.name} — ${state.error.take(120)}",
        )
    }
    Surface(
        color = containerColor,
        shape = RoundedCornerShape(12.dp),
        modifier = modifier
            .fillMaxWidth(0.85f)
            .padding(start = 32.dp, end = 16.dp, top = 2.dp, bottom = 2.dp)
            .clickable { expanded = !expanded },
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier.size(16.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    when (val ic = statusIcon) {
                        is RunningDot -> RunningDotIndicator()
                        else -> Icon(
                            imageVector = ic as androidx.compose.ui.graphics.vector.ImageVector,
                            contentDescription = null,
                            tint = statusTint,
                            modifier = Modifier.size(14.dp),
                        )
                    }
                }
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = summary,
                    style = MaterialTheme.typography.labelSmall,
                    color = contentColor,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                Icon(
                    imageVector = Icons.Filled.ExpandMore,
                    contentDescription = if (expanded) "Collapse" else "Expand",
                    tint = contentColor.copy(alpha = 0.7f),
                    modifier = Modifier.size(16.dp),
                )
            }
            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut(),
            ) {
                Column(modifier = Modifier.padding(top = 8.dp)) {
                    when (state) {
                        is ToolCallState.Running -> {
                            if (state.inFlight.args.isNotBlank()) {
                                DetailRow("Args", state.inFlight.args, contentColor)
                            }
                            DetailRow(
                                "Status",
                                "Aura is invoking this tool. The result will appear here when it returns.",
                                contentColor,
                            )
                        }
                        is ToolCallState.Done -> {
                            if (state.args.isNotBlank()) DetailRow("Args", state.args, contentColor)
                            DetailRow("Result", state.result, contentColor)
                        }
                        is ToolCallState.Failed -> {
                            if (state.args.isNotBlank()) DetailRow("Args", state.args, contentColor)
                            DetailRow("Error", state.error, contentColor)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String, color: Color) {
    Column(modifier = Modifier.padding(bottom = 6.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = color.copy(alpha = 0.6f),
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = value.take(2000),
            style = MaterialTheme.typography.bodySmall,
            color = color.copy(alpha = 0.95f),
        )
    }
}

/**
 * The pulsing dot used while a tool call is in flight. Implemented
 * as a separate composable so the parent doesn't have to know about
 * the [androidx.compose.animation.core.InfiniteTransition] state.
 */
@Composable
private fun RunningDotIndicator() {
    val transition = rememberInfiniteTransition(label = "tool-running-dot")
    val alpha by transition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(700, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "tool-running-alpha",
    )
    val scale by transition.animateFloat(
        initialValue = 0.85f,
        targetValue = 1.1f,
        animationSpec = infiniteRepeatable(
            animation = tween(700, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "tool-running-scale",
    )
    Box(
        modifier = Modifier
            .size(10.dp * scale)
            .background(
                color = AuraThemeTokens.colors.assistantAccent.copy(alpha = alpha),
                shape = CircleShape,
            ),
    )
}

private data class ToolCallBadgeStyle(
    val container: Color,
    val content: Color,
    val icon: Any,  // ImageVector or RunningDot sentinel
    val tint: Color,
    val summary: String,
)

/** Sentinel value to distinguish the running dot from real icons. */
private class RunningDot
