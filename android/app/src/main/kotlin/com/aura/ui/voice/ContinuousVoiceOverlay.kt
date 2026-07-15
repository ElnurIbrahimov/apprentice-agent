package com.aura.ui.voice

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlin.math.PI
import kotlin.math.sin

import com.aura.ui.theme.AuraThemeTokens
/**
 * Full-screen continuous voice mode overlay — premium redesign with
 * a layered breathing orb (3 concentric pulsing rings), a live
 * waveform visualizer, phase label, and a clean stop button.
 *
 * The visual language:
 * - LISTENING: red/tertiary orb with waveform bars
 * - THINKING: amber orb with rotating dots
 * - SPEAKING: teal/primary orb with sound waves
 * - IDLE: gray, slow breath
 */
@Composable
fun ContinuousVoiceOverlay(
    state: ContinuousVoiceState,
    onStop: () -> Unit,
) {
    val phase = state.phase
    val accent = when (phase) {
        VoiceModeState.LISTENING -> AuraThemeTokens.colors.assistantAccent
        VoiceModeState.THINKING -> AuraThemeTokens.colors.assistantAccent
        VoiceModeState.SPEAKING -> AuraThemeTokens.colors.actionPrimary
        VoiceModeState.IDLE -> AuraThemeTokens.colors.borderDefault
    }
    val phaseLabel = when (phase) {
        VoiceModeState.LISTENING -> "Listening…"
        VoiceModeState.THINKING -> "Thinking…"
        VoiceModeState.SPEAKING -> "Speaking…"
        VoiceModeState.IDLE -> "Starting…"
    }
    val phaseIcon = when (phase) {
        VoiceModeState.LISTENING -> Icons.Filled.Mic
        VoiceModeState.THINKING -> Icons.Filled.GraphicEq
        VoiceModeState.SPEAKING -> Icons.Filled.VolumeUp
        VoiceModeState.IDLE -> Icons.Filled.Mic
    }

    // Three independent breathing rings with phase-specific timing.
    val ring1 by rememberBreath(1600, phase)
    val ring2 by rememberBreath(1600, phase, offset = 533)
    val ring3 by rememberBreath(1600, phase, offset = 1066)

    Surface(
        color = AuraThemeTokens.colors.background,
        modifier = Modifier.fillMaxSize(),
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            // Subtle radial gradient behind the orb to add depth.
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                accent.copy(alpha = 0.12f),
                                AuraThemeTokens.colors.background,
                            ),
                            radius = 800f,
                        ),
                    ),
            )

            Column(
                modifier = Modifier.fillMaxSize().padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                Spacer(modifier = Modifier.weight(0.5f))

                // ── Breathing orb (3 layered rings + core) ─────────────────
                Box(
                    modifier = Modifier.size(260.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    // Three concentric rings — each pulses with a phase
                    // offset so the whole shape looks like it's breathing
                    // outward, not just wiggling.
                    PulseRing(color = accent, scale = ring1, alpha = 0.18f)
                    PulseRing(color = accent, scale = ring2, alpha = 0.30f)
                    PulseRing(color = accent, scale = ring3, alpha = 0.45f)

                    // Core orb
                    Box(
                        modifier = Modifier
                            .size(96.dp)
                            .background(
                                Brush.radialGradient(
                                    colors = listOf(
                                        accent,
                                        accent.copy(alpha = 0.7f),
                                    ),
                                ),
                                CircleShape,
                            ),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            imageVector = phaseIcon,
                            contentDescription = phaseLabel,
                            tint = AuraThemeTokens.colors.onActionPrimary,
                            modifier = Modifier.size(40.dp),
                        )
                    }
                }

                Spacer(modifier = Modifier.height(40.dp))

                // ── Phase label ────────────────────────────────────────────
                Text(
                    text = phaseLabel,
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = AuraThemeTokens.colors.textPrimary,
                )

                Spacer(modifier = Modifier.height(20.dp))

                // ── Waveform / transcript area ─────────────────────────────
                when (phase) {
                    VoiceModeState.LISTENING -> {
                        if (state.partialTranscript.isNotBlank()) {
                            Text(
                                text = state.partialTranscript,
                                style = MaterialTheme.typography.bodyLarge,
                                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.7f),
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                            )
                        } else {
                            WaveformVisualizer(
                                color = accent,
                                bars = 24,
                                isActive = true,
                            )
                        }
                    }
                    VoiceModeState.SPEAKING -> {
                        if (state.lastResponse.isNotBlank()) {
                            Text(
                                text = state.lastResponse.take(200),
                                style = MaterialTheme.typography.bodyMedium,
                                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                            )
                        } else {
                            WaveformVisualizer(
                                color = accent,
                                bars = 24,
                                isActive = true,
                            )
                        }
                    }
                    VoiceModeState.THINKING -> {
                        ThinkingDots(color = accent)
                    }
                    VoiceModeState.IDLE -> {
                        // Subtle hint while warming up.
                    }
                }

                // ── Error ───────────────────────────────────────────────────
                state.error?.let { err ->
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = err,
                        style = MaterialTheme.typography.labelSmall,
                        color = AuraThemeTokens.colors.error,
                    )
                }

                Spacer(modifier = Modifier.weight(1f))

                // ── Stop button ────────────────────────────────────────────
                IconButton(
                    onClick = onStop,
                    modifier = Modifier
                        .size(72.dp)
                        .background(AuraThemeTokens.colors.surface1, CircleShape),
                ) {
                    Icon(
                        imageVector = Icons.Filled.Stop,
                        contentDescription = "Stop voice mode",
                        tint = AuraThemeTokens.colors.textPrimary,
                        modifier = Modifier.size(32.dp),
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Tap to stop · or say \"stop listening\"",
                    style = MaterialTheme.typography.labelSmall,
                    color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.5f),
                )
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

/**
 * A single pulsing ring. Composed of 2 stacked scaled circles with
 * decreasing alpha to give a halo effect.
 */
@Composable
private fun PulseRing(color: Color, scale: Float, alpha: Float) {
    Box(
        modifier = Modifier
            .size(220.dp)
            .background(
                color = color.copy(alpha = alpha * (1f - scale).coerceIn(0f, 1f)),
                shape = CircleShape,
            ),
    )
}

/**
 * Phase-aware breathing animation. Each phase has a slightly different
 * rhythm so the visual feedback matches what the user expects.
 */
@Composable
private fun rememberBreath(durationMs: Int, phase: VoiceModeState, offset: Int = 0): androidx.compose.runtime.State<Float> {
    val transition = rememberInfiniteTransition(label = "breath-$offset")
    val actualDuration = when (phase) {
        VoiceModeState.LISTENING -> durationMs
        VoiceModeState.THINKING -> (durationMs * 0.7f).toInt()  // faster thinking
        VoiceModeState.SPEAKING -> (durationMs * 1.1f).toInt()  // slower speech
        VoiceModeState.IDLE -> (durationMs * 1.5f).toInt()      // relaxed idle
    }
    return transition.animateFloat(
        initialValue = 0.0f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(actualDuration, easing = LinearEasing),
            repeatMode = RepeatMode.Restart,
            initialStartOffset = androidx.compose.animation.core.StartOffset(offset),
        ),
        label = "breath-value-$offset",
    )
}

/**
 * Animated waveform — a row of vertical bars whose heights oscillate
 * with sine waves. Used during LISTENING and SPEAKING.
 */
@Composable
private fun WaveformVisualizer(
    color: Color,
    bars: Int,
    isActive: Boolean,
) {
    val transition = rememberInfiniteTransition(label = "waveform")
    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = (2 * PI).toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart,
        ),
        label = "wave-phase",
    )
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        modifier = Modifier.height(48.dp),
    ) {
        for (i in 0 until bars) {
            val offset = i * 0.5f
            val normalized = (sin(phase + offset) + 1f) / 2f
            val heightDp = (8 + normalized * 32).dp
            Box(
                modifier = Modifier
                    .width(3.dp)
                    .height(heightDp)
                    .background(
                        color.copy(alpha = if (isActive) 0.5f + normalized * 0.5f else 0.3f),
                        androidx.compose.foundation.shape.RoundedCornerShape(2.dp),
                    ),
            )
        }
    }
}

/**
 * Three pulsing dots for the THINKING phase. Like the iMessage
 * typing indicator.
 */
@Composable
private fun ThinkingDots(color: Color) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (i in 0 until 3) {
            val transition = rememberInfiniteTransition(label = "dot-$i")
            val alpha by transition.animateFloat(
                initialValue = 0.3f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(durationMillis = 600),
                    repeatMode = RepeatMode.Reverse,
                    initialStartOffset = androidx.compose.animation.core.StartOffset(i * 200),
                ),
                label = "dot-alpha-$i",
            )
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .background(color.copy(alpha = alpha), CircleShape),
            )
        }
    }
}