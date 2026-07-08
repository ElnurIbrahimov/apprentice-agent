package com.aura.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aura.tools.Citation
import com.aura.ui.theme.AuraTokens
import com.aura.ui.theme.Fraunces
import com.aura.ui.theme.InterDisplay
import com.aura.ui.theme.JetBrainsMono
import com.aura.ui.util.formatRelativeTime
import kotlin.math.sin

/**
 * The Aura AI avatar used in the chat header and as the inline
 * avatar to the left of every assistant message.
 *
 * This is the Android equivalent of `AuraBreathingAvatar.tsx` on
 * the web: a 36–44dp circle with a violet radial glow ring that
 * breathes in and out via a slow `infiniteRepeatable` animation.
 * The inner circle is a subtle white-to-transparent gradient
 * with a 1px white/07 border — the same look as the Web avatar.
 *
 * `isThinking` swaps the inner icon from the sparkles to a bolt
 * and speeds the breathing animation up — the Web calls this the
 * "thinking" state.
 */
@Composable
fun AuraAiAvatar(
    isActive: Boolean = true,
    isThinking: Boolean = false,
    isProactive: Boolean = false,
    size: Dp = 36.dp,
    modifier: Modifier = Modifier,
) {
    val transition = rememberInfiniteTransition(label = "aura-avatar")
    val ringScale by transition.animateFloat(
        initialValue = 1f,
        targetValue = if (isThinking) 1.18f else 1.12f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (isThinking) 900 else 2400,
                easing = LinearEasing,
            ),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "ring-scale",
    )
    val ringAlpha by transition.animateFloat(
        initialValue = 0.25f,
        targetValue = if (isThinking) 0.55f else 0.40f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (isThinking) 900 else 2400,
                easing = LinearEasing,
            ),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "ring-alpha",
    )
    Box(
        modifier = modifier.size(size + 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        // Outer breathing ring — radial gradient blur
        Box(
            modifier = Modifier
                .size(size + 8.dp)
                .graphicsLayer {
                    scaleX = ringScale
                    scaleY = ringScale
                    alpha = if (isActive) ringAlpha else 0.10f
                }
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            AuraTokens.Dark.glowPurple,
                            AuraTokens.Dark.glowBlue.copy(alpha = 0.20f),
                            Color.Transparent,
                        ),
                        radius = (size.value * 1.4f),
                    ),
                    shape = CircleShape,
                ),
        )
        // Inner avatar
        Box(
            modifier = Modifier
                .size(size)
                .clip(CircleShape)
                .background(
                    brush = if (isProactive) AuraTokens.proactiveAvatarGradient
                            else AuraTokens.avatarGradient,
                )
                .border(
                    width = 1.dp,
                    color = AuraTokens.Dark.avatarBorder,
                    shape = CircleShape,
                ),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = if (isProactive || isThinking) Icons.Filled.Bolt else Icons.Filled.AutoAwesome,
                contentDescription = null,
                tint = if (isProactive) AuraTokens.Dark.accentPurple else Color.White,
                modifier = Modifier.size(size * 0.5f),
            )
        }
        // Proactive ping dot
        if (isProactive) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .size(10.dp)
                    .background(
                        color = AuraTokens.Dark.proactivePing,
                        shape = CircleShape,
                    ),
            )
        }
    }
}

/**
 * Animated thinking-shimmer row. Three dots rising and falling
 * with a 150ms stagger — the universal "AI is thinking" pattern.
 * Replaces the [TypingIndicator] inside the message list because
 * the new design shows the shimmer next to the AI avatar, not
 * inside a bubble.
 */
@Composable
fun ThinkingShimmer(modifier: Modifier = Modifier) {
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        for (i in 0 until 3) {
            val transition = rememberInfiniteTransition(label = "dot-$i")
            val offset by transition.animateFloat(
                initialValue = 0f,
                targetValue = -4f,
                animationSpec = infiniteRepeatable(
                    animation = tween(500, easing = LinearEasing),
                    repeatMode = RepeatMode.Reverse,
                    initialStartOffset = androidx.compose.animation.core.StartOffset(i * 150),
                ),
                label = "y-$i",
            )
            val alpha by transition.animateFloat(
                initialValue = 0.35f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(500, easing = LinearEasing),
                    repeatMode = RepeatMode.Reverse,
                    initialStartOffset = androidx.compose.animation.core.StartOffset(i * 150),
                ),
                label = "alpha-$i",
            )
            Box(
                modifier = Modifier
                    .graphicsLayer { translationY = offset }
                    .alpha(alpha)
                    .size(6.dp)
                    .background(
                        color = AuraTokens.Dark.textSecondary,
                        shape = CircleShape,
                    ),
            )
        }
    }
}

/**
 * Message bubble — the new web-aligned design.
 *
 * User bubble: asymmetric radius 24/24/4/24 (pointed at sender),
 * white background, black text, 0 4px 20px rgba(0,0,0,0.25) shadow.
 * 12dp horizontal + 10dp vertical padding, body text 1rem weight 500.
 *
 * Assistant bubble: NO bubble. Just the avatar to the left,
 * the role label (AURA in caps), the streaming text, citations,
 * and a footer with model + timestamp + hover actions. This
 * matches Aura Web's design where the assistant feels like a
 * document, not a chat reply.
 *
 * Stagger entry: pass [animationIndex] 0..4 to get the
 * spring-up entry animation. >4 falls back to index 4.
 */
@Composable
fun MessageBubble(
    text: String,
    isUser: Boolean,
    citations: List<Citation> = emptyList(),
    isStreaming: Boolean = false,
    timestamp: Long = 0L,
    modelLabel: String? = null,
    isProactive: Boolean = false,
    animationIndex: Int = 0,
    onShowSources: () -> Unit = {},
) {
    val context = LocalContext.current
    var copied by remember { mutableStateOf(false) }

    if (isUser) {
        // ── User bubble: pointed at sender ─────────────────────────────
        UserBubble(
            text = text,
            animationIndex = animationIndex,
        )
    } else {
        // ── Assistant: avatar + content, NO bubble ─────────────────────
        AssistantMessage(
            text = text,
            citations = citations,
            isStreaming = isStreaming,
            isProactive = isProactive,
            timestamp = timestamp,
            modelLabel = modelLabel,
            animationIndex = animationIndex,
            copied = copied,
            onCopiedChange = { copied = it },
            onCopy = {
                val clipboard = context.getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                clipboard.setPrimaryClip(android.content.ClipData.newPlainText("Aura", text))
                copied = true
            },
            onShowSources = onShowSources,
        )
    }
}

@Composable
private fun UserBubble(text: String, animationIndex: Int) {
    val springEased = remember { androidx.compose.animation.core.Animatable(0f) }
    LaunchedEffect(Unit) {
        springEased.animateTo(
            targetValue = 1f,
            animationSpec = androidx.compose.animation.core.spring(
                dampingRatio = 0.65f,
                stiffness = androidx.compose.animation.core.Spring.StiffnessMediumLow,
            ),
        )
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .graphicsLayer {
                translationY = (1f - springEased.value) * 16f
                alpha = springEased.value
            }
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.End,
    ) {
        // Pointed-corner bubble: top-left 24, top-right 24,
        // bottom-left 24 (rounded away from sender), bottom-right 4
        // (pointed TOWARD sender). The web is the opposite for
        // the right-aligned user bubble: pointed at the sender.
        Box(
            modifier = Modifier
                .widthIn(max = 340.dp)
                .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp, bottomStart = 24.dp, bottomEnd = 4.dp))
                .background(
                    brush = Brush.linearGradient(
                        colors = listOf(
                            AuraTokens.Dark.userBubbleBg,
                            AuraTokens.Dark.userBubbleBg,
                        ),
                    ),
                )
                .padding(horizontal = 22.dp, vertical = 12.dp),
        ) {
            Text(
                text = text.ifBlank { "…" },
                color = AuraTokens.Dark.userBubbleText,
                fontFamily = InterDisplay,
                fontWeight = FontWeight.Medium,
                fontSize = 16.sp,
                lineHeight = 26.sp,
            )
        }
    }
}

@Composable
private fun AssistantMessage(
    text: String,
    citations: List<Citation>,
    isStreaming: Boolean,
    isProactive: Boolean,
    timestamp: Long,
    modelLabel: String?,
    animationIndex: Int,
    copied: Boolean,
    onCopiedChange: (Boolean) -> Unit,
    onCopy: () -> Unit,
    onShowSources: () -> Unit,
) {
    val springEased = remember { androidx.compose.animation.core.Animatable(0f) }
    LaunchedEffect(Unit) {
        val delayMs = (animationIndex.coerceIn(0, 4) * 30L)
        kotlinx.coroutines.delay(delayMs)
        springEased.animateTo(
            targetValue = 1f,
            animationSpec = androidx.compose.animation.core.spring(
                dampingRatio = 0.65f,
                stiffness = androidx.compose.animation.core.Spring.StiffnessMediumLow,
            ),
        )
    }
    LaunchedEffect(copied) {
        if (copied) {
            kotlinx.coroutines.delay(1500)
            onCopiedChange(false)
        }
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .graphicsLayer {
                translationY = (1f - springEased.value) * 16f
                alpha = springEased.value
            }
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        AuraAiAvatar(
            isThinking = isStreaming,
            isProactive = isProactive,
            size = 36.dp,
        )
        Column(modifier = Modifier.weight(1f)) {
            // Role label
            Text(
                text = "AURA",
                fontFamily = InterDisplay,
                fontWeight = FontWeight.SemiBold,
                fontSize = 12.sp,
                letterSpacing = 0.5.sp,
                color = AuraTokens.Dark.textPrimary,
            )
            if (isProactive) {
                Spacer(Modifier.height(4.dp))
                Surface(
                    color = AuraTokens.Dark.modeAgent,
                    shape = RoundedCornerShape(999.dp),
                ) {
                    Text(
                        text = "Proactive",
                        fontFamily = InterDisplay,
                        fontWeight = FontWeight.Medium,
                        fontSize = 10.sp,
                        color = AuraTokens.Dark.accentPurple,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                    )
                }
            }
            Spacer(Modifier.height(6.dp))
            // Content — streaming text (with the cursor + tok/s from StreamingText)
            StreamingText(
                text = text.ifBlank { "…" },
                isStreaming = isStreaming,
                style = MaterialTheme.typography.bodyLarge.copy(
                    color = AuraTokens.Dark.textPrimary,
                ),
            )
            // Citations row
            if (citations.isNotEmpty() && !isStreaming) {
                Spacer(Modifier.height(8.dp))
                CitationChipRow(citations = citations, onShowSources = onShowSources)
            }
            // Footer: model + timestamp + actions
            if (!isStreaming) {
                Spacer(Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (!modelLabel.isNullOrBlank()) {
                        Text(
                            text = modelLabel,
                            fontFamily = JetBrainsMono,
                            fontSize = 10.sp,
                            color = AuraTokens.Dark.textTertiary,
                        )
                    }
                    if (timestamp > 0) {
                        Text(
                            text = com.aura.ui.util.formatRelativeTime(timestamp),
                            fontFamily = InterDisplay,
                            fontSize = 10.sp,
                            color = AuraTokens.Dark.textQuaternary,
                        )
                    }
                    Spacer(Modifier.weight(1f))
                    // Copy action
                    BubbleAction(
                        icon = androidx.compose.material.icons.Icons.Filled.ContentCopy,
                        label = if (copied) "Copied" else "Copy",
                        onClick = onCopy,
                    )
                }
            }
        }
    }
}

@Composable
private fun CitationChipRow(citations: List<Citation>, onShowSources: () -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        citations.forEach { c ->
            var showDialog by remember { mutableStateOf(false) }
            CitationChip(
                index = c.index,
                onClick = { showDialog = true },
            )
            if (showDialog) {
                val context = LocalContext.current
                androidx.compose.material3.AlertDialog(
                    onDismissRequest = { showDialog = false },
                    title = {
                        Text(
                            text = c.title.ifBlank { "Source ${c.index}" },
                            fontFamily = InterDisplay,
                            fontWeight = FontWeight.SemiBold,
                        )
                    },
                    text = {
                        if (c.url.isNotBlank()) {
                            Text(
                                text = c.url,
                                fontFamily = JetBrainsMono,
                                fontSize = 12.sp,
                                color = AuraTokens.Dark.accentPurple,
                            )
                        }
                    },
                    confirmButton = {
                        if (c.url.isNotBlank()) {
                            androidx.compose.material3.TextButton(onClick = {
                                val intent = android.content.Intent(android.content.Intent.ACTION_VIEW, android.net.Uri.parse(c.url))
                                context.startActivity(intent)
                                showDialog = false
                            }) { Text("Open") }
                        } else {
                            androidx.compose.material3.TextButton(onClick = { showDialog = false }) { Text("Close") }
                        }
                    },
                    dismissButton = {
                        androidx.compose.material3.TextButton(onClick = { showDialog = false }) { Text("Close") }
                    },
                )
            }
        }
    }
}

@Composable
private fun CitationChip(index: Int, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(18.dp)
            .clip(CircleShape)
            .background(
                color = AuraTokens.Dark.glowPurple.copy(alpha = 0.35f),
            )
            .border(
                width = 1.dp,
                color = AuraTokens.Dark.accentPurple.copy(alpha = 0.5f),
                shape = CircleShape,
            )
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = index.toString(),
            fontFamily = InterDisplay,
            fontWeight = FontWeight.Bold,
            fontSize = 10.sp,
            color = AuraTokens.Dark.textPrimary,
        )
    }
}

@Composable
private fun BubbleAction(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(2.dp),
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 4.dp, vertical = 2.dp),
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = AuraTokens.Dark.textTertiary,
            modifier = Modifier.size(14.dp),
        )
        if (label.isNotBlank()) {
            Text(
                text = label,
                fontFamily = InterDisplay,
                fontSize = 10.sp,
                color = AuraTokens.Dark.textTertiary,
            )
        }
    }
}
