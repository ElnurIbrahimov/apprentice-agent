package com.aura.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aura.ui.theme.AuraTokens
import com.aura.ui.theme.Fraunces
import com.aura.ui.theme.InterDisplay

/**
 * Quick-action prompts shown in the empty chat state. Tapping
 * one sets the draft and focuses the input — same as the Aura
 * Web `QUICK_ACTIONS` cards in `ChatContainer.tsx`.
 *
 * The list is intentionally short (5 items) and biased toward
 * the cases the user reaches for first: research, code, idea,
 * rewrite, search. Each action has a single icon and a two-line
 * label: the prompt and a sub-label describing what Aura will do.
 */
data class QuickAction(
    val prompt: String,
    val title: String,
    val subtitle: String,
    val icon: ImageVector,
    val accent: Color,
)

val DefaultQuickActions: List<QuickAction> = listOf(
    QuickAction(
        prompt = "Research the latest on quantum computing and write a 1-page summary",
        title = "Research a topic",
        subtitle = "Deep dive with citations",
        icon = Icons.Filled.Search,
        accent = AuraTokens.Dark.modeResearch,
    ),
    QuickAction(
        prompt = "Help me write a Python script that watches a folder for new files",
        title = "Write code",
        subtitle = "Real working snippets",
        icon = Icons.Filled.Code,
        accent = AuraTokens.Dark.modeAgent,
    ),
    QuickAction(
        prompt = "Brainstorm 5 product names for a meditation app for teens",
        title = "Brainstorm an idea",
        subtitle = "Multiple options + rationale",
        icon = Icons.Filled.Lightbulb,
        accent = AuraTokens.Dark.modeDelegate,
    ),
    QuickAction(
        prompt = "Rewrite this paragraph to sound more confident: 'I think maybe we could try the new design'",
        title = "Rewrite text",
        subtitle = "Tone, length, clarity",
        icon = Icons.Filled.SwapHoriz,
        accent = AuraTokens.Dark.modeCompare,
    ),
    QuickAction(
        prompt = "Search my memories for anything I saved about travel in Japan",
        title = "Search memories",
        subtitle = "Recall what you saved",
        icon = Icons.Filled.Search,
        accent = AuraTokens.Dark.modeDeepResearch,
    ),
)

/**
 * Empty-state hero for the chat screen. Shown when there are
 * zero conversation turns. Matches Aura Web's
 * `<h1>What should we explore?</h1>` empty state in
 * `ChatContainer.tsx`.
 *
 * The hero has a breathing radial-purple glow behind the title,
 * a Fraunces-serif H1, an Inter subtitle, and a horizontal
 * scroller of quick-action cards. Each card has its own accent
 * color matching the Web's mode palette.
 *
 * Cards use a spring-up entry animation staggered by index.
 * The hero glow has a slow `infiniteRepeatable` scale + alpha
 * animation to feel "alive" without being distracting.
 */
@Composable
fun EmptyChatState(
    onPickQuickAction: (String) -> Unit,
    actions: List<QuickAction> = DefaultQuickActions,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 32.dp),
        ) {
            BreathingGlow()
            Spacer(Modifier.height(40.dp))
            Text(
                text = "What should we explore?",
                fontFamily = Fraunces,
                fontWeight = FontWeight.Normal,
                fontSize = 36.sp,
                lineHeight = 44.sp,
                letterSpacing = (-0.8).sp,
                color = AuraTokens.Dark.textPrimary,
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = "Research, create, code, and compare — all in one place.",
                fontFamily = InterDisplay,
                fontSize = 14.sp,
                color = AuraTokens.Dark.textSecondary,
            )
            Spacer(Modifier.height(32.dp))
            QuickActionRow(actions = actions, onPick = onPickQuickAction)
        }
    }
}

@Composable
private fun BreathingGlow() {
    val transition = rememberInfiniteTransition(label = "hero-glow")
    val scale by transition.animateFloat(
        initialValue = 1f,
        targetValue = 1.18f,
        animationSpec = infiniteRepeatable(
            animation = tween(3600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "scale",
    )
    val alpha by transition.animateFloat(
        initialValue = 0.18f,
        targetValue = 0.32f,
        animationSpec = infiniteRepeatable(
            animation = tween(3600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "alpha",
    )
    Box(
        modifier = Modifier
            .size(220.dp)
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
                this.alpha = alpha
            }
            .background(
                brush = Brush.radialGradient(
                    colors = listOf(
                        AuraTokens.Dark.glowPurple,
                        Color.Transparent,
                    ),
                ),
                shape = CircleShape,
            ),
    )
}

@Composable
private fun QuickActionRow(
    actions: List<QuickAction>,
    onPick: (String) -> Unit,
) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 4.dp),
    ) {
        items(actions.size) { i ->
            val action = actions[i]
            val entry = remember { androidx.compose.animation.core.Animatable(0f) }
            androidx.compose.runtime.LaunchedEffect(Unit) {
                kotlinx.coroutines.delay(i * 60L)
                entry.animateTo(
                    targetValue = 1f,
                    animationSpec = androidx.compose.animation.core.spring(
                        dampingRatio = 0.7f,
                        stiffness = androidx.compose.animation.core.Spring.StiffnessMediumLow,
                    ),
                )
            }
            QuickActionCard(
                action = action,
                onClick = { onPick(action.prompt) },
                progress = entry.value,
            )
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.items(
    count: Int,
    content: @Composable (Int) -> Unit,
) {
    items(count = count, key = null, itemContent = { i -> content(i) })
}

@Composable
private fun QuickActionCard(
    action: QuickAction,
    onClick: () -> Unit,
    progress: Float,
) {
    Box(
        modifier = Modifier
            .widthIn(min = 180.dp, max = 220.dp)
            .graphicsLayer {
                translationY = (1f - progress) * 16f
                alpha = progress
            }
            .clip(RoundedCornerShape(12.dp))
            .background(AuraTokens.Dark.surface1)
            .clickable(onClick = onClick)
            .padding(14.dp),
    ) {
        Column {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(action.accent),
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    imageVector = action.icon,
                    contentDescription = null,
                    tint = AuraTokens.Dark.textPrimary,
                    modifier = Modifier.size(18.dp),
                )
            }
            Spacer(Modifier.height(10.dp))
            Text(
                text = action.title,
                fontFamily = InterDisplay,
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp,
                color = AuraTokens.Dark.textPrimary,
            )
            Spacer(Modifier.height(2.dp))
            Text(
                text = action.subtitle,
                fontFamily = InterDisplay,
                fontSize = 12.sp,
                color = AuraTokens.Dark.textTertiary,
            )
        }
    }
}
