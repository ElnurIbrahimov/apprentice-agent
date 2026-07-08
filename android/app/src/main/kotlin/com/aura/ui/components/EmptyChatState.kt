package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aura.ui.theme.AuraTokens
import com.aura.ui.theme.Fraunces
import com.aura.ui.theme.InterDisplay

/**
 * Quick-action chips shown BELOW the input bar in the empty
 * chat state. Web's "starter chips" — small 28dp round icon +
 * 11sp label, all on a 1-row horizontal scroller. Tap a chip
 * to send its prompt.
 *
 * Sized for the input bar's horizontal padding (10dp) so the
 * chips align visually with the input above.
 */
data class QuickChip(
    val prompt: String,
    val label: String,
    val emoji: String,
    val icon: ImageVector? = null,
)

val DefaultQuickChips: List<QuickChip> = listOf(
    QuickChip(
        prompt = "Research the latest on quantum computing",
        label = "Research",
        emoji = "🔍",
    ),
    QuickChip(
        prompt = "Help me write a Python script",
        label = "Code",
        emoji = "💻",
    ),
    QuickChip(
        prompt = "Brainstorm 5 product names for a meditation app",
        label = "Brainstorm",
        emoji = "💡",
    ),
    QuickChip(
        prompt = "Rewrite this to be more confident: 'I think maybe we could'",
        label = "Rewrite",
        emoji = "✍️",
    ),
    QuickChip(
        prompt = "Search my memories for anything I saved",
        label = "Memory",
        emoji = "🧠",
    ),
)

/**
 * Empty-state hero for the chat screen. Shown when there are
 * zero conversation turns. Matches Aura Web's empty state:
 * a small logomark (28dp) above a Fraunces H1 welcome line,
 * with a row of small quick-action chips below the input.
 *
 * The chips sit OUTSIDE the hero (they live in the input bar
 * area in the web). The hero itself is just the logomark +
 * welcome text — no breathing glow, no card grid. The chat
 * is the hero; the input is always reachable.
 */
@Composable
fun EmptyChatState(
    modifier: Modifier = Modifier,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier.fillMaxSize(),
    ) {
        // Spacer pushes the welcome down from the top bar by
        // ~25% of the available height. Web has roughly the
        // same offset (the welcome sits between the top bar
        // and the input, biased toward the top).
        Spacer(Modifier.weight(0.6f))
        AuraLogomark(size = 28.dp)
        Spacer(Modifier.height(14.dp))
        Text(
            text = "Welcome",
            fontFamily = Fraunces,
            fontWeight = FontWeight.Normal,
            fontSize = 28.sp,
            color = AuraTokens.Dark.textSecondary,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = "What should we explore?",
            fontFamily = InterDisplay,
            fontWeight = FontWeight.Medium,
            fontSize = 14.sp,
            color = AuraTokens.Dark.textTertiary,
        )
        // Push the rest of the screen to the bottom so the
        // input bar (rendered separately by ChatScreen) lines
        // up with where the user expects to see it.
        Spacer(Modifier.weight(1f))
        // (chips are rendered by ChatScreen below the input bar — not here)
    }
}

/**
 * Small Aura logomark for the empty state. The web uses a
 * 28px square gradient with the "A" character; on Android
 * we render a 28dp circle with the Aura "✦" glyph in the
 * brand violet, surrounded by a faint 1px border.
 */
@Composable
fun AuraLogomark(size: androidx.compose.ui.unit.Dp = 28.dp) {
    Box(
        modifier = Modifier
            .size(size)
            .clip(CircleShape)
            .background(AuraTokens.Dark.glowPurple.copy(alpha = 0.18f))
            .background(
                brush = androidx.compose.ui.graphics.Brush.linearGradient(
                    colors = listOf(
                        AuraTokens.Dark.glowPurple.copy(alpha = 0.35f),
                        Color.Transparent,
                    ),
                ),
            ),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = "✦",
            fontFamily = Fraunces,
            fontSize = (size.value * 0.6f).sp,
            color = AuraTokens.Dark.accentPurple,
        )
    }
}

@Composable
fun QuickChipRow(
    chips: List<QuickChip> = DefaultQuickChips,
    onPick: (String) -> Unit,
) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        contentPadding = PaddingValues(horizontal = 10.dp),
    ) {
        items(chips.size) { i ->
            QuickChipView(chip = chips[i], onClick = { onPick(chips[i].prompt) })
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
private fun QuickChipView(chip: QuickChip, onClick: () -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(AuraTokens.Dark.surface2)
            .border(1.dp, AuraTokens.Dark.borderSubtle, RoundedCornerShape(999.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 5.dp),
    ) {
        Text(
            text = chip.emoji,
            fontSize = 12.sp,
        )
        Text(
            text = chip.label,
            fontFamily = InterDisplay,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            color = AuraTokens.Dark.textSecondary,
        )
    }
}
