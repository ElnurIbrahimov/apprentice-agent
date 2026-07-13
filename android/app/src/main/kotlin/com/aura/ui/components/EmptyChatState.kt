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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Memory
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
import com.aura.ui.theme.AuraThemeTokens
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
    val icon: ImageVector,
)

val DefaultQuickChips: List<QuickChip> = listOf(
    QuickChip(
        prompt = "Research the latest on quantum computing",
        label = "Research",
        icon = Icons.Filled.Search,
    ),
    QuickChip(
        prompt = "Help me write a Python script",
        label = "Code",
        icon = Icons.Filled.Code,
    ),
    QuickChip(
        prompt = "Brainstorm 5 product names for a meditation app",
        label = "Brainstorm",
        icon = Icons.Filled.Lightbulb,
    ),
    QuickChip(
        prompt = "Rewrite this to be more confident: 'I think maybe we could'",
        label = "Rewrite",
        icon = Icons.Filled.SwapHoriz,
    ),
    QuickChip(
        prompt = "Search my memories for anything I saved",
        label = "Memory",
        icon = Icons.Filled.Memory,
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
    val colors = AuraThemeTokens.colors
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier.fillMaxSize(),
    ) {
        // Spacer pushes the welcome down from the top bar by
        // ~25% of the available height. Web has roughly the
        // same offset (the welcome sits between the top bar
        // and the input, biased toward the top).
        Spacer(Modifier.weight(0.6f))
        // Logomark — slightly bigger than the tiny 28dp we had
        // before, so the empty state has a visible anchor.
        AuraLogomark(size = 40.dp)
        Spacer(Modifier.height(16.dp))
        Text(
            text = "Welcome to Aura",
            fontFamily = Fraunces,
            fontWeight = FontWeight.SemiBold,
            fontSize = 28.sp,
            // Brighter than textSecondary (0xFFA1A1AA) so the
            // welcome actually reads on the dark background.
            color = colors.textPrimary,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = "What should we explore?",
            fontFamily = InterDisplay,
            fontWeight = FontWeight.Medium,
            fontSize = 15.sp,
            // Brighter than textTertiary (0xFF6B6B6B) which was
            // near-invisible. textSecondary reads well.
            color = colors.textSecondary,
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
    val colors = AuraThemeTokens.colors
    Box(
        modifier = Modifier
            .size(size)
            .clip(CircleShape)
            .background(colors.selection.copy(alpha = 0.18f))
            .background(
                brush = androidx.compose.ui.graphics.Brush.linearGradient(
                    colors = listOf(
                        colors.selection.copy(alpha = 0.35f),
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
            color = colors.assistantAccent,
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
    val colors = AuraThemeTokens.colors
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        modifier = Modifier
            .clip(RoundedCornerShape(999.dp))
            .background(colors.surface2)
            .border(1.dp, colors.borderSubtle, RoundedCornerShape(999.dp))
            .clickable(onClick = onClick)
            .heightIn(min = 48.dp)
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        Icon(
            imageVector = chip.icon,
            contentDescription = null,
            modifier = Modifier.size(16.dp),
            tint = colors.textSecondary,
        )
        Text(
            text = chip.label,
            fontFamily = InterDisplay,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            color = colors.textSecondary,
        )
    }
}
