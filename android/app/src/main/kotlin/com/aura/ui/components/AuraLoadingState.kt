package com.aura.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraLoadingState(
    modifier: Modifier = Modifier,
    label: kotlin.String = "Loading…",
    rows: Int = 4,
) {
    val colors = AuraThemeTokens.colors
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = colors.textSecondary)
        repeat(rows.coerceAtLeast(1)) { index ->
            AuraSkeleton(
                height = if (index == 0) 72.dp else 56.dp,
                widthFraction = if (index % 3 == 2) 0.72f else 1f,
                testTag = "skeleton-row-$index",
            )
        }
    }
}
