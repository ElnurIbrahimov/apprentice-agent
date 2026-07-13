package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraInlineStatus(
    text: kotlin.String,
    modifier: Modifier = Modifier,
    tone: InlineStatusTone = InlineStatusTone.Info,
) {
    val colors = AuraThemeTokens.colors
    val accent = when (tone) {
        InlineStatusTone.Info -> colors.info
        InlineStatusTone.Success -> colors.success
        InlineStatusTone.Warning -> colors.warning
        InlineStatusTone.Error -> colors.error
    }
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(accent.copy(alpha = 0.12f), RoundedCornerShape(AuraDimensions.controlRadius))
            .padding(horizontal = AuraSpacing.sm, vertical = AuraSpacing.xs),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(AuraSpacing.xs),
    ) {
        Box(Modifier.size(8.dp).background(accent, RoundedCornerShape(50)))
        Text(text, style = MaterialTheme.typography.bodySmall, color = colors.textPrimary)
    }
}

enum class InlineStatusTone { Info, Success, Warning, Error }
