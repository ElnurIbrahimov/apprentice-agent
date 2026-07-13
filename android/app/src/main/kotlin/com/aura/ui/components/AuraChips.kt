package com.aura.ui.components

import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraFilterChip(
    selected: Boolean,
    onClick: () -> Unit,
    label: @Composable () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    leadingIcon: (@Composable () -> Unit)? = null,
) {
    val colors = AuraThemeTokens.colors
    FilterChip(
        selected = selected,
        onClick = onClick,
        label = label,
        modifier = modifier.defaultMinSize(minHeight = AuraDimensions.minimumTouchTarget),
        enabled = enabled,
        leadingIcon = leadingIcon,
        colors = FilterChipDefaults.filterChipColors(
            containerColor = colors.surface1,
            labelColor = colors.textSecondary,
            selectedContainerColor = colors.selection,
            selectedLabelColor = colors.textPrimary,
            disabledContainerColor = colors.actionDisabled,
            disabledLabelColor = colors.onActionDisabled,
        ),
        border = FilterChipDefaults.filterChipBorder(
            enabled = enabled,
            selected = selected,
            borderColor = colors.borderDefault,
            selectedBorderColor = colors.borderFocus,
        ),
    )
}
