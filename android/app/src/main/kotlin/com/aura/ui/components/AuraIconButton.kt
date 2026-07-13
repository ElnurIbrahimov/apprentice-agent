package com.aura.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens

/** A 48dp semantic hit target containing a compact 40dp visual control. */
@Composable
fun AuraIconButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    containerColor: Color = Color.Transparent,
    content: @Composable () -> Unit,
) {
    val colors = AuraThemeTokens.colors
    Box(
        modifier = modifier
            .size(AuraDimensions.minimumTouchTarget)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            modifier = Modifier.size(AuraDimensions.iconButtonVisualSize),
            shape = RoundedCornerShape(AuraDimensions.controlRadius),
            color = if (enabled) containerColor else colors.actionDisabled,
        ) {
            Box(contentAlignment = Alignment.Center) { content() }
        }
    }
}
