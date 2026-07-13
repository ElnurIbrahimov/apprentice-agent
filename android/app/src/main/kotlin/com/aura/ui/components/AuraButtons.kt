package com.aura.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraPrimaryButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colors = AuraThemeTokens.colors
    Button(
        onClick = onClick,
        modifier = modifier.defaultMinSize(minHeight = AuraDimensions.minimumTouchTarget),
        enabled = enabled,
        shape = RoundedCornerShape(AuraDimensions.controlRadius),
        colors = ButtonDefaults.buttonColors(
            containerColor = colors.actionPrimary,
            contentColor = colors.onActionPrimary,
            disabledContainerColor = colors.actionDisabled,
            disabledContentColor = colors.onActionDisabled,
        ),
        contentPadding = PaddingValues(horizontal = AuraDimensions.cardPadding),
        content = { content() },
    )
}

@Composable
fun AuraSecondaryButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colors = AuraThemeTokens.colors
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.defaultMinSize(minHeight = AuraDimensions.minimumTouchTarget),
        enabled = enabled,
        shape = RoundedCornerShape(AuraDimensions.controlRadius),
        border = BorderStroke(1.dp, colors.borderDefault),
        colors = ButtonDefaults.outlinedButtonColors(
            contentColor = colors.textPrimary,
            disabledContentColor = colors.onActionDisabled,
        ),
        contentPadding = PaddingValues(horizontal = AuraDimensions.cardPadding),
        content = { content() },
    )
}
