package com.aura.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraCard(
    modifier: Modifier = Modifier,
    elevated: Boolean = false,
    containerColor: Color = Color.Unspecified,
    contentPadding: PaddingValues = PaddingValues(AuraDimensions.cardPadding),
    content: @Composable ColumnScope.() -> Unit,
) {
    val colors = AuraThemeTokens.colors
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(AuraDimensions.cardRadius),
        color = if (containerColor == Color.Unspecified) {
            if (elevated) colors.surface2 else colors.surface1
        } else {
            containerColor
        },
        border = BorderStroke(1.dp, colors.borderSubtle),
        tonalElevation = if (elevated) 2.dp else 0.dp,
    ) {
        Column(
            modifier = Modifier.padding(contentPadding),
            content = content,
        )
    }
}
