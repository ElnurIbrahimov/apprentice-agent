package com.aura.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Inbox
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraEmptyState(
    title: kotlin.String,
    message: kotlin.String,
    actionLabel: kotlin.String,
    onAction: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector = Icons.Outlined.Inbox,
) {
    AuraStatePane(
        title = title,
        message = message,
        icon = icon,
        modifier = modifier,
        actionLabel = actionLabel,
        onAction = onAction,
        isError = false,
    )
}

@Composable
internal fun AuraStatePane(
    title: kotlin.String,
    message: kotlin.String,
    icon: ImageVector,
    modifier: Modifier,
    actionLabel: kotlin.String,
    onAction: () -> Unit,
    isError: Boolean,
) {
    val colors = AuraThemeTokens.colors
    Column(
        modifier = modifier.fillMaxWidth().padding(AuraSpacing.lg),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = if (isError) colors.error else colors.textTertiary,
            modifier = Modifier.size(32.dp),
        )
        Text(title, style = MaterialTheme.typography.titleLarge, color = colors.textPrimary, textAlign = TextAlign.Center)
        Text(message, style = MaterialTheme.typography.bodyMedium, color = colors.textSecondary, textAlign = TextAlign.Center)
        AuraPrimaryButton(onClick = onAction) { Text(actionLabel) }
    }
}
