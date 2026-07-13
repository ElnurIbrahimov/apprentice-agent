package com.aura.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

@Composable
fun AuraErrorState(
    title: kotlin.String,
    message: kotlin.String,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
    retryLabel: kotlin.String = "Try again",
) {
    AuraStatePane(
        title = title,
        message = message,
        icon = Icons.Outlined.ErrorOutline,
        modifier = modifier,
        actionLabel = retryLabel,
        onAction = onRetry,
        isError = true,
    )
}
