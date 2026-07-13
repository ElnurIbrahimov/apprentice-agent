package com.aura.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Fingerprint
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun AuraStartupState(modifier: Modifier = Modifier) {
    val colors = AuraThemeTokens.colors
    Box(
        modifier = modifier
            .fillMaxSize()
            .testTag("aura-startup"),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 280.dp)
                .padding(AuraSpacing.lg)
                .testTag("aura-startup-content"),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            Surface(
                modifier = Modifier
                    .size(56.dp)
                    .testTag("aura-startup-mark"),
                shape = CircleShape,
                color = colors.actionPrimary.copy(alpha = 0.14f),
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = Icons.Filled.AutoAwesome,
                        contentDescription = null,
                        tint = colors.actionPrimary,
                        modifier = Modifier.size(28.dp),
                    )
                }
            }
            Text(
                text = "AURA",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 3.sp,
                color = colors.textPrimary,
            )
            Text(
                text = "Preparing your space…",
                style = MaterialTheme.typography.bodyMedium,
                color = colors.textSecondary,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
internal fun AuraAppLockContent(
    statusMessage: String?,
    onUnlock: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = AuraThemeTokens.colors
    Box(
        modifier = modifier
            .fillMaxSize()
            .testTag("aura-app-lock"),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .padding(horizontal = AuraSpacing.lg)
                .testTag("app-lock-content"),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
        ) {
            Surface(
                modifier = Modifier.size(72.dp),
                shape = CircleShape,
                color = colors.actionPrimary.copy(alpha = 0.14f),
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = Icons.Filled.Lock,
                        contentDescription = null,
                        tint = colors.actionPrimary,
                        modifier = Modifier.size(34.dp),
                    )
                }
            }
            Text(
                text = "Aura is locked",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.SemiBold,
                color = colors.textPrimary,
                textAlign = TextAlign.Center,
            )
            Text(
                text = "Authenticate to open your conversations.",
                style = MaterialTheme.typography.bodyMedium,
                color = colors.textSecondary,
                textAlign = TextAlign.Center,
            )
            statusMessage?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.error,
                    textAlign = TextAlign.Center,
                )
            }
            Spacer(Modifier.height(AuraSpacing.xs))
            AuraPrimaryButton(
                onClick = onUnlock,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("app-lock-action"),
            ) {
                Icon(Icons.Filled.Fingerprint, contentDescription = null)
                Spacer(Modifier.width(AuraSpacing.xs))
                Text("Unlock")
            }
        }
    }
}
