package com.aura.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp

import com.aura.ui.theme.AuraThemeTokens
/**
 * Card for the "Custom Endpoint" provider. The user supplies a base URL
 * (e.g. https://api.example.com/v1) and an API key; the card persists
 * both to the secure DataStore via [SettingsViewModel] and exposes a
 * "Test connection" button that hits the live `/models` endpoint.
 *
 * This card replaces the "Custom" entry in [ProviderKeyField] for the
 * one prefix that needs TWO fields, not one.
 */
@Composable
fun CustomEndpointCard(
    baseUrl: String,
    apiKey: String,
    isConfigured: Boolean,
    testing: Boolean,
    result: String?,
    onBaseUrlChange: (String) -> Unit,
    onApiKeyChange: (String) -> Unit,
    onTest: () -> Unit,
    onClear: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var visible by remember { mutableStateOf(false) }
    Surface(
        color = AuraThemeTokens.colors.surface1.copy(alpha = 0.4f),
        shape = RoundedCornerShape(14.dp),
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .testTag("custom-endpoint-card"),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Custom Endpoint",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                val status = when {
                    testing -> "Testing"
                    isConfigured -> "Configured"
                    baseUrl.isNotBlank() || apiKey.isNotBlank() -> "Unsaved"
                    else -> null
                }
                if (status != null) {
                    Text(
                        text = status,
                        style = MaterialTheme.typography.labelSmall,
                        color = when (status) {
                            "Configured" -> AuraThemeTokens.colors.actionPrimary
                            "Testing" -> AuraThemeTokens.colors.textPrimary
                            else -> AuraThemeTokens.colors.error
                        },
                    )
                }
            }
            Spacer(Modifier.height(2.dp))
            Text(
                text = "Any OpenAI-compatible chat-completions URL. " +
                    "Models are pulled from <URL>/models unless you provide a static list.",
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
            )
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = baseUrl,
                onValueChange = onBaseUrlChange,
                label = { Text("Base URL") },
                placeholder = { Text("https://api.example.com/v1") },
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("custom-endpoint-url"),
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            )
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = apiKey,
                onValueChange = onApiKeyChange,
                label = { Text("API key") },
                placeholder = { Text("sk-…") },
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("custom-endpoint-key"),
                singleLine = true,
                visualTransformation = if (visible) VisualTransformation.None else PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                trailingIcon = {
                    IconButton(onClick = { visible = !visible }) {
                        Icon(
                            imageVector = if (visible) Icons.Filled.VisibilityOff else Icons.Filled.Visibility,
                            contentDescription = if (visible) "Hide" else "Show",
                        )
                    }
                },
            )
            Spacer(Modifier.height(8.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                TextButton(
                    onClick = onTest,
                    enabled = !testing && baseUrl.isNotBlank() && apiKey.isNotBlank(),
                    modifier = Modifier.testTag("custom-endpoint-test"),
                ) {
                    if (testing) {
                        CircularProgressIndicator(modifier = Modifier.height(16.dp), strokeWidth = 2.dp)
                    } else {
                        Text("Save & Test")
                    }
                }
                if (isConfigured) {
                    TextButton(onClick = onClear) {
                        Text("Clear", color = AuraThemeTokens.colors.error)
                    }
                }
                result?.let { msg ->
                    Text(
                        text = msg,
                        style = MaterialTheme.typography.labelSmall,
                        color = when {
                            msg.startsWith("✓") -> AuraThemeTokens.colors.actionPrimary
                            msg.startsWith("✗") -> AuraThemeTokens.colors.error
                            else -> AuraThemeTokens.colors.textPrimary
                        },
                    )
                }
            }
        }
    }
}
