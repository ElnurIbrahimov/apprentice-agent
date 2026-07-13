package com.aura.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.TextButton
import androidx.compose.ui.unit.dp
import com.aura.providers.ProviderCredentialState

@Composable
fun ProviderKeyField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    helperText: String? = null,
    modifier: Modifier = Modifier,
    onVerify: (() -> Unit)? = null,
    verifyResult: String? = null,
    verifying: Boolean = false,
    credentialState: ProviderCredentialState? = null,
) {
    var visible by remember { mutableStateOf(false) }
    val testId = label.lowercase().replace(' ', '-')
    Column(modifier = modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
            Text(
                text = label,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.weight(1f),
            )
            val statusLabel = when {
                verifying -> "Testing"
                verifyResult?.startsWith("✓") == true || credentialState == ProviderCredentialState.Valid -> "Verified"
                credentialState == ProviderCredentialState.Invalid -> "Invalid"
                credentialState == ProviderCredentialState.StorageError -> "Storage error"
                credentialState == ProviderCredentialState.Saved -> "Saved · test required"
                value.isNotEmpty() -> "Unsaved draft"
                else -> null
            }
            if (statusLabel != null) {
                Text(
                    text = statusLabel,
                    style = MaterialTheme.typography.labelSmall,
                    color = when (statusLabel) {
                        "Verified" -> MaterialTheme.colorScheme.primary
                        "Invalid", "Storage error" -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
            }
        }
        Spacer(modifier = Modifier.height(4.dp))
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.fillMaxWidth().testTag("provider-key-$testId"),
            placeholder = { Text("Paste API key") },
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
        if (helperText != null) {
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = helperText,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
            )
        }
        if (onVerify != null) {
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                TextButton(
                    onClick = onVerify,
                    enabled = !verifying,
                    modifier = Modifier.testTag("provider-test-$testId"),
                ) {
                    if (verifying) {
                        CircularProgressIndicator(modifier = Modifier.height(16.dp), strokeWidth = 2.dp)
                    } else {
                        Text("Save & Test")
                    }
                }
                if (verifyResult != null) {
                    Text(
                        text = verifyResult,
                        style = MaterialTheme.typography.labelSmall,
                        color = if (verifyResult.startsWith("✓"))
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.error,
                    )
                }
            }
        }
    }
}
