package com.aura.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelPickerSheet(
    currentModel: String,
    models: List<String>,
    onPick: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = MaterialTheme.colorScheme.surface,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 8.dp),
        ) {
            Text(
                text = "Pick a model",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(bottom = 12.dp),
            )
            if (models.isEmpty()) {
                Text(
                    text = "No models available. Add a provider API key in Settings.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    modifier = Modifier.padding(vertical = 24.dp),
                )
            } else {
                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    items(models) { id ->
                        ModelRow(
                            id = id,
                            isCurrent = id == currentModel,
                            onClick = {
                                onPick(id)
                                onDismiss()
                            },
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ModelRow(id: String, isCurrent: Boolean, onClick: () -> Unit) {
    val parts = id.split(":", limit = 2)
    val provider = parts.getOrNull(0) ?: "?"
    val model = parts.getOrNull(1) ?: id
    val displayName = when (id) {
        "ollama:deepseek-v3.2:cloud" -> "DeepSeek V3.2 — cheap, fast, smart"
        "ollama:kimi-k2.6:cloud" -> "Kimi K2.6 — best tool calls"
        "anthropic:claude-sonnet-4-5" -> "Claude Sonnet 4.5 — best overall"
        "ollama:minimax-m2.7:cloud" -> "MiniMax M2.7 — code specialist"
        "ollama:qwen3-coder:480b-cloud" -> "Qwen3 Coder 480B — heavy code"
        else -> "$model ($provider)"
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 12.dp, horizontal = 4.dp),
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = displayName,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (isCurrent) FontWeight.SemiBold else FontWeight.Normal,
            )
            Text(
                text = id,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
            )
        }
        if (isCurrent) {
            Icon(
                imageVector = Icons.Filled.Check,
                contentDescription = "current",
                tint = MaterialTheme.colorScheme.primary,
            )
        }
    }
}
