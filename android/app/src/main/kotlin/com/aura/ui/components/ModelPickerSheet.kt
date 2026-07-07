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

/**
 * Format a model ID (e.g. "ollama:deepseek-v4-pro:cloud") into a
 * human-friendly display name. Derives everything from the ID —
 * no hardcoded model lists that go stale.
 *
 * Format: "DeepSeek V4 Pro" (capitalized words, suffix stripped)
 */
fun formatModelName(id: String): String {
    val parts = id.split(":", limit = 2)
    val model = parts.getOrNull(1) ?: id
    val provider = parts.getOrNull(0) ?: ""

    // Strip common suffixes like :cloud, :latest
    val clean = model.replace(Regex(":cloud$|:latest$|:free$"), "")

    // Convert kebab/snake to Title Case
    val displayName = clean
        .replace("-", " ")
        .replace("_", " ")
        .split(" ")
        .joinToString(" ") { word ->
            // Keep version numbers as-is (v4, 3.2, 480b)
            if (word.any { it.isDigit() } || word.length <= 2) word.uppercase()
            else word.replaceFirstChar { it.uppercase() }
        }

    val providerLabel = when (provider) {
        "ollama" -> "Ollama"
        "anthropic" -> "Anthropic"
        "openai" -> "OpenAI"
        "deepseek" -> "DeepSeek"
        "gemini" -> "Gemini"
        "groq" -> "Groq"
        "openrouter" -> "OpenRouter"
        "nvidia" -> "NVIDIA"
        "moa" -> "MoA"
        else -> provider.replaceFirstChar { it.uppercase() }
    }

    return "$displayName · $providerLabel"
}

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
                text = "Choose model",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 4.dp),
            )
            Text(
                text = "${models.size} models available",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
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
                            displayName = formatModelName(id),
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
private fun ModelRow(
    id: String,
    displayName: String,
    isCurrent: Boolean,
    onClick: () -> Unit,
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 14.dp, horizontal = 4.dp),
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = displayName,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (isCurrent) FontWeight.SemiBold else FontWeight.Normal,
                color = if (isCurrent) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = id,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
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