package com.aura.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Format a model ID (e.g. "ollama:deepseek-v4-pro:cloud") into a
 * human-friendly display name. Derives everything from the ID —
 * no hardcoded model lists that go go stale.
 */
fun formatModelName(id: String): String {
    val parts = id.split(":", limit = 2)
    val model = parts.getOrNull(1) ?: id
    val provider = parts.getOrNull(0) ?: ""

    val clean = model.replace(Regex(":cloud$|:latest$|:free$"), "")
    val displayName = clean
        .replace("-", " ")
        .replace("_", " ")
        .split(" ")
        .joinToString(" ") { word ->
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

private fun providerLabel(prefix: String): String = com.aura.providers.providerLabel(prefix)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelPickerSheet(
    currentModel: String,
    models: List<String>,
    isLoading: Boolean = false,
    errorMessage: String? = null,
    staleProviderPrefixes: Set<String> = emptySet(),
    /** Explains whether this picker changes chat-only state or the global default. */
    selectionScopeLabel: String? = null,
    /** Optional explicit promotion from a chat override to the global default. */
    onMakeDefault: (() -> Unit)? = null,
    onPick: (String) -> Unit,
    onRefresh: () -> Unit = {},
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = MaterialTheme.colorScheme.surface,
    ) {
        ModelPickerContent(
            currentModel = currentModel,
            models = models,
            isLoading = isLoading,
            errorMessage = errorMessage,
            staleProviderPrefixes = staleProviderPrefixes,
            selectionScopeLabel = selectionScopeLabel,
            onMakeDefault = onMakeDefault,
            onPick = onPick,
            onRefresh = onRefresh,
            onDismiss = onDismiss,
        )
    }
}

/** Picker body without the focus-owning modal window, for deterministic UI tests. */
@Composable
fun ModelPickerContent(
    currentModel: String,
    models: List<String>,
    isLoading: Boolean = false,
    errorMessage: String? = null,
    staleProviderPrefixes: Set<String> = emptySet(),
    selectionScopeLabel: String? = null,
    onMakeDefault: (() -> Unit)? = null,
    onPick: (String) -> Unit,
    onRefresh: () -> Unit = {},
    onDismiss: () -> Unit,
) {
    var query by remember { mutableStateOf("") }

    // Group by provider, filter by search query
    val grouped = remember(models, query) {
        val filtered = if (query.isBlank()) {
            models
        } else {
            val q = query.trim().lowercase()
            models.filter { it.lowercase().contains(q) }
        }
        filtered
            .groupBy { it.substringBefore(":", missingDelimiterValue = "other") }
            .toSortedMap()
    }
    val totalCount = grouped.values.sumOf { it.size }
    val currentUnavailable = currentModel.isNotBlank() && currentModel !in models

    Column(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 320.dp)
                .testTag("model-picker")
                .padding(horizontal = 20.dp, vertical = 8.dp),
        ) {
            // Header row: title + model count
            Row(
                modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Choose model",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = if (query.isBlank()) "$totalCount models available"
                               else "${grouped.values.sumOf { it.size }} of $totalCount match",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                    )
                    selectionScopeLabel?.let { label ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = label,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                            )
                            onMakeDefault?.let { promote ->
                                TextButton(onClick = promote, contentPadding = PaddingValues(start = 8.dp)) {
                                    Text("Make default", style = MaterialTheme.typography.labelSmall)
                                }
                            }
                        }
                    }
                }
                IconButton(
                    onClick = onRefresh,
                    enabled = !isLoading,
                ) {
                    Icon(
                        imageVector = Icons.Filled.Refresh,
                        contentDescription = "Refresh models",
                        // Subtle spin while loading
                        modifier = if (isLoading) Modifier
                            .size(20.dp)
                            .rotate(0f)  // static; full spin would need a LaunchedEffect
                        else Modifier.size(20.dp),
                    )
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Filled.Close, contentDescription = "Close")
                }
            }

            // Search field
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("model-search")
                    .padding(bottom = 12.dp),
                placeholder = { Text("Search models…", style = MaterialTheme.typography.bodyMedium) },
                leadingIcon = {
                    Icon(
                        Icons.Filled.Search,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                    )
                },
                singleLine = true,
                shape = RoundedCornerShape(16.dp),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                ),
                textStyle = MaterialTheme.typography.bodyMedium,
            )

            if (isLoading && models.isEmpty()) {
                // First-load loading state
                Column(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(28.dp),
                        strokeWidth = 2.dp,
                    )
                    Spacer(Modifier.height(12.dp))
                    Text(
                        text = "Loading models from your providers…",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    )
                }
            } else if (models.isEmpty() && errorMessage != null) {
                // Surface the error so the user knows it's a network/key problem,
                // not "the app doesn't have any models"
                Column(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = "Couldn't load models",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = errorMessage,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = "Tap refresh, or check the API key in Settings.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    )
                }
            } else if (models.isEmpty()) {
                Text(
                    text = "No verified models available. Save & Test a provider in Settings.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    modifier = Modifier.padding(vertical = 24.dp),
                )
            } else if (grouped.isEmpty()) {
                Text(
                    text = "No models match \"$query\"",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    modifier = Modifier.padding(vertical = 24.dp),
                )
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(bottom = 8.dp),
                ) {
                    if (errorMessage != null) {
                        item(key = "models-warning") {
                            Surface(
                                color = MaterialTheme.colorScheme.errorContainer,
                                shape = RoundedCornerShape(14.dp),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(bottom = 12.dp),
                            ) {
                                Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
                                    Text(
                                        text = "Last refresh failed",
                                        style = MaterialTheme.typography.labelLarge,
                                        fontWeight = FontWeight.SemiBold,
                                        color = MaterialTheme.colorScheme.onErrorContainer,
                                    )
                                    Spacer(Modifier.height(4.dp))
                                    Text(
                                        text = errorMessage,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onErrorContainer,
                                    )
                                }
                            }
                        }
                    }
                    if (currentUnavailable) {
                        item(key = "unavailable-model-header") {
                            ProviderHeader(name = "Current selection", count = 1)
                        }
                        item(key = "unavailable-model-row") {
                            ModelRow(
                                id = currentModel,
                                displayName = formatModelName(currentModel),
                                isCurrent = true,
                                enabled = false,
                                statusLabel = "Unavailable",
                                onClick = {},
                            )
                        }
                    }
                    grouped.forEach { (provider, items) ->
                        item(key = "header-$provider") {
                            ProviderHeader(
                                name = providerLabel(provider),
                                count = items.size,
                            )
                        }
                        items(items, key = { it }) { id ->
                            ModelRow(
                                id = id,
                                displayName = formatModelName(id),
                                isCurrent = id == currentModel,
                                statusLabel = if (provider in staleProviderPrefixes) "Cached" else null,
                                onClick = {
                                    onPick(id)
                                    onDismiss()
                                },
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
    }
}

@Composable
private fun ProviderHeader(name: String, count: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 16.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = name,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.primary,
        )
        Spacer(modifier = Modifier.size(8.dp))
        Surface(
            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
            shape = RoundedCornerShape(8.dp),
        ) {
            Text(
                text = "$count",
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 1.dp),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
            )
        }
    }
}

@Composable
private fun ModelRow(
    id: String,
    displayName: String,
    isCurrent: Boolean,
    enabled: Boolean = true,
    statusLabel: String? = null,
    onClick: () -> Unit,
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .widthIn(min = 0.dp)
            .testTag("model-row-$id")
            .clickable(enabled = enabled, onClick = onClick)
            .padding(vertical = 12.dp, horizontal = 4.dp),
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = displayName,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (isCurrent) FontWeight.SemiBold else FontWeight.Normal,
                color = when {
                    !enabled -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f)
                    isCurrent -> MaterialTheme.colorScheme.primary
                    else -> MaterialTheme.colorScheme.onSurface
                },
            )
            Text(
                text = id,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
            )
        }
        statusLabel?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.labelSmall,
                color = if (enabled) MaterialTheme.colorScheme.tertiary
                    else MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 8.dp),
            )
        }
        if (isCurrent && enabled) {
            Box(
                modifier = Modifier.size(28.dp),
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    imageVector = Icons.Filled.Check,
                    contentDescription = "current",
                    tint = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}