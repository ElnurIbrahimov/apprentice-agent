package com.aura.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.material3.TimePicker
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.settings.BackupViewModel
import com.aura.ui.settings.ProviderKeyField
import com.aura.ui.settings.SettingsViewModel
import com.aura.ui.settings.UsageViewModel
import com.aura.ui.util.modelDisplayName
import kotlinx.coroutines.launch

// ────────────────────────────────────────────────────────────
// Reusable collapsible section card
// ────────────────────────────────────────────────────────────

@Composable
private fun SettingsSection(
    emoji: String,
    title: String,
    subtitle: String,
    initialExpanded: Boolean,
    content: @Composable () -> Unit,
) {
    var expanded by remember { mutableStateOf(initialExpanded) }

    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.40f),
        shape = RoundedCornerShape(14.dp),
        tonalElevation = 0.dp,
        modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp),
    ) {
        Column(modifier = Modifier.animateContentSize()) {
            // ── Header row (clickable to toggle) ──
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded }
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text = emoji, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    if (subtitle.isNotEmpty()) {
                        Text(
                            text = subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
                        )
                    }
                }
                Icon(
                    imageVector = if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                    contentDescription = if (expanded) "Collapse" else "Expand",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                    modifier = Modifier.padding(start = 4.dp),
                )
            }

            // ── Animated content ──
            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically(),
                exit = shrinkVertically(),
            ) {
                Column(
                    modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 14.dp),
                ) {
                    HorizontalDivider(
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f),
                        modifier = Modifier.padding(bottom = 10.dp),
                    )
                    content()
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// Main Settings screen
// ────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class, androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateProfile: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel(),
    backupViewModel: BackupViewModel = hiltViewModel(),
    usageViewModel: UsageViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val usage by usageViewModel.usage.collectAsState()
    var showDefaultModelPicker by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp),
    ) {
        // ── Header ──────────────────────────────────────────────────────
        Column(modifier = Modifier.padding(top = 16.dp, bottom = 6.dp)) {
            Text(
                text = "Settings",
                style = MaterialTheme.typography.displaySmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Connect providers, manage memory, customize Aura",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
        }

        Spacer(modifier = Modifier.height(4.dp))

        // ════════════════════════════════════════════════════════════════
        // 1. AI & MODELS — always shows status; keys + model picker below
        // ════════════════════════════════════════════════════════════════
        SettingsSection(
            emoji = "\uD83E\uDD16",
            title = "AI & Models",
            subtitle = "Providers, API keys, default and embedding models",
            initialExpanded = true,
        ) {
            // ── Provider status banner ──
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        text = if (state.configuredProviders.isEmpty())
                            "No providers configured yet."
                        else
                            "Configured providers:",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Medium,
                    )
                    if (state.configuredProviders.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(4.dp))
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            state.configuredProviders.forEach { name ->
                                AssistChip(
                                    onClick = {},
                                    label = { Text(name) },
                                    colors = AssistChipDefaults.assistChipColors(
                                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                                        labelColor = MaterialTheme.colorScheme.onPrimaryContainer,
                                    ),
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── API Keys section header ──
            Text(
                text = "API Keys",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Stored locally, never leave your device. Changes take effect immediately.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(4.dp))

            ProviderKeyField(
                label = "Ollama Cloud",
                value = state.ollamaKey,
                onValueChange = viewModel::saveOllamaKey,
                helperText = "Get a key at ollama.com/settings/keys",
                onVerify = { viewModel.verifyKey("ollama") },
                verifyResult = state.verifyResults["ollama"],
                verifying = state.verifying == "ollama",
            )
            ProviderKeyField(
                label = "Anthropic",
                value = state.anthropicKey,
                onValueChange = viewModel::saveAnthropicKey,
                helperText = "Get a key at console.anthropic.com/settings/keys",
                onVerify = { viewModel.verifyKey("anthropic") },
                verifyResult = state.verifyResults["anthropic"],
                verifying = state.verifying == "anthropic",
            )
            ProviderKeyField(
                label = "OpenAI",
                value = state.openaiKey,
                onValueChange = viewModel::saveOpenaiKey,
                helperText = "Get a key at platform.openai.com/api-keys",
                onVerify = { viewModel.verifyKey("openai") },
                verifyResult = state.verifyResults["openai"],
                verifying = state.verifying == "openai",
            )
            ProviderKeyField(
                label = "DeepSeek",
                value = state.deepseekKey,
                onValueChange = viewModel::saveDeepseekKey,
                helperText = "Get a key at platform.deepseek.com/api_keys",
                onVerify = { viewModel.verifyKey("deepseek") },
                verifyResult = state.verifyResults["deepseek"],
                verifying = state.verifying == "deepseek",
            )
            ProviderKeyField(
                label = "Groq",
                value = state.groqKey,
                onValueChange = viewModel::saveGroqKey,
                helperText = "Get a key at console.groq.com/keys",
                onVerify = { viewModel.verifyKey("groq") },
                verifyResult = state.verifyResults["groq"],
                verifying = state.verifying == "groq",
            )
            ProviderKeyField(
                label = "OpenRouter",
                value = state.openrouterKey,
                onValueChange = viewModel::saveOpenrouterKey,
                helperText = "Get key at openrouter.ai/keys",
                onVerify = { viewModel.verifyKey("openrouter") },
                verifyResult = state.verifyResults["openrouter"],
                verifying = state.verifying == "openrouter",
            )

            Spacer(modifier = Modifier.height(12.dp))

            // ── Default model picker ──
            Text(
                text = "Default model",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Default: ${state.defaultModel}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = modelDisplayName(state.defaultModel),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(6.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    onClick = {
                        showDefaultModelPicker = true
                        viewModel.refreshModels()
                    },
                ) {
                    Text("Choose model")
                }
                if (state.modelsError == null && state.availableModels.isNotEmpty()) {
                    Text(
                        text = "${state.availableModels.size} live models loaded",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    )
                } else if (state.modelsError != null) {
                    Text(
                        text = "Model refresh failed — open the picker for details",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
            if (showDefaultModelPicker) {
                ModelPickerSheet(
                    currentModel = state.defaultModel,
                    models = state.availableModels,
                    isLoading = state.modelsLoading,
                    errorMessage = state.modelsError,
                    onPick = viewModel::setDefaultModel,
                    onRefresh = { viewModel.refreshModels() },
                    onDismiss = { showDefaultModelPicker = false },
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Embedding model ──
            Text(
                text = "Embedding model",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Used to turn memories into vectors locally. Current: ${state.embeddingModel}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            )
            Spacer(modifier = Modifier.height(4.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(
                    "nomic-embed-text" to "Nomic Embed (default)",
                    "mxbai-embed-large" to "mixed-bread large",
                    "all-minilm" to "all-MiniLM",
                    "snowflake-arctic-embed" to "Snowflake Arctic",
                ).forEach { (id, label) ->
                    AssistChip(
                        onClick = { viewModel.setEmbeddingModel(id) },
                        label = { Text(label) },
                        colors = AssistChipDefaults.assistChipColors(
                            containerColor = if (state.embeddingModel == id)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.surfaceVariant,
                            labelColor = if (state.embeddingModel == id)
                                MaterialTheme.colorScheme.onPrimary
                            else
                                MaterialTheme.colorScheme.onSurfaceVariant,
                        ),
                    )
                }
            }
        }

        // Usage is provider-central and persistent. Exact counts are shown
        // when the API reports them; fallback estimates are labelled per model.
        SettingsSection(
            emoji = "📊",
            title = "Usage",
            subtitle = "${usage.totalTokens} tokens across ${usage.calls} model calls",
            initialExpanded = false,
        ) {
            Text(
                text = "${usage.promptTokens} input · ${usage.completionTokens} output tokens",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Tool results processed: ${"%.1f".format(usage.toolResultChars / 1000.0)} KB",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
            )
            Spacer(modifier = Modifier.height(8.dp))
            if (usage.models.isEmpty()) {
                Text(
                    "No model calls recorded yet.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
                )
            } else {
                usage.models.take(8).forEach { model ->
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = modelDisplayName(model.modelId),
                            modifier = Modifier.weight(1f),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            text = "${model.promptTokens + model.completionTokens} · ${model.calls} calls${if (model.estimated) " ~" else ""}",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "~ marks estimated tokens. Cost is not guessed: provider APIs do not return reliable live pricing.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
            )
            var confirmUsageReset by remember { mutableStateOf(false) }
            TextButton(onClick = { confirmUsageReset = true }) { Text("Reset usage") }
            if (confirmUsageReset) {
                AlertDialog(
                    onDismissRequest = { confirmUsageReset = false },
                    title = { Text("Reset usage history?") },
                    text = { Text("This clears token, call, and tool-result counters on this device.") },
                    confirmButton = {
                        TextButton(onClick = {
                            usageViewModel.reset()
                            confirmUsageReset = false
                        }) { Text("Reset") }
                    },
                    dismissButton = {
                        TextButton(onClick = { confirmUsageReset = false }) { Text("Cancel") }
                    },
                )
            }
        }

        // ════════════════════════════════════════════════════════════════
        // 2. APPEARANCE
        // ════════════════════════════════════════════════════════════════
        SettingsSection(
            emoji = "\uD83C\uDFA8",
            title = "Appearance",
            subtitle = "Light, dark, or follow the system theme",
            initialExpanded = false,
        ) {
            Spacer(modifier = Modifier.height(2.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(
                    "system" to "System",
                    "light" to "Light",
                    "dark" to "Dark",
                ).forEach { (id, label) ->
                    AssistChip(
                        onClick = { viewModel.setThemeMode(id) },
                        label = { Text(label) },
                        colors = AssistChipDefaults.assistChipColors(
                            containerColor = if (state.themeMode == id)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.surfaceVariant,
                            labelColor = if (state.themeMode == id)
                                MaterialTheme.colorScheme.onPrimary
                            else
                                MaterialTheme.colorScheme.onSurfaceVariant,
                        ),
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        // ════════════════════════════════════════════════════════════════
        // 3. PERSONA — custom identity + specialist overrides
        // ════════════════════════════════════════════════════════════════
        SettingsSection(
            emoji = "\uD83E\uDDD8",
            title = "Persona",
            subtitle = "Custom system prompt and specialist overrides",
            initialExpanded = false,
        ) {
            Text(
                text = "Custom identity prompt",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Prepended to Aura's built-in identity. Use this for language, tone, or persona. Leave blank for default.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(6.dp))

            var identityText by remember(state.customIdentity) { mutableStateOf(state.customIdentity) }
            OutlinedTextField(
                value = identityText,
                onValueChange = { identityText = it },
                label = { Text("Identity prompt") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 3,
                maxLines = 8,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                TextButton(onClick = { identityText = ""; viewModel.setCustomIdentity("") }) {
                    Text("Reset")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(onClick = { viewModel.setCustomIdentity(identityText.trim()) }) {
                    Text("Save")
                }
            }

            // ── Specialist prompt overrides ──
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Specialist prompts",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Override the built-in system prompt for each specialist. Tap a specialist to edit its prompt.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(8.dp))

            val overridesMap = remember(state.specialistOverrides) {
                try { Json.decodeFromString<Map<String, String>>(state.specialistOverrides) }
                catch (e: Exception) { emptyMap() }
            }
            var editingSpecialist by remember { mutableStateOf<com.aura.agent.Specialist?>(null) }

            for (specialist in com.aura.agent.Specialist.ALL) {
                val hasOverride = overridesMap.containsKey(specialist.name) && !overridesMap[specialist.name].isNullOrBlank()
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "${specialist.icon} ${specialist.name.replaceFirstChar { it.uppercase() }}",
                        style = MaterialTheme.typography.bodyLarge,
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    if (hasOverride) {
                        Text(
                            "✏️",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    TextButton(onClick = { editingSpecialist = specialist }) {
                        Text(if (hasOverride) "Edit" else "Default")
                    }
                }
            }

            editingSpecialist?.let { specialist ->
                var promptText by remember(specialist) {
                    mutableStateOf(overridesMap[specialist.name] ?: specialist.systemPrompt)
                }
                AlertDialog(
                    onDismissRequest = { editingSpecialist = null },
                    title = { Text("${specialist.icon} ${specialist.name.replaceFirstChar { it.uppercase() }}") },
                    text = {
                        OutlinedTextField(
                            value = promptText,
                            onValueChange = { promptText = it },
                            label = { Text("System prompt") },
                            modifier = Modifier.fillMaxWidth(),
                            minLines = 4,
                            maxLines = 12,
                        )
                    },
                    confirmButton = {
                        Button(onClick = {
                            val updated = overridesMap.toMutableMap().apply { put(specialist.name, promptText.trim()) }
                            viewModel.setSpecialistOverrides(
                                Json.encodeToString(MapSerializer(String.serializer(), String.serializer()), updated)
                            )
                            editingSpecialist = null
                        }) { Text("Save") }
                    },
                    dismissButton = {
                        Row {
                            TextButton(onClick = {
                                val updated = overridesMap.toMutableMap().apply { remove(specialist.name) }
                                viewModel.setSpecialistOverrides(
                                    Json.encodeToString(MapSerializer(String.serializer(), String.serializer()), updated)
                                )
                                editingSpecialist = null
                            }) { Text("Reset to default") }
                            Spacer(modifier = Modifier.width(8.dp))
                            TextButton(onClick = { editingSpecialist = null }) { Text("Cancel") }
                        }
                    },
                )
            }
        }

        // ════════════════════════════════════════════════════════════════
        // 4. PRIVACY — app lock, profile, proactive worker toggles
        // ════════════════════════════════════════════════════════════════
        SettingsSection(
            emoji = "\uD83D\uDD12",
            title = "Privacy",
            subtitle = "Biometric lock, proactive worker toggles",
            initialExpanded = false,
        ) {
            Text(
                text = "Require biometric authentication to open Aura. Toggle the proactive workers off if you don't want the 7am brief or the calendar monitor.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(8.dp))

            // Notification-listener access is a special system setting, not a
            // runtime permission. Re-check when the settings Activity returns.
            val notificationContext = androidx.compose.ui.platform.LocalContext.current
            var notificationAccessEnabled by remember {
                mutableStateOf(
                    androidx.core.app.NotificationManagerCompat
                        .getEnabledListenerPackages(notificationContext)
                        .contains(notificationContext.packageName),
                )
            }
            val notificationAccessLauncher =
                androidx.activity.compose.rememberLauncherForActivityResult(
                    androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult(),
                ) {
                    notificationAccessEnabled = androidx.core.app.NotificationManagerCompat
                        .getEnabledListenerPackages(notificationContext)
                        .contains(notificationContext.packageName)
                }
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Notification access", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = if (notificationAccessEnabled)
                            "Enabled — Aura can read active device notifications"
                        else
                            "Off — notification summaries only see Aura itself",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    )
                }
                OutlinedButton(
                    onClick = {
                        notificationAccessLauncher.launch(
                            android.content.Intent(android.provider.Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS),
                        )
                    },
                ) { Text(if (notificationAccessEnabled) "Manage" else "Enable") }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // App lock row.
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "App lock", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = if (state.appLockEnabled)
                            "Enabled — biometric required to open Aura"
                        else
                            "Off — Aura opens straight to chat",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    )
                }
                Switch(
                    checked = state.appLockEnabled,
                    onCheckedChange = { viewModel.setAppLockEnabled(it) },
                )
            }

            // Profile row.
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Profile", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = "Name, traits, and facts Aura has learned",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    )
                }
                TextButton(onClick = onNavigateProfile) { Text("Edit") }
            }

            // Morning brief toggle.
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Morning brief", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = if (state.morningBriefEnabled)
                            "On — %02d:00 daily summary".format(state.morningBriefHour)
                        else
                            "Off — no morning notification",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    )
                }
                Switch(
                    checked = state.morningBriefEnabled,
                    onCheckedChange = { viewModel.setMorningBriefEnabled(it) },
                )
            }
            if (state.morningBriefEnabled) {
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Brief at:", style = MaterialTheme.typography.bodySmall)
                    var showBriefTimePicker by remember { mutableStateOf(false) }
                    OutlinedButton(onClick = { showBriefTimePicker = true }) {
                        Text("%02d:00".format(state.morningBriefHour))
                    }
                    if (showBriefTimePicker) {
                        val tpState = rememberTimePickerState(
                            initialHour = state.morningBriefHour,
                            initialMinute = 0,
                        )
                        AlertDialog(
                            onDismissRequest = { showBriefTimePicker = false },
                            title = { Text("Morning brief time") },
                            text = { TimePicker(state = tpState) },
                            confirmButton = {
                                TextButton(onClick = {
                                    viewModel.setMorningBriefHour(tpState.hour)
                                    showBriefTimePicker = false
                                }) { Text("Set") }
                            },
                            dismissButton = {
                                TextButton(onClick = { showBriefTimePicker = false }) { Text("Cancel") }
                            },
                        )
                    }
                }
            }

            // Calendar monitor toggle.
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Calendar monitor", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = if (state.calendarMonitorEnabled)
                            "On — runs in background, shows notification"
                        else
                            "Off — stops the persistent foreground service",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    )
                }
                Switch(
                    checked = state.calendarMonitorEnabled,
                    onCheckedChange = { viewModel.setCalendarMonitorEnabled(it) },
                )
            }
        }

        // ════════════════════════════════════════════════════════════════
        // 5. DATA & BACKUP — export / restore
        // ════════════════════════════════════════════════════════════════
        val backupState by backupViewModel.state.collectAsState()
        val context = androidx.compose.ui.platform.LocalContext.current
        val importLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
            contract = androidx.activity.result.contract.ActivityResultContracts.OpenDocument(),
        ) { uri: android.net.Uri? ->
            if (uri != null) backupViewModel.stageImport(uri)
        }
        val coroutineScope = androidx.compose.runtime.rememberCoroutineScope()

        SettingsSection(
            emoji = "\uD83D\uDCBE",
            title = "Data & Backup",
            subtitle = "Export or restore memories, conversations, and profile",
            initialExpanded = false,
        ) {
            OutlinedButton(
                onClick = {
                    coroutineScope.launch {
                        val file = backupViewModel.prepareExportFile()
                        if (file != null) {
                            val uri = androidx.core.content.FileProvider.getUriForFile(
                                context,
                                "${context.packageName}.fileprovider",
                                file,
                            )
                            val share = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                                type = "application/json"
                                putExtra(android.content.Intent.EXTRA_STREAM, uri)
                                addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            context.startActivity(
                                android.content.Intent.createChooser(share, "Share Aura backup")
                                    .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                            )
                        }
                    }
                },
                enabled = !backupState.exportInFlight,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (backupState.exportInFlight) "Exporting…" else "Export to JSON")
            }

            Spacer(Modifier.height(8.dp))

            OutlinedButton(
                onClick = {
                    importLauncher.launch(arrayOf("application/json", "*/*"))
                },
                enabled = !backupState.importInFlight,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (backupState.importInFlight) "Restoring…" else "Restore from JSON")
            }

            backupState.lastResult?.let { result ->
                Spacer(Modifier.height(8.dp))
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = result,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(1f),
                        )
                        TextButton(onClick = { backupViewModel.clearResult() }) {
                            Text("Dismiss")
                        }
                    }
                }
            }

            if (backupState.showImportConfirm) {
                AlertDialog(
                    onDismissRequest = { backupViewModel.cancelImport() },
                    title = { Text("Restore from backup?") },
                    text = {
                        Text(
                            "This will add the rows from the backup file to " +
                                "your existing data. Existing rows with the " +
                                "same id are replaced; new rows are added. " +
                                "Embeddings are NOT included — after restoring, " +
                                "go to the Memory tab and tap 'Rebuild embeddings' " +
                                "to re-embed everything in one pass.",
                        )
                    },
                    confirmButton = {
                        TextButton(onClick = { backupViewModel.confirmImport(purgeFirst = false) }) {
                            Text("Add to existing")
                        }
                    },
                    dismissButton = {
                        TextButton(onClick = { backupViewModel.cancelImport() }) {
                            Text("Cancel")
                        }
                    },
                )
            }
        }

        // ── Footer ─────────────────────────────────────────────────────
        Spacer(modifier = Modifier.height(20.dp))
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp, top = 8.dp),
        ) {
            Surface(
                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                shape = RoundedCornerShape(24.dp),
            ) {
                Text(
                    text = "\u2726",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(12.dp),
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Aura",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            )
            Text(
                text = "v" + com.aura.BuildConfig.VERSION_NAME,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.4f),
            )
        }
        Spacer(modifier = Modifier.height(16.dp))
    }
}
