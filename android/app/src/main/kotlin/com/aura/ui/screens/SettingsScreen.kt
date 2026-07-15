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
import com.aura.ui.settings.CustomEndpointCard
import com.aura.ui.settings.ProviderKeyField
import com.aura.ui.settings.SETTINGS_CREDENTIAL_SPECS
import com.aura.ui.settings.SettingsViewModel
import com.aura.ui.settings.UsageViewModel
import com.aura.ui.util.modelDisplayName
import kotlinx.coroutines.launch

import com.aura.ui.theme.AuraThemeTokens
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
        color = AuraThemeTokens.colors.surface1.copy(alpha = 0.40f),
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
                        color = AuraThemeTokens.colors.textPrimary,
                    )
                    if (subtitle.isNotEmpty()) {
                        Text(
                            text = subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.55f),
                        )
                    }
                }
                Icon(
                    imageVector = if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                    contentDescription = if (expanded) "Collapse" else "Expand",
                    tint = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.5f),
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
                        color = AuraThemeTokens.colors.borderDefault.copy(alpha = 0.5f),
                        modifier = Modifier.padding(bottom = 10.dp),
                    )
                    content()
                }
            }
        }
    }
}

@Composable
private fun RoleModelRow(
    title: String,
    value: String,
    onChoose: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            Text(
                text = if (value.isBlank()) "Not selected" else modelDisplayName(value),
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary,
            )
        }
        OutlinedButton(onClick = onChoose) { Text("Choose") }
    }
}

// ────────────────────────────────────────────────────────────
// Main Settings screen
// ────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class, androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateProfile: () -> Unit,
    onNavigateIdentity: () -> Unit = {},
    onNavigateDiagnostics: () -> Unit = {},
    viewModel: SettingsViewModel = hiltViewModel(),
    backupViewModel: BackupViewModel = hiltViewModel(),
    usageViewModel: UsageViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val usage by usageViewModel.usage.collectAsState()
    var activeModelRole by remember { mutableStateOf<String?>(null) }

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
                color = AuraThemeTokens.colors.textPrimary,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Connect providers, manage memory, customize Aura",
                style = MaterialTheme.typography.bodyLarge,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                color = AuraThemeTokens.colors.surface1,
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
                                        containerColor = AuraThemeTokens.colors.actionPrimary,
                                        labelColor = AuraThemeTokens.colors.onActionPrimary,
                                    ),
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Custom endpoint card (URL + key + test) ──
            CustomEndpointCard(
                baseUrl = state.customBaseUrl,
                apiKey = state.customApiKey,
                isConfigured = state.customIsConfigured,
                testing = state.customTesting,
                result = state.customResult,
                onBaseUrlChange = viewModel::updateCustomBaseUrl,
                onApiKeyChange = viewModel::updateCustomApiKey,
                onTest = viewModel::saveAndTestCustomEndpoint,
                onClear = viewModel::clearCustomEndpoint,
            )

            // ── API Keys section header ──
            Text(
                text = "API Keys",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Stored locally. Model providers use Save & Test; tool services use Save.",
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(4.dp))

            SETTINGS_CREDENTIAL_SPECS.forEach { credential ->
                ProviderKeyField(
                    label = credential.label,
                    value = state.keyDrafts[credential.prefix].orEmpty(),
                    onValueChange = { value ->
                        viewModel.updateCredentialDraft(credential.prefix, value)
                    },
                    helperText = credential.helperText,
                    onVerify = { viewModel.verifyKey(credential.prefix) },
                    verifyResult = state.verifyResults[credential.prefix],
                    verifying = state.verifying == credential.prefix,
                    credentialState = state.credentialStates[credential.prefix],
                    actionLabel = if (credential.testsModelCatalog) "Save & Test" else "Save",
                    requiresTest = credential.testsModelCatalog,
                    enabled = credential.isConsumed,
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Catalog-backed model roles ──
            Text(
                text = "Model roles",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Every role is selected from verified provider catalogs. Unset roles never invent a fallback model.",
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary,
            )
            Spacer(modifier = Modifier.height(6.dp))

            RoleModelRow("Chat default", state.defaultModel) {
                activeModelRole = "chat"
            }
            RoleModelRow("Embedding", state.embeddingModel) {
                activeModelRole = "embedding"
            }
            RoleModelRow("Vision", state.visionModel) {
                activeModelRole = "vision"
            }
            RoleModelRow("Background tasks", state.backgroundModel) {
                activeModelRole = "background"
            }
            RoleModelRow("Deep Mode", state.deepModeModel) {
                activeModelRole = "deep"
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Mixture of Agents",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Choose at least two reference models and one aggregator. Aura then exposes the virtual MoA custom model.",
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary,
            )
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                state.moaReferenceModels.forEach { model ->
                    AssistChip(
                        onClick = {
                            viewModel.setMoaReferenceModels(
                                state.moaReferenceModels.filterNot { it == model },
                            )
                        },
                        label = { Text("${modelDisplayName(model)} ×") },
                    )
                }
            }
            RoleModelRow(
                title = "Reference models (${state.moaReferenceModels.size}/4)",
                value = state.moaReferenceModels.firstOrNull().orEmpty(),
            ) {
                activeModelRole = "moa-reference"
            }
            RoleModelRow("Aggregator", state.moaAggregatorModel) {
                activeModelRole = "moa-aggregator"
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedButton(
                    onClick = viewModel::refreshModels,
                    enabled = !state.modelsLoading,
                ) {
                    Text(if (state.modelsLoading) "Refreshing…" else "Refresh catalog")
                }
                Text(
                    text = when {
                        state.modelsError != null -> state.modelsError!!
                        state.availableModels.isEmpty() -> "Save & Test a provider to load models"
                        else -> "${state.availableModels.size} verified models"
                    },
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodySmall,
                    color = if (state.modelsError == null) {
                        AuraThemeTokens.colors.textPrimary
                    } else {
                        AuraThemeTokens.colors.error
                    },
                )
            }

            activeModelRole?.let { role ->
                val selectableModels = when (role) {
                    "embedding" -> state.availableModels.filter { it.startsWith("ollama:") }
                    "moa-reference", "moa-aggregator" ->
                        state.availableModels.filterNot { it.startsWith("moa:") }
                    else -> state.availableModels
                }
                val current = when (role) {
                    "chat" -> state.defaultModel
                    "embedding" -> state.embeddingModel
                    "vision" -> state.visionModel
                    "background" -> state.backgroundModel
                    "deep" -> state.deepModeModel
                    "moa-aggregator" -> state.moaAggregatorModel
                    "moa-reference" -> state.moaReferenceModels.firstOrNull().orEmpty()
                    else -> ""
                }
                ModelPickerSheet(
                    currentModel = current,
                    models = selectableModels,
                    isLoading = state.modelsLoading,
                    errorMessage = state.modelsError,
                    onPick = { model ->
                        when (role) {
                            "chat" -> viewModel.setDefaultModel(model)
                            "embedding" -> viewModel.setEmbeddingModel(model)
                            "vision" -> viewModel.setVisionModel(model)
                            "background" -> viewModel.setBackgroundModel(model)
                            "deep" -> viewModel.setDeepModeModel(model)
                            "moa-aggregator" -> viewModel.setMoaAggregatorModel(model)
                            "moa-reference" -> {
                                val selected = state.moaReferenceModels
                                viewModel.setMoaReferenceModels(
                                    if (model in selected) selected - model else selected + model,
                                )
                            }
                        }
                        if (role != "moa-reference") activeModelRole = null
                    },
                    onRefresh = viewModel::refreshModels,
                    onDismiss = { activeModelRole = null },
                )
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
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.65f),
            )
            Spacer(modifier = Modifier.height(8.dp))
            if (usage.models.isEmpty()) {
                Text(
                    "No model calls recorded yet.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.65f),
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
                            color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.65f),
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "~ marks estimated tokens. Cost is not guessed: provider APIs do not return reliable live pricing.",
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.55f),
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
                                AuraThemeTokens.colors.actionPrimary
                            else
                                AuraThemeTokens.colors.surface1,
                            labelColor = if (state.themeMode == id)
                                AuraThemeTokens.colors.onActionPrimary
                            else
                                AuraThemeTokens.colors.textPrimary,
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
            subtitle = "Aura identity and specialist overrides",
            initialExpanded = false,
        ) {
            Surface(
                color = AuraThemeTokens.colors.surface1,
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(onClick = onNavigateIdentity),
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Aura identity",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = if (state.identityCustomized) {
                                "Customized · backed up with your settings"
                            } else {
                                "Using the bundled default persona"
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = AuraThemeTokens.colors.textPrimary,
                        )
                    }
                    OutlinedButton(onClick = onNavigateIdentity) {
                        Text(if (state.identityCustomized) "Edit" else "Customize")
                    }
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
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                            color = AuraThemeTokens.colors.actionPrimary,
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
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                            "Off — enable to summarize device notifications",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                        color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                        color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                        color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
                        color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
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
            subtitle = "Export, restore, and inspect local diagnostics",
            initialExpanded = false,
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Diagnostics", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "Local crash and error history",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.6f),
                    )
                }
                TextButton(onClick = onNavigateDiagnostics) { Text("Open") }
            }
            Spacer(Modifier.height(8.dp))

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
                    color = AuraThemeTokens.colors.surface1,
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
                color = AuraThemeTokens.colors.actionPrimary.copy(alpha = 0.12f),
                shape = RoundedCornerShape(24.dp),
            ) {
                Text(
                    text = "\u2726",
                    style = MaterialTheme.typography.titleLarge,
                    color = AuraThemeTokens.colors.actionPrimary,
                    modifier = Modifier.padding(12.dp),
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Aura",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.7f),
            )
            Text(
                text = "v" + com.aura.BuildConfig.VERSION_NAME,
                style = MaterialTheme.typography.labelSmall,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.4f),
            )
        }
        Spacer(modifier = Modifier.height(16.dp))
    }
}
