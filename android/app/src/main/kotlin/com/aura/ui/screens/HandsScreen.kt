package com.aura.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.hands.Hand
import com.aura.hands.HandRun
import com.aura.hands.HandRunStatus
import com.aura.hands.HandScheduleType
import com.aura.ui.components.AuraScreenHeader
import com.aura.ui.viewmodel.HandsViewModel
import java.text.DateFormat
import java.time.DayOfWeek
import java.util.Date
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

@Composable
fun HandsScreen(
    viewModel: HandsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    var selectedTab by remember { mutableIntStateOf(0) }
    var editingHand by remember { mutableStateOf<Hand?>(null) }
    var showNewHand by remember { mutableStateOf(false) }
    var runHand by remember { mutableStateOf<Hand?>(null) }
    var deleteHand by remember { mutableStateOf<Hand?>(null) }
    var confirmClearHistory by remember { mutableStateOf(false) }
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(state.error) {
        state.error?.let {
            snackbar.showSnackbar(it)
            viewModel.clearError()
        }
    }

    Scaffold(
        contentWindowInsets = WindowInsets(0),
        snackbarHost = { SnackbarHost(snackbar) },
        floatingActionButton = {
            if (selectedTab == 0) {
                ExtendedFloatingActionButton(
                    onClick = { showNewHand = true },
                    icon = { Icon(Icons.Filled.Add, contentDescription = null) },
                    text = { Text("Add hand") },
                )
            }
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(horizontal = 20.dp)) {
            AuraScreenHeader(
                title = "Hands",
                subtitle = "Automations with explicit inputs, gates, and history",
            )
            TabRow(selectedTabIndex = selectedTab) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("Automations  ${state.hands.size}") },
                    icon = { Icon(Icons.Filled.AutoAwesome, contentDescription = null) },
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("Run history  ${state.runs.size}") },
                    icon = { Icon(Icons.Filled.History, contentDescription = null) },
                )
            }
            Spacer(Modifier.height(10.dp))

            if (selectedTab == 0) {
                when {
                    state.loading -> HandsSkeletonLoading()
                    state.hands.isEmpty() -> HandsEmptyState()
                    else -> LazyColumn(
                        contentPadding = PaddingValues(bottom = 96.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        items(state.hands, key = { it.id }) { hand ->
                            HandCard(
                                hand = hand,
                                lastRun = state.runs.firstOrNull { it.handId == hand.id },
                                isRunning = state.running == hand.name,
                                onRun = { runHand = hand },
                                onToggle = { viewModel.toggle(hand) },
                                onEdit = { editingHand = hand },
                                onDelete = { deleteHand = hand },
                            )
                        }
                    }
                }
            } else {
                RunHistory(
                    runs = state.runs,
                    onClear = { confirmClearHistory = true },
                )
            }
        }
    }

    if (showNewHand) {
        HandEditorDialog(
            initial = null,
            toolDefinitions = viewModel.toolRegistry.definitions(),
            onDismiss = { showNewHand = false },
            onSave = { draft ->
                viewModel.save(null, draft)
                showNewHand = false
            },
        )
    }
    editingHand?.let { hand ->
        HandEditorDialog(
            initial = hand,
            toolDefinitions = viewModel.toolRegistry.definitions(),
            onDismiss = { editingHand = null },
            onSave = { draft ->
                viewModel.save(hand, draft)
                editingHand = null
            },
        )
    }
    runHand?.let { hand ->
        RunHandDialog(
            hand = hand,
            onDismiss = { runHand = null },
            onRun = { variables ->
                viewModel.runHand(hand, variables)
                runHand = null
            },
        )
    }
    deleteHand?.let { hand ->
        AlertDialog(
            onDismissRequest = { deleteHand = null },
            title = { Text("Delete ${hand.name}?") },
            text = { Text("The hand is removed. Its run-history snapshots remain available until you clear history.") },
            confirmButton = {
                TextButton(onClick = { viewModel.delete(hand); deleteHand = null }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleteHand = null }) { Text("Cancel") } },
        )
    }
    if (confirmClearHistory) {
        AlertDialog(
            onDismissRequest = { confirmClearHistory = false },
            title = { Text("Clear run history?") },
            text = { Text("This removes local automation outputs and statuses. Saved hands are not changed.") },
            confirmButton = {
                TextButton(onClick = { viewModel.clearHistory(); confirmClearHistory = false }) { Text("Clear") }
            },
            dismissButton = { TextButton(onClick = { confirmClearHistory = false }) { Text("Cancel") } },
        )
    }
    state.lastResult?.let { output ->
        AlertDialog(
            onDismissRequest = viewModel::clearResult,
            title = { Text("Hand finished") },
            text = { Text(output) },
            confirmButton = { TextButton(onClick = viewModel::clearResult) { Text("Done") } },
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun HandCard(
    hand: Hand,
    lastRun: HandRun?,
    isRunning: Boolean,
    onRun: () -> Unit,
    onToggle: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = if (hand.enabled) 0.85f else 0.45f),
        shape = RoundedCornerShape(18.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.14f),
                    shape = CircleShape,
                ) {
                    Icon(
                        Icons.Filled.Build,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(10.dp).size(20.dp),
                    )
                }
                Spacer(Modifier.size(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(hand.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        hand.triggerPhrase.ifBlank { "Manual or agent-triggered" },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Switch(checked = hand.enabled, onCheckedChange = { onToggle() })
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                MetadataPill(Icons.Filled.Schedule, scheduleLabel(hand))
                MetadataPill(null, "${jsonArrayCount(hand.steps)} steps")
                val variableCount = jsonObjectCount(hand.variables)
                if (variableCount > 0) MetadataPill(null, "$variableCount vars")
                val conditionCount = jsonArrayCount(hand.conditions)
                if (conditionCount > 0) MetadataPill(null, "$conditionCount gates")
                if (lastRun != null) {
                    MetadataPill(null, "Last: ${lastRun.status.replace('_', ' ')}")
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                Button(onClick = onRun, enabled = hand.enabled && !isRunning, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null)
                    Text(if (isRunning) "Running…" else "Run")
                }
                IconButton(onClick = onEdit) { Icon(Icons.Filled.Edit, contentDescription = "Edit hand") }
                IconButton(onClick = onDelete) {
                    Icon(Icons.Filled.Delete, contentDescription = "Delete hand", tint = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
}

@Composable
private fun MetadataPill(icon: androidx.compose.ui.graphics.vector.ImageVector?, text: String) {
    Surface(color = MaterialTheme.colorScheme.surface, shape = RoundedCornerShape(999.dp)) {
        Row(
            Modifier.padding(horizontal = 9.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            if (icon != null) Icon(icon, contentDescription = null, modifier = Modifier.size(13.dp))
            Text(text, style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
private fun RunHistory(runs: List<HandRun>, onClear: () -> Unit) {
    if (runs.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Filled.History, contentDescription = null, modifier = Modifier.size(36.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.height(10.dp))
                Text("No runs yet", style = MaterialTheme.typography.titleMedium)
                Text("Manual, agent, phrase, and scheduled runs appear here", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        return
    }
    var filter by remember { mutableStateOf("all") }
    val filteredRuns = when (filter) {
        "success" -> runs.filter { it.status == HandRunStatus.SUCCESS.value }
        "failed" -> runs.filter {
            it.status in setOf(
                HandRunStatus.FAILED.value,
                HandRunStatus.NEEDS_PERMISSION.value,
                HandRunStatus.NEEDS_APPROVAL.value,
            )
        }
        else -> runs
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("all" to "All", "success" to "Success", "failed" to "Failed").forEach { (value, label) ->
                FilterChip(
                    selected = filter == value,
                    onClick = { filter = value },
                    label = { Text(label) },
                )
            }
            Spacer(Modifier.weight(1f))
            TextButton(onClick = onClear) { Text("Clear") }
        }
        if (filteredRuns.isEmpty()) {
            Box(Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                Text("No $filter runs", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            LazyColumn(contentPadding = PaddingValues(bottom = 96.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(filteredRuns, key = { it.id }) { run -> RunHistoryCard(run) }
            }
        }
    }
}

@Composable
private fun RunHistoryCard(run: HandRun) {
    var expanded by remember(run.id) { mutableStateOf(false) }
    val color = statusColor(run.status)
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.fillMaxWidth().clickable { expanded = !expanded },
    ) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(9.dp).background(color, CircleShape))
                Spacer(Modifier.size(9.dp))
                Text(run.handName, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                Text(run.status.replace('_', ' '), style = MaterialTheme.typography.labelSmall, color = color)
            }
            Text(
                "${run.trigger.replaceFirstChar { it.uppercase() }} · ${DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(run.startedAt))}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            AnimatedVisibility(expanded) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    HorizontalDivider()
                    if (run.failedStep != null) Text("Stopped at step ${run.failedStep}", color = MaterialTheme.colorScheme.error)
                    Text(run.output.ifBlank { "No output recorded" }, style = MaterialTheme.typography.bodySmall)
                    if (run.variablesJson != "{}") {
                        Text("Inputs: ${run.variablesJson}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

@Composable
private fun RunHandDialog(hand: Hand, onDismiss: () -> Unit, onRun: (Map<String, String>) -> Unit) {
    val defaults = remember(hand.id, hand.variables, hand.steps) { runtimeVariableInputs(hand) }
    var values by remember(hand.id) { mutableStateOf(defaults) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Run ${hand.name}") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (defaults.isEmpty()) {
                    Text("This hand has no runtime variables. Its conditions and steps will run with saved defaults.")
                } else {
                    Text("Override inputs for this run only", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    defaults.keys.forEach { key ->
                        OutlinedTextField(
                            value = values[key].orEmpty(),
                            onValueChange = { values = values + (key to it) },
                            label = { Text(key) },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = { onRun(values) }) { Text("Run") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun HandsSkeletonLoading() {
    val transition = rememberInfiniteTransition(label = "hands-skeleton")
    val alpha by transition.animateFloat(0.25f, 0.60f, infiniteRepeatable(tween(850)), label = "pulse")
    Column(verticalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
        repeat(3) {
            Box(Modifier.fillMaxWidth().height(126.dp).background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = alpha), RoundedCornerShape(18.dp)))
        }
    }
}

@Composable
private fun HandsEmptyState() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(horizontal = 28.dp)) {
            Surface(color = MaterialTheme.colorScheme.primary.copy(alpha = 0.14f), shape = RoundedCornerShape(22.dp)) {
                Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(18.dp).size(32.dp))
            }
            Spacer(Modifier.height(14.dp))
            Text("Build your first hand", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(5.dp))
            Text(
                "Chain tools, name the inputs, decide when it may run, and keep every result inspectable.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun statusColor(status: String): Color = when (status) {
    HandRunStatus.SUCCESS.value -> Color(0xFF5FD3A8)
    HandRunStatus.RUNNING.value -> MaterialTheme.colorScheme.primary
    HandRunStatus.SKIPPED.value -> Color(0xFFE4B865)
    HandRunStatus.NEEDS_PERMISSION.value, HandRunStatus.NEEDS_APPROVAL.value -> Color(0xFFE4B865)
    else -> MaterialTheme.colorScheme.error
}

private fun scheduleLabel(hand: Hand): String = when (HandScheduleType.from(hand.scheduleType)) {
    HandScheduleType.NONE -> "Manual"
    HandScheduleType.DAILY -> "Daily %02d:%02d".format(hand.scheduleHour, hand.scheduleMinute)
    HandScheduleType.WEEKDAYS -> "Weekdays %02d:%02d".format(hand.scheduleHour, hand.scheduleMinute)
    HandScheduleType.WEEKLY -> {
        val day = runCatching { DayOfWeek.of(hand.scheduleDayOfWeek).name.take(3).lowercase().replaceFirstChar { it.uppercase() } }.getOrDefault("Mon")
        "$day %02d:%02d".format(hand.scheduleHour, hand.scheduleMinute)
    }
}

private fun jsonArrayCount(raw: String): Int = runCatching {
    (Json.parseToJsonElement(raw) as? kotlinx.serialization.json.JsonArray)?.size ?: 0
}.getOrDefault(0)

private fun jsonObjectCount(raw: String): Int = runCatching {
    Json.parseToJsonElement(raw).jsonObject.size
}.getOrDefault(0)

internal fun runtimeVariableInputs(hand: Hand): Map<String, String> {
    val inputs = LinkedHashMap(parseVariableDefaults(hand.variables))
    Regex("""\{\{\s*([A-Za-z][A-Za-z0-9_.-]*)\s*\}\}""")
        .findAll(hand.steps)
        .map { it.groupValues[1] }
        .forEach { inputs.putIfAbsent(it, "") }
    return inputs
}

private fun parseVariableDefaults(raw: String): Map<String, String> = runCatching {
    Json.parseToJsonElement(raw).jsonObject.mapValues { it.value.jsonPrimitive.content }
}.getOrDefault(emptyMap())
