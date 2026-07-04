package com.aura.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.hands.Hand
import com.aura.hands.HandStep
import com.aura.providers.ToolProperty
import com.aura.ui.viewmodel.HandsViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HandsScreen(
    viewModel: HandsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val toolRegistry = viewModel.toolRegistry
    var showAddDialog by remember { mutableStateOf(false) }

    Scaffold(
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { showAddDialog = true },
                icon = { Icon(Icons.Filled.Add, null) },
                text = { Text("Add hand") },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
            Text("Hands", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
            Text("Automation macros. ${state.hands.size} configured.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
            Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp))

            state.lastResult?.let { msg ->
                Surface(color = MaterialTheme.colorScheme.secondaryContainer, shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)) {
                    Text(msg, Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSecondaryContainer)
                }
            }

            if (state.loading) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            else if (state.hands.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("🤖", style = MaterialTheme.typography.displayLarge); Spacer(Modifier.height(8.dp)); Text("No hands yet", style = MaterialTheme.typography.titleMedium); Text("Tap + to add one, or ask Aura to set one up.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)) } }
            else LazyColumn(contentPadding = PaddingValues(top = 8.dp, bottom = 80.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.hands, key = { it.id }) { h ->
                    HandRow(
                        hand = h,
                        isRunning = state.running == h.name,
                        onRun = { viewModel.runHand(h) },
                        onToggle = { viewModel.toggle(h) },
                        onDelete = { viewModel.delete(h.name) },
                    )
                }
            }
        }
    }

    if (showAddDialog) {
        AddHandDialog(
            toolDefinitions = toolRegistry.definitions(),
            onDismiss = { showAddDialog = false },
            onSave = { name, trigger, steps ->
                viewModel.add(name, trigger, steps)
                showAddDialog = false
            },
        )
    }
}

@Composable
private fun HandRow(hand: Hand, isRunning: Boolean, onRun: () -> Unit, onToggle: () -> Unit, onDelete: () -> Unit) {
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Build, null, tint = if (hand.enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f), modifier = Modifier.size(24.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(hand.name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                Text(hand.triggerPhrase.ifBlank { "no trigger phrase" }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f))
            }
            Switch(checked = hand.enabled, onCheckedChange = { onToggle() })
            Spacer(Modifier.width(4.dp))
            FilledIconButton(onClick = onRun, enabled = hand.enabled && !isRunning) { Icon(Icons.Filled.PlayArrow, "Run", tint = MaterialTheme.colorScheme.onPrimary) }
            Spacer(Modifier.width(4.dp))
            IconButton(onClick = onDelete, modifier = Modifier.size(32.dp)) { Icon(Icons.Filled.Delete, "Delete", tint = MaterialTheme.colorScheme.error.copy(alpha = 0.6f), modifier = Modifier.size(18.dp)) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun AddHandDialog(
    toolDefinitions: List<com.aura.providers.ToolDefinition>,
    onDismiss: () -> Unit,
    onSave: (String, String, String) -> Unit,
) {
    var name by remember { mutableStateOf("") }
    var trigger by remember { mutableStateOf("") }
    var steps by remember { mutableStateOf(listOf<HandStep>()) }
    var selectedTool by remember { mutableStateOf<com.aura.providers.ToolDefinition?>(null) }
    var argValues by remember { mutableStateOf(mapOf<String, String>()) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New hand") },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth().heightIn(max = 480.dp),
            ) {
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") }, singleLine = true)
                OutlinedTextField(
                    value = trigger,
                    onValueChange = { trigger = it },
                    label = { Text("Trigger phrase (optional)") },
                    singleLine = true,
                    supportingText = { Text("What the user says to trigger it (wires to chat loop)") },
                )

                HorizontalDivider()
                Text("Steps", fontWeight = FontWeight.SemiBold)
                if (steps.isEmpty()) {
                    Text("No steps yet. Pick a tool below.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
                } else {
                    steps.forEachIndexed { idx, step ->
                        Surface(
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                                Column(Modifier.weight(1f)) {
                                    Text("${idx + 1}. ${step.tool}", fontWeight = FontWeight.Medium)
                                    if (step.args.isNotEmpty()) {
                                        Text(step.args.entries.joinToString { "${it.key}=${it.value}" }, style = MaterialTheme.typography.bodySmall)
                                    }
                                }
                                TextButton(onClick = { steps = steps.toMutableList().apply { removeAt(idx) } }) { Text("Remove") }
                            }
                        }
                    }
                }

                // Tool picker
                var expanded by remember { mutableStateOf(false) }
                ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
                    OutlinedTextField(
                        value = selectedTool?.name ?: "",
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Add tool step") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        modifier = Modifier.menuAnchor(),
                    )
                    ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        toolDefinitions.forEach { def ->
                            DropdownMenuItem(
                                text = { Text(def.name) },
                                onClick = {
                                    selectedTool = def
                                    argValues = emptyMap()
                                    expanded = false
                                },
                            )
                        }
                    }
                }

                selectedTool?.let { def ->
                    Text(def.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f))
                    def.parameters.properties.forEach { (argName, prop) ->
                        ArgField(name = argName, property = prop, value = argValues[argName] ?: "", onValueChange = { argValues = argValues + (argName to it) })
                    }
                    Button(
                        onClick = {
                            val args = def.parameters.properties.mapNotNull { (argName, _) ->
                                val v = argValues[argName]
                                if (v.isNullOrBlank() && def.parameters.required.contains(argName)) null
                                else if (v.isNullOrBlank()) null
                                else argName to v
                            }.toMap()
                            steps = steps + HandStep(tool = def.name, args = args)
                            selectedTool = null
                            argValues = emptyMap()
                        },
                        modifier = Modifier.align(Alignment.End),
                    ) { Text("Add step") }
                }
            }
        },
        confirmButton = {
            TextButton(
                enabled = name.isNotBlank(),
                onClick = { onSave(name.trim(), trigger.trim(), stepsToJson(steps)) },
            ) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ArgField(name: String, property: ToolProperty, value: String, onValueChange: (String) -> Unit) {
    if (property.enum.isNotEmpty()) {
        var expanded by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
            OutlinedTextField(
                value = value,
                onValueChange = {},
                readOnly = true,
                label = { Text(name) },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                property.enum.forEach { opt ->
                    DropdownMenuItem(
                        text = { Text(opt) },
                        onClick = { onValueChange(opt); expanded = false },
                    )
                }
            }
        }
    } else {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            label = { Text(name) },
            singleLine = property.type != "string" || (property.description?.length ?: 0) < 40,
            minLines = if (property.type == "string" && (property.description?.length ?: 0) >= 40) 2 else 1,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

private fun stepsToJson(steps: List<HandStep>): String {
    val array = kotlinx.serialization.json.JsonArray(steps.map { it.toJsonObject() })
    return array.toString()
}
