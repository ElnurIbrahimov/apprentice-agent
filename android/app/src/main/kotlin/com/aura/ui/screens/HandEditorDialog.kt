package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.aura.hands.Hand
import com.aura.hands.HandCondition
import com.aura.hands.HandScheduleType
import com.aura.hands.HandStep
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolProperty
import com.aura.ui.viewmodel.HandDraft
import java.time.DayOfWeek
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
internal fun HandEditorDialog(
    initial: Hand?,
    toolDefinitions: List<ToolDefinition>,
    onDismiss: () -> Unit,
    onSave: (HandDraft) -> Unit,
) {
    val json = remember { Json { ignoreUnknownKeys = true } }
    var name by remember(initial?.id) { mutableStateOf(initial?.name.orEmpty()) }
    var trigger by remember(initial?.id) { mutableStateOf(initial?.triggerPhrase.orEmpty()) }
    var steps by remember(initial?.id) { mutableStateOf(parseSteps(initial?.steps.orEmpty(), json)) }
    var variables by remember(initial?.id) { mutableStateOf(parseVariables(initial?.variables.orEmpty(), json)) }
    var conditions by remember(initial?.id) { mutableStateOf(parseConditions(initial?.conditions.orEmpty(), json)) }
    var scheduleType by remember(initial?.id) { mutableStateOf(initial?.scheduleType ?: HandScheduleType.NONE.value) }
    var scheduleHour by remember(initial?.id) { mutableStateOf((initial?.scheduleHour ?: 9).toString()) }
    var scheduleMinute by remember(initial?.id) { mutableStateOf((initial?.scheduleMinute ?: 0).toString()) }
    var scheduleDay by remember(initial?.id) { mutableStateOf(initial?.scheduleDayOfWeek ?: 1) }
    var newVariableName by remember { mutableStateOf("") }
    var newVariableValue by remember { mutableStateOf("") }
    var newConditionVariable by remember { mutableStateOf("") }
    var newConditionOperator by remember { mutableStateOf("equals") }
    var newConditionValue by remember { mutableStateOf("") }
    var selectedTool by remember { mutableStateOf<ToolDefinition?>(null) }
    var argValues by remember { mutableStateOf<Map<String, String>>(emptyMap()) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Column {
                Text(if (initial == null) "New hand" else "Edit hand")
                Text(
                    "A repeatable, inspectable tool workflow",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth().heightIn(max = 620.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = trigger,
                    onValueChange = { trigger = it },
                    label = { Text("Trigger phrase (optional)") },
                    supportingText = { Text("Saying this in chat asks Aura to run the hand") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )

                EditorSection("Steps", "Use {{variable}} inside any text argument") {
                    StepEditor(
                        steps = steps,
                        toolDefinitions = toolDefinitions,
                        selectedTool = selectedTool,
                        argValues = argValues,
                        onSelectedTool = { selectedTool = it; argValues = emptyMap() },
                        onArgValues = { argValues = it },
                        onSteps = { steps = it },
                    )
                }

                EditorSection("Variables", "Defaults can be overridden each time you run") {
                    if (variables.isEmpty()) {
                        Text("No variables", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    variables.forEach { (key, value) ->
                        Surface(
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Row(Modifier.padding(start = 12.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text("{{$key}}", fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                                Text(value.ifBlank { "empty" }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                IconButton(onClick = { variables = variables - key }) {
                                    Icon(Icons.Filled.Close, contentDescription = "Remove variable")
                                }
                            }
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        OutlinedTextField(
                            value = newVariableName,
                            onValueChange = { newVariableName = it.filter { char -> char.isLetterOrDigit() || char == '_' } },
                            label = { Text("Variable") },
                            singleLine = true,
                            modifier = Modifier.weight(1f),
                        )
                        OutlinedTextField(
                            value = newVariableValue,
                            onValueChange = { newVariableValue = it },
                            label = { Text("Default") },
                            singleLine = true,
                            modifier = Modifier.weight(1f),
                        )
                    }
                    OutlinedButton(
                        onClick = {
                            val key = newVariableName.trim()
                            if (key.isNotBlank()) {
                                variables = variables + (key to newVariableValue)
                                newVariableName = ""
                                newVariableValue = ""
                            }
                        },
                        enabled = newVariableName.isNotBlank(),
                    ) {
                        Icon(Icons.Filled.Add, contentDescription = null)
                        Text("Add variable")
                    }
                }

                EditorSection("Conditions", "All conditions must pass before any tool runs") {
                    conditions.forEachIndexed { index, condition ->
                        Surface(
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Row(Modifier.padding(start = 12.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(condition.failureDescription(), style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                                IconButton(onClick = { conditions = conditions.toMutableList().apply { removeAt(index) } }) {
                                    Icon(Icons.Filled.Close, contentDescription = "Remove condition")
                                }
                            }
                        }
                    }
                    OutlinedTextField(
                        value = newConditionVariable,
                        onValueChange = { newConditionVariable = it },
                        label = { Text("Variable name") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    OperatorPicker(newConditionOperator) { newConditionOperator = it }
                    if (newConditionOperator !in setOf("not_empty", "empty")) {
                        OutlinedTextField(
                            value = newConditionValue,
                            onValueChange = { newConditionValue = it },
                            label = { Text("Expected value") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                    OutlinedButton(
                        onClick = {
                            conditions = conditions + HandCondition(
                                variable = newConditionVariable.trim(),
                                operator = newConditionOperator,
                                value = newConditionValue,
                            )
                            newConditionVariable = ""
                            newConditionValue = ""
                        },
                        enabled = newConditionVariable.isNotBlank(),
                    ) { Text("Add condition") }
                }

                EditorSection("Schedule", "WorkManager runs the next local-time occurrence") {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        HandScheduleType.entries.forEach { type ->
                            FilterChip(
                                selected = scheduleType == type.value,
                                onClick = { scheduleType = type.value },
                                label = { Text(type.value.replaceFirstChar { it.uppercase() }) },
                            )
                        }
                    }
                    if (scheduleType != HandScheduleType.NONE.value) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(
                                value = scheduleHour,
                                onValueChange = { scheduleHour = it.filter(Char::isDigit).take(2) },
                                label = { Text("Hour") },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            OutlinedTextField(
                                value = scheduleMinute,
                                onValueChange = { scheduleMinute = it.filter(Char::isDigit).take(2) },
                                label = { Text("Minute") },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                        }
                    }
                    if (scheduleType == HandScheduleType.WEEKLY.value) {
                        DayPicker(scheduleDay) { scheduleDay = it }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                enabled = name.isNotBlank() && steps.isNotEmpty(),
                onClick = {
                    onSave(
                        HandDraft(
                            name = name.trim(),
                            triggerPhrase = trigger.trim(),
                            stepsJson = JsonArray(steps.map { it.toJsonObject() }).toString(),
                            variables = variables,
                            conditions = conditions,
                            scheduleType = scheduleType,
                            scheduleHour = scheduleHour.toIntOrNull()?.coerceIn(0, 23) ?: 9,
                            scheduleMinute = scheduleMinute.toIntOrNull()?.coerceIn(0, 59) ?: 0,
                            scheduleDayOfWeek = scheduleDay,
                        ),
                    )
                },
            ) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun EditorSection(title: String, subtitle: String, content: @Composable () -> Unit) {
    HorizontalDivider()
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
        Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        content()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun StepEditor(
    steps: List<HandStep>,
    toolDefinitions: List<ToolDefinition>,
    selectedTool: ToolDefinition?,
    argValues: Map<String, String>,
    onSelectedTool: (ToolDefinition?) -> Unit,
    onArgValues: (Map<String, String>) -> Unit,
    onSteps: (List<HandStep>) -> Unit,
) {
    if (steps.isEmpty()) {
        Text("Pick a tool to add the first step", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    steps.forEachIndexed { index, step ->
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(Modifier.padding(start = 12.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("${index + 1}. ${step.tool}", fontWeight = FontWeight.SemiBold)
                    if (step.args.isNotEmpty()) {
                        Text(
                            step.args.entries.joinToString { "${it.key}=${it.value}" },
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                IconButton(onClick = { onSteps(steps.toMutableList().apply { removeAt(index) }) }) {
                    Icon(Icons.Filled.Close, contentDescription = "Remove step")
                }
            }
        }
    }

    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
        OutlinedTextField(
            value = selectedTool?.name.orEmpty(),
            onValueChange = {},
            readOnly = true,
            label = { Text("Add tool step") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable, enabled = true).fillMaxWidth(),
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            toolDefinitions.sortedBy { it.name }.forEach { definition ->
                DropdownMenuItem(
                    text = { Text(definition.name) },
                    onClick = { onSelectedTool(definition); expanded = false },
                )
            }
        }
    }
    selectedTool?.let { definition ->
        Text(definition.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        definition.parameters.properties.forEach { (argument, property) ->
            HandArgumentField(
                name = argument,
                property = property,
                value = argValues[argument].orEmpty(),
                onValueChange = { onArgValues(argValues + (argument to it)) },
            )
        }
        Button(
            onClick = {
                val args = argValues.filterValues { it.isNotBlank() }
                onSteps(steps + HandStep(definition.name, args))
                onSelectedTool(null)
                onArgValues(emptyMap())
            },
        ) { Text("Add step") }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun OperatorPicker(value: String, onValue: (String) -> Unit) {
    val operators = listOf(
        "equals", "not_equals", "contains", "not_contains",
        "greater_than", "less_than", "not_empty", "empty",
    )
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded, { expanded = it }) {
        OutlinedTextField(
            value = value.replace('_', ' '),
            onValueChange = {},
            readOnly = true,
            label = { Text("Operator") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable, enabled = true).fillMaxWidth(),
        )
        ExposedDropdownMenu(expanded, { expanded = false }) {
            operators.forEach { operator ->
                DropdownMenuItem(
                    text = { Text(operator.replace('_', ' ')) },
                    onClick = { onValue(operator); expanded = false },
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DayPicker(selected: Int, onSelected: (Int) -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        DayOfWeek.entries.forEach { day ->
            AssistChip(
                onClick = { onSelected(day.value) },
                label = { Text(day.name.take(3).lowercase().replaceFirstChar { it.uppercase() }) },
                leadingIcon = if (selected == day.value) ({ Text("•") }) else null,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HandArgumentField(
    name: String,
    property: ToolProperty,
    value: String,
    onValueChange: (String) -> Unit,
) {
    if (property.enum.isNotEmpty()) {
        var expanded by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(expanded, { expanded = it }) {
            OutlinedTextField(
                value = value,
                onValueChange = {},
                readOnly = true,
                label = { Text(name) },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable, enabled = true).fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded, { expanded = false }) {
                property.enum.forEach { option ->
                    DropdownMenuItem(
                        text = { Text(option) },
                        onClick = { onValueChange(option); expanded = false },
                    )
                }
            }
        }
    } else {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            label = { Text(name) },
            supportingText = property.description?.let { description -> ({ Text(description) }) },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

private fun parseSteps(raw: String, json: Json): List<HandStep> = runCatching {
    (json.parseToJsonElement(raw) as? JsonArray).orEmpty().mapNotNull { element ->
        val obj = element as? JsonObject ?: return@mapNotNull null
        val tool = obj["tool"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
        val args = (obj["args"] as? JsonObject)?.mapValues { it.value.jsonPrimitive.contentOrNull.orEmpty() }.orEmpty()
        HandStep(tool, args)
    }
}.getOrDefault(emptyList())

private fun parseVariables(raw: String, json: Json): Map<String, String> = runCatching {
    json.parseToJsonElement(raw).jsonObject.mapValues { it.value.jsonPrimitive.contentOrNull.orEmpty() }
}.getOrDefault(emptyMap())

private fun parseConditions(raw: String, json: Json): List<HandCondition> = runCatching {
    json.decodeFromString<List<HandCondition>>(raw)
}.getOrDefault(emptyList())
