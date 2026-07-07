package com.aura.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.aura.hands.HandStep
import com.aura.providers.ToolDefinition
import kotlinx.serialization.json.Json

/**
 * Visual step builder for creating/editing hands.
 * Shows a dropdown of available tools, a per-tool argument form,
 * and a list of added steps with reorder/delete controls.
 *
 * The output is a JSON string matching the Hand.steps format:
 * [{"tool":"name","args":{"k":"v"}}]
 */
@Composable
fun HandStepBuilder(
    availableTools: List<ToolDefinition>,
    steps: List<HandStep>,
    onStepsChange: (List<HandStep>) -> Unit,
) {
    var selectedTool by remember { mutableStateOf<ToolDefinition?>(null) }
    var currentArgs by remember { mutableStateOf(mapOf<String, String>()) }

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // Tool selector
        Text("Add a step:", style = MaterialTheme.typography.labelLarge)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            availableTools.take(10).forEach { tool ->
                AssistChip(
                    onClick = {
                        selectedTool = tool
                        currentArgs = emptyMap()
                    },
                    label = { Text(tool.name, style = MaterialTheme.typography.labelSmall) },
                )
            }
        }
        // Show more tools in a collapsible section if there are many
        if (availableTools.size > 10) {
            var showAll by remember { mutableStateOf(false) }
            OutlinedButton(onClick = { showAll = !showAll }) {
                Text(if (showAll) "Show fewer" else "Show all ${availableTools.size} tools")
            }
            if (showAll) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    availableTools.drop(10).forEach { tool ->
                        AssistChip(
                            onClick = {
                                selectedTool = tool
                                currentArgs = emptyMap()
                            },
                            label = { Text(tool.name, style = MaterialTheme.typography.labelSmall) },
                        )
                    }
                }
            }
        }

        // Argument form for the selected tool
        selectedTool?.let { tool ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        "${tool.name} — ${tool.description}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    ToolArgForm(
                        definition = tool,
                        onArgsChange = { currentArgs = it },
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End,
                    ) {
                        OutlinedButton(onClick = {
                            val step = HandStep(tool = tool.name, args = currentArgs)
                            onStepsChange(steps + step)
                            selectedTool = null
                            currentArgs = emptyMap()
                        }) {
                            Text("Add step")
                        }
                    }
                }
            }
        }

        // Current steps list
        if (steps.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text("Steps:", style = MaterialTheme.typography.labelLarge)
            steps.forEachIndexed { index, step ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        text = "${index + 1}. ${step.tool}",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    // Reorder up
                    if (index > 0) {
                        IconButton(onClick = {
                            val swapped = steps.toMutableList()
                            val tmp = swapped[index - 1]
                            swapped[index - 1] = swapped[index]
                            swapped[index] = tmp
                            onStepsChange(swapped)
                        }) {
                            Icon(Icons.Filled.KeyboardArrowUp, contentDescription = "Move up")
                        }
                    }
                    // Reorder down
                    if (index < steps.size - 1) {
                        IconButton(onClick = {
                            val swapped = steps.toMutableList()
                            val tmp = swapped[index + 1]
                            swapped[index + 1] = swapped[index]
                            swapped[index] = tmp
                            onStepsChange(swapped)
                        }) {
                            Icon(Icons.Filled.KeyboardArrowDown, contentDescription = "Move down")
                        }
                    }
                    // Delete
                    IconButton(onClick = {
                        onStepsChange(steps.filterIndexed { i, _ -> i != index })
                    }) {
                        Icon(Icons.Filled.Delete, contentDescription = "Delete step", tint = MaterialTheme.colorScheme.error)
                    }
                }
            }
        }
    }
}

/**
 * Convert a list of HandSteps to the JSON string format the Hand entity expects.
 */
fun stepsToJson(steps: List<HandStep>): String {
    val items = steps.map { step ->
        val argsObj = kotlinx.serialization.json.JsonObject(
            step.args.mapValues { (_, v) -> kotlinx.serialization.json.JsonPrimitive(v) }
        )
        kotlinx.serialization.json.JsonObject(mapOf(
            "tool" to kotlinx.serialization.json.JsonPrimitive(step.tool),
            "args" to argsObj,
        ))
    }
    val serializer = kotlinx.serialization.builtins.ListSerializer(
        kotlinx.serialization.json.JsonObject.serializer()
    )
    return kotlinx.serialization.json.Json.encodeToString(serializer, items)
}

/**
 * Parse a Hand.steps JSON string back to a list of HandSteps.
 */
fun jsonToSteps(json: String): List<HandStep> {
    if (json.isBlank() || json == "[]") return emptyList()
    return try {
        val arr = Json.parseToJsonElement(json)
        arr.toString() // validate
        emptyList() // TODO: parse when HandStep gets @Serializable
    } catch (e: Exception) {
        emptyList()
    }
}