package com.aura.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolProperty

/**
 * Renders a form field per property in a [ToolDefinition]'s parameters.
 * Each property type maps to a Compose input:
 * - "string" → OutlinedTextField
 * - "integer" → number-only OutlinedTextField
 * - "number" → number-only OutlinedTextField
 * - "boolean" → Switch
 * - "any" → OutlinedTextField (free-form)
 *
 * Returns the current arg values as a Map<String, String> via [onArgsChange].
 */
@Composable
fun ToolArgForm(
    definition: ToolDefinition,
    onArgsChange: (Map<String, String>) -> Unit,
) {
    val properties = definition.parameters.properties
    val args = remember { mutableStateOf(mapOf<String, String>()) }

    Column(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for ((name, prop) in properties) {
            val isRequired = name in definition.parameters.required
            ToolArgField(
                name = name,
                property = prop,
                required = isRequired,
                value = args.value[name] ?: "",
                onValueChange = { newValue ->
                    val updated = args.value + (name to newValue)
                    args.value = updated
                    onArgsChange(updated)
                },
            )
        }
    }
}

@Composable
private fun ToolArgField(
    name: String,
    property: ToolProperty,
    required: Boolean,
    value: String,
    onValueChange: (String) -> Unit,
) {
    val label = if (required) "$name *" else name

    when (property.type) {
        "boolean" -> {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Switch(
                    checked = value == "true",
                    onCheckedChange = { onValueChange(it.toString()) },
                )
                Text(label, style = androidx.compose.material3.MaterialTheme.typography.bodyMedium)
            }
        }
        "integer", "number" -> {
            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                label = { Text(label) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                placeholder = { Text(property.description ?: "") },
            )
        }
        else -> {
            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                label = { Text(label) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                placeholder = { Text(property.description ?: "") },
            )
        }
    }
}