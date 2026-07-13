package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TimePicker
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.rememberTimePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun ReminderEditorDialog(
    title: String,
    confirmLabel: String,
    initialMessage: String = "",
    initialTriggerAt: Long? = null,
    initialRecurrence: String = "none",
    onDismiss: () -> Unit,
    onConfirm: (message: String, triggerAt: Long, recurrence: String) -> Unit,
) {
    val initialCalendar = remember(initialTriggerAt) {
        Calendar.getInstance().apply {
            if (initialTriggerAt != null) timeInMillis = initialTriggerAt
        }
    }
    var message by remember(initialMessage) { mutableStateOf(initialMessage) }
    var selectedDateMillis by remember(initialTriggerAt) {
        mutableLongStateOf(initialTriggerAt ?: 0L)
    }
    var selectedHour by remember(initialTriggerAt) {
        mutableIntStateOf(if (initialTriggerAt == null) -1 else initialCalendar.get(Calendar.HOUR_OF_DAY))
    }
    var selectedMinute by remember(initialTriggerAt) {
        mutableIntStateOf(if (initialTriggerAt == null) -1 else initialCalendar.get(Calendar.MINUTE))
    }
    var recurrence by remember(initialRecurrence) { mutableStateOf(initialRecurrence) }
    var showDatePicker by remember { mutableStateOf(false) }
    var showTimePicker by remember { mutableStateOf(false) }

    fun resolvedTrigger(): Long? {
        if (selectedDateMillis == 0L || selectedHour < 0) return null
        return Calendar.getInstance().apply {
            timeInMillis = selectedDateMillis
            set(Calendar.HOUR_OF_DAY, selectedHour)
            set(Calendar.MINUTE, selectedMinute)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = message,
                    onValueChange = { message = it },
                    label = { Text("Reminder message") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 1,
                    maxLines = 3,
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    OutlinedButton(
                        onClick = { showDatePicker = true },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(
                            if (selectedDateMillis > 0L)
                                SimpleDateFormat("MMM d", Locale.US).format(Date(selectedDateMillis))
                            else "Date",
                        )
                    }
                    OutlinedButton(
                        onClick = { showTimePicker = true },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(
                            if (selectedHour >= 0) String.format(
                                Locale.US,
                                "%02d:%02d",
                                selectedHour,
                                selectedMinute,
                            ) else "Time",
                        )
                    }
                }
                Text("Repeat")
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("none", "daily", "weekly", "monthly").forEach { option ->
                        AssistChip(
                            onClick = { recurrence = option },
                            label = { Text(option.replaceFirstChar { it.uppercase() }) },
                            colors = if (recurrence == option) {
                                AssistChipDefaults.assistChipColors(
                                    containerColor = androidx.compose.material3.MaterialTheme.colorScheme.primaryContainer,
                                )
                            } else AssistChipDefaults.assistChipColors(),
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    resolvedTrigger()?.let { onConfirm(message.trim(), it, recurrence) }
                },
                enabled = message.isNotBlank() && resolvedTrigger() != null,
            ) { Text(confirmLabel) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )

    if (showDatePicker) {
        val picker = rememberDatePickerState(
            initialSelectedDateMillis = selectedDateMillis.takeIf { it > 0L },
        )
        DatePickerDialog(
            onDismissRequest = { showDatePicker = false },
            confirmButton = {
                TextButton(onClick = {
                    picker.selectedDateMillis?.let { selectedDateMillis = it }
                    showDatePicker = false
                }) { Text("OK") }
            },
            dismissButton = { TextButton(onClick = { showDatePicker = false }) { Text("Cancel") } },
        ) { DatePicker(state = picker) }
    }

    if (showTimePicker) {
        val picker = rememberTimePickerState(
            initialHour = selectedHour.takeIf { it >= 0 } ?: Calendar.getInstance().get(Calendar.HOUR_OF_DAY),
            initialMinute = selectedMinute.takeIf { it >= 0 } ?: Calendar.getInstance().get(Calendar.MINUTE),
        )
        AlertDialog(
            onDismissRequest = { showTimePicker = false },
            title = { Text("Reminder time") },
            text = { TimePicker(state = picker) },
            confirmButton = {
                TextButton(onClick = {
                    selectedHour = picker.hour
                    selectedMinute = picker.minute
                    showTimePicker = false
                }) { Text("OK") }
            },
            dismissButton = { TextButton(onClick = { showTimePicker = false }) { Text("Cancel") } },
        )
    }
}
