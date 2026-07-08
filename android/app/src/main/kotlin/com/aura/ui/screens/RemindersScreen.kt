package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.outlined.NotificationsOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.tasks.ReminderEntity
import com.aura.ui.viewmodel.RemindersViewModel
import java.text.DateFormat
import java.util.Date

/**
 * "My Reminders" — the missing UI for the [com.aura.tasks.ReminderEntity]
 * data that the `set_reminder` tool has been writing to Room for
 * weeks. Before this screen existed, the only way to know about
 * upcoming reminders was to wait for them to fire (or grep the
 * WorkManager database).
 *
 * Each row shows the message and the time it'll fire. The trash
 * icon cancels the reminder: removes the Room row AND cancels the
 * WorkManager job so the notification never appears.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RemindersScreen(
    onBack: () -> Unit,
    viewModel: RemindersViewModel = hiltViewModel(),
) {
    val reminders by viewModel.reminders.collectAsState()
    var pendingCancel by remember { mutableStateOf<ReminderEntity?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Reminders", fontWeight = FontWeight.SemiBold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                ),
            )
        },
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (reminders.isEmpty()) {
                Column(
                    modifier = Modifier.fillMaxSize().padding(32.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                ) {
                    Icon(
                        imageVector = Icons.Outlined.NotificationsOff,
                        contentDescription = null,
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f),
                    )
                    Spacer(Modifier.height(16.dp))
                    Text(
                        text = "No upcoming reminders",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f),
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = "Ask Aura to set one — \"remind me to call mom at 6pm\".",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                    )
                }
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    item {
                        Text(
                            text = "${reminders.size} upcoming",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                            modifier = Modifier.padding(bottom = 4.dp),
                        )
                    }
                    items(reminders, key = { it.id }) { r ->
                        ReminderRow(reminder = r, onCancel = { pendingCancel = r })
                    }
                }
            }
        }
    }

    // Cancel confirmation
    pendingCancel?.let { r ->
        AlertDialog(
            onDismissRequest = { pendingCancel = null },
            title = { Text("Cancel reminder?") },
            text = {
                Text(
                    "\"${r.message}\" was scheduled for " +
                        "${DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(r.triggerAt))}. " +
                        "It will not fire.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.cancel(r.id)
                    pendingCancel = null
                }) { Text("Cancel reminder") }
            },
            dismissButton = {
                TextButton(onClick = { pendingCancel = null }) { Text("Keep") }
            },
        )
    }
}

@Composable
private fun ReminderRow(reminder: ReminderEntity, onCancel: () -> Unit) {
    val now = remember(reminder.triggerAt) { System.currentTimeMillis() }
    val when_ = remember(reminder.triggerAt) {
        DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
            .format(Date(reminder.triggerAt))
    }
    val relative = remember(reminder.triggerAt, now) {
        val ms = reminder.triggerAt - now
        when {
            ms < 0 -> "firing now"
            ms < 60_000L -> "in <1 min"
            ms < 3_600_000L -> "in ${ms / 60_000} min"
            ms < 86_400_000L -> "in ${ms / 3_600_000}h"
            else -> "in ${ms / 86_400_000}d"
        }
    }
    Surface(
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 1.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                imageVector = Icons.Default.NotificationsActive,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    text = reminder.message,
                    style = MaterialTheme.typography.bodyLarge,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = "$when_ · $relative",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
            IconButton(onClick = onCancel) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Cancel reminder",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                )
            }
        }
    }
}
