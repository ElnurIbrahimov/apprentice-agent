package com.aura.ui.screens.chat

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.aura.tools.Citation
import com.aura.ui.theme.AuraThemeTokens

@Composable
fun StopStreamingDialog(
    visible: Boolean,
    onStop: () -> Unit,
    onKeepStreaming: () -> Unit,
) {
    if (!visible) return
    AlertDialog(
        onDismissRequest = onKeepStreaming,
        title = { Text("Stop streaming?") },
        text = { Text("Aura will keep and save the response generated so far.") },
        confirmButton = {
            TextButton(onClick = onStop) { Text("Stop") }
        },
        dismissButton = {
            TextButton(onClick = onKeepStreaming) { Text("Keep streaming") }
        },
    )
}

@Composable
fun DeleteConversationDialog(
    visible: Boolean,
    onDelete: () -> Unit,
    onDismiss: () -> Unit,
) {
    if (!visible) return
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Delete conversation?") },
        text = { Text("This conversation will be permanently deleted. This cannot be undone.") },
        confirmButton = {
            TextButton(onClick = onDelete) {
                Text("Delete", color = AuraThemeTokens.colors.error)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SourcesSheet(citations: List<Citation>, onDismiss: () -> Unit) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = "Sources",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            LazyColumn {
                items(citations, key = { it.url ?: it.title ?: it.hashCode().toString() }) { citation ->
                    Surface(
                        color = AuraThemeTokens.colors.surface1,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(
                                text = citation.title ?: citation.url ?: "Source",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = AuraThemeTokens.colors.textPrimary,
                            )
                            citation.url?.let { url ->
                                Text(
                                    text = url,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = AuraThemeTokens.colors.actionPrimary,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun PermissionDialog(
    permission: String?,
    rationale: String?,
    onGrant: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    if (permission == null) return
    val context = LocalContext.current
    val isNotificationAccess = permission == "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE"
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) onGrant(permission) else onDismiss()
    }
    val activity = context as? Activity
    val shouldShowRationale = activity?.shouldShowRequestPermissionRationale(permission) ?: false
    val permanentlyDenied = !isNotificationAccess && !shouldShowRationale &&
        ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Permission needed") },
        text = {
            Text(buildString {
                append(rationale ?: "Aura needs access to continue.")
                append("\n\nPermission: $permission")
                if (permanentlyDenied) {
                    append("\n\nThe system won't show the prompt again. Open Settings to enable it manually.")
                }
            })
        },
        confirmButton = {
            when {
                isNotificationAccess -> TextButton(onClick = {
                    context.startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
                    onDismiss()
                }) { Text("Open Notification access") }
                permanentlyDenied -> TextButton(onClick = {
                    context.startActivity(
                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                            data = Uri.fromParts("package", context.packageName, null)
                            flags = Intent.FLAG_ACTIVITY_NEW_TASK
                        },
                    )
                    onDismiss()
                }) { Text("Open Settings") }
                else -> TextButton(onClick = { permissionLauncher.launch(permission) }) {
                    Text("Grant")
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Deny") }
        },
    )
}
