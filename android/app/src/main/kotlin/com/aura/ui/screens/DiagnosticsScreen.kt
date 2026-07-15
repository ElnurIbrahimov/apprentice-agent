package com.aura.ui.screens

import android.content.Intent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.core.error.CrashLogEntry
import com.aura.ui.viewmodel.DiagnosticsViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

import com.aura.ui.theme.AuraThemeTokens
@Composable
fun DiagnosticsScreen(
    onBack: () -> Unit,
    viewModel: DiagnosticsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    var showClearConfirm by remember { mutableStateOf(false) }

    LaunchedEffect(state.exportFile) {
        val file = state.exportFile ?: return@LaunchedEffect
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        val share = Intent(Intent.ACTION_SEND).apply {
            type = "application/x-ndjson"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(share, "Share Aura diagnostics"))
        viewModel.consumeExport()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 20.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp, bottom = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack, modifier = Modifier.size(42.dp)) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
            }
            Spacer(Modifier.width(4.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    "Diagnostics",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "${state.entries.size} local ${if (state.entries.size == 1) "entry" else "entries"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.62f),
                )
            }
            IconButton(onClick = viewModel::refresh, enabled = !state.loading) {
                Icon(Icons.Filled.Refresh, contentDescription = "Refresh diagnostics")
            }
        }

        Surface(
            color = AuraThemeTokens.colors.actionPrimary.copy(alpha = 0.42f),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(
                modifier = Modifier.padding(13.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Icon(
                    Icons.Filled.BugReport,
                    contentDescription = null,
                    tint = AuraThemeTokens.colors.actionPrimary,
                    modifier = Modifier.size(20.dp),
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    "Logs stay on this device. Aura never uploads them; they leave only when you tap Share.",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.onActionPrimary,
                )
            }
        }
        Spacer(Modifier.height(10.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Button(
                onClick = viewModel::prepareExport,
                enabled = !state.exporting,
                modifier = Modifier.weight(1f),
            ) {
                if (state.exporting) {
                    CircularProgressIndicator(Modifier.size(17.dp), strokeWidth = 2.dp)
                } else {
                    Icon(Icons.Filled.Share, contentDescription = null, modifier = Modifier.size(17.dp))
                }
                Spacer(Modifier.width(6.dp))
                Text(if (state.exporting) "Preparing…" else "Share log")
            }
            OutlinedButton(
                onClick = { showClearConfirm = true },
                enabled = state.entries.isNotEmpty(),
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    Icons.Filled.Delete,
                    contentDescription = null,
                    tint = AuraThemeTokens.colors.error,
                    modifier = Modifier.size(17.dp),
                )
                Spacer(Modifier.width(6.dp))
                Text("Clear", color = AuraThemeTokens.colors.error)
            }
        }
        Spacer(Modifier.height(10.dp))

        state.error?.let { error ->
            Surface(
                color = AuraThemeTokens.colors.error,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
            ) {
                Text(
                    error,
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.textPrimary,
                )
            }
        }

        when {
            state.loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            state.entries.isEmpty() -> DiagnosticsEmptyState()
            else -> LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(state.entries, key = { "${it.timestamp}:${it.code}:${it.message.hashCode()}" }) { entry ->
                    DiagnosticCard(entry)
                }
                item { Spacer(Modifier.height(12.dp)) }
            }
        }
    }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text("Clear diagnostic history?") },
            text = { Text("This permanently deletes ${state.entries.size} local entries. Export first if you need them.") },
            confirmButton = {
                TextButton(onClick = {
                    showClearConfirm = false
                    viewModel.clearAll()
                }) { Text("Clear all", color = AuraThemeTokens.colors.error) }
            },
            dismissButton = { TextButton(onClick = { showClearConfirm = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun DiagnosticCard(entry: CrashLogEntry) {
    var expanded by remember(entry.timestamp, entry.code) { mutableStateOf(false) }
    val canExpand = !entry.stackTrace.isNullOrBlank()
    val accent = if (entry.fatal) AuraThemeTokens.colors.error else AuraThemeTokens.colors.assistantAccent

    Surface(
        color = AuraThemeTokens.colors.surface1.copy(alpha = 0.38f),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .then(if (canExpand) Modifier.clickable { expanded = !expanded } else Modifier),
    ) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(color = accent.copy(alpha = 0.14f), shape = CircleShape) {
                    Text(
                        if (entry.fatal) "FATAL" else "ERROR",
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        style = MaterialTheme.typography.labelSmall,
                        color = accent,
                        fontWeight = FontWeight.Bold,
                    )
                }
                Spacer(Modifier.width(8.dp))
                Text(
                    entry.code,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (canExpand) {
                    Icon(
                        if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                        contentDescription = if (expanded) "Collapse stack trace" else "Expand stack trace",
                        modifier = Modifier.size(20.dp),
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Text(entry.message, style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(7.dp))
            Text(
                buildString {
                    append(DIAGNOSTIC_DATE_FORMAT.format(Date(entry.timestamp)))
                    if (entry.threadName.isNotBlank()) append(" · ${entry.threadName}")
                },
                style = MaterialTheme.typography.bodySmall,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.68f),
            )
            if (expanded) {
                HorizontalDivider(Modifier.padding(vertical = 10.dp))
                Text(
                    entry.stackTrace.orEmpty(),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = AuraThemeTokens.colors.textPrimary,
                )
            }
        }
    }
}

@Composable
private fun DiagnosticsEmptyState() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Surface(
                color = AuraThemeTokens.colors.actionPrimary.copy(alpha = 0.10f),
                shape = CircleShape,
            ) {
                Icon(
                    Icons.Filled.BugReport,
                    contentDescription = null,
                    tint = AuraThemeTokens.colors.actionPrimary,
                    modifier = Modifier.padding(16.dp).size(28.dp),
                )
            }
            Spacer(Modifier.height(12.dp))
            Text("No diagnostic entries", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(5.dp))
            Text(
                "That is the good kind of empty.",
                style = MaterialTheme.typography.bodyMedium,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.62f),
            )
        }
    }
}

private val DIAGNOSTIC_DATE_FORMAT = SimpleDateFormat("MMM d, yyyy · HH:mm:ss", Locale.getDefault())
