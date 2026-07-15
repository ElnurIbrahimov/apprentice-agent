package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.aura.documents.DocumentEntity
import com.aura.ui.viewmodel.DocumentImportUiState

import com.aura.ui.theme.AuraThemeTokens
@Composable
internal fun DocumentLibraryDialog(
    state: DocumentImportUiState,
    onImport: () -> Unit,
    onOpen: (DocumentEntity) -> Unit,
    onDelete: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var pendingDelete by remember { mutableStateOf<DocumentEntity?>(null) }

    AlertDialog(
        onDismissRequest = { if (!state.importing) onDismiss() },
        icon = { Icon(Icons.Filled.Description, contentDescription = null) },
        title = { Text("Document memory") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    "Import local documents into Aura's searchable memory. Text stays on-device except embeddings sent through your configured embedding provider.",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.textPrimary,
                )
                Button(
                    onClick = onImport,
                    enabled = !state.importing,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    if (state.importing) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp,
                        )
                        Text("  ${state.stage ?: "Importing…"}")
                    } else {
                        Icon(Icons.Filled.UploadFile, contentDescription = null)
                        Text("  Import PDF, DOCX, or text")
                    }
                }
                if (state.documents.isEmpty()) {
                    Text(
                        "No documents imported yet.",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(vertical = 12.dp),
                    )
                } else {
                    LazyColumn(
                        modifier = Modifier.heightIn(max = 380.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        items(state.documents, key = { it.id }) { document ->
                            Surface(
                                shape = RoundedCornerShape(14.dp),
                                color = AuraThemeTokens.colors.surface1.copy(alpha = 0.45f),
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(start = 12.dp, top = 10.dp, bottom = 10.dp, end = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Icon(
                                        Icons.Filled.Description,
                                        contentDescription = null,
                                        tint = AuraThemeTokens.colors.actionPrimary,
                                    )
                                    Column(
                                        modifier = Modifier
                                            .weight(1f)
                                            .padding(horizontal = 10.dp),
                                    ) {
                                        Text(
                                            document.name,
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = FontWeight.SemiBold,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                        )
                                        Text(
                                            "${document.chunkCount} chunks · ${formatCharacterCount(document.characterCount)}",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = AuraThemeTokens.colors.textPrimary,
                                        )
                                    }
                                    IconButton(onClick = { onOpen(document) }) {
                                        Icon(Icons.Filled.FileOpen, contentDescription = "Open document")
                                    }
                                    IconButton(onClick = { pendingDelete = document }) {
                                        Icon(
                                            Icons.Filled.Delete,
                                            contentDescription = "Delete document memory",
                                            tint = AuraThemeTokens.colors.error,
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss, enabled = !state.importing) { Text("Done") }
        },
    )

    pendingDelete?.let { document ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("Forget ${document.name}?") },
            text = { Text("Aura will delete the document record and all ${document.chunkCount} searchable memory chunks created from it.") },
            confirmButton = {
                TextButton(onClick = {
                    onDelete(document.id)
                    pendingDelete = null
                }) { Text("Forget", color = AuraThemeTokens.colors.error) }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("Cancel") }
            },
        )
    }
}

internal fun formatCharacterCount(count: Int): String = when {
    count >= 1_000_000 -> "%.1fM chars".format(count / 1_000_000f)
    count >= 1_000 -> "%.1fK chars".format(count / 1_000f)
    else -> "$count chars"
}