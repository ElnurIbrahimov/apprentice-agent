package com.aura.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.memory.MemoryEditEntity
import com.aura.memory.MemoryEntity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

import com.aura.ui.theme.AuraThemeTokens
internal fun memoryEditHeadline(edit: MemoryEditEntity): String = when {
    edit.oldCategory != edit.newCategory -> "${edit.oldCategory} → ${edit.newCategory}"
    edit.oldContent != edit.newContent -> "Content updated"
    else -> "Memory updated"
}

@Composable
internal fun MemoryHistoryDialog(
    memory: MemoryEntity,
    entries: List<MemoryEditEntity>,
    loading: Boolean,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit history") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    text = memory.content,
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.textPrimary,
                    maxLines = 3,
                )
                when {
                    loading -> CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.CenterHorizontally),
                        strokeWidth = 2.dp,
                    )
                    entries.isEmpty() -> Text(
                        "No edits yet. This memory is still in its original form.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    else -> LazyColumn(
                        modifier = Modifier.heightIn(max = 420.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        items(entries, key = { it.id }) { edit ->
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(14.dp),
                                color = AuraThemeTokens.colors.surface1.copy(alpha = 0.45f),
                            ) {
                                Column(
                                    modifier = Modifier.padding(12.dp),
                                    verticalArrangement = Arrangement.spacedBy(4.dp),
                                ) {
                                    Text(
                                        memoryEditHeadline(edit),
                                        style = MaterialTheme.typography.labelLarge,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Text(
                                        SimpleDateFormat("MMM d, yyyy · HH:mm", Locale.getDefault())
                                            .format(Date(edit.editedAt)),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = AuraThemeTokens.colors.textPrimary,
                                    )
                                    if (edit.oldContent != edit.newContent) {
                                        Text(
                                            "Before: ${edit.oldContent}",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = AuraThemeTokens.colors.textPrimary,
                                        )
                                        Text(
                                            "After: ${edit.newContent}",
                                            style = MaterialTheme.typography.bodySmall,
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
            TextButton(onClick = onDismiss) { Text("Done") }
        },
    )
}