package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.QuestionMark
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.agent.RecallSummary

/**
 * A small chip rendered below an assistant turn that summarizes
 * what Aura recalled from its long-term stores for that turn.
 *
 * Format: "Used 3 memories · 1 hand" or "No memories used for this
 * answer" when recall was performed but found nothing.
 *
 * Tapping the chip opens a bottom sheet listing the recalled
 * memory/hand IDs. The chat UI wires the sheet to the actual
 * content via [recalledMemoryContents] and [recalledHandContents]
 * so the model layer can lazy-load from the store.
 *
 * Hidden when [recall] is null (incognito mode or any turn where
 * memoryEnabled was false).
 */
@Composable
fun MemoryRecallChip(
    recall: RecallSummary,
    recalledMemoryContents: List<RecalledMemory> = emptyList(),
    recalledHandContents: List<RecalledHand> = emptyList(),
    modifier: Modifier = Modifier,
) {
    var sheetOpen by remember { mutableStateOf(false) }
    val summary = recall.summary()
    val isEmpty = recall.memoryIds.isEmpty() && recall.handIds.isEmpty()
    val container = if (isEmpty) {
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    } else {
        MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.6f)
    }
    val content = if (isEmpty) {
        MaterialTheme.colorScheme.onSurfaceVariant
    } else {
        MaterialTheme.colorScheme.onTertiaryContainer
    }
    Surface(
        color = container,
        shape = RoundedCornerShape(12.dp),
        modifier = modifier
            .fillMaxWidth(0.85f)
            .padding(start = 32.dp, end = 16.dp, top = 4.dp, bottom = 4.dp)
            .clickable { sheetOpen = true },
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(color = content.copy(alpha = 0.5f), shape = CircleShape),
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = summary,
                style = MaterialTheme.typography.labelSmall,
                color = content,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
    if (sheetOpen) {
        MemoryRecallSheet(
            recall = recall,
            memories = recalledMemoryContents,
            hands = recalledHandContents,
            onDismiss = { sheetOpen = false },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MemoryRecallSheet(
    recall: RecallSummary,
    memories: List<RecalledMemory>,
    hands: List<RecalledHand>,
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = recall.summary(),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            if (memories.isEmpty() && hands.isEmpty()) {
                Text(
                    text = "Aura looked at its memories for this turn but found nothing relevant.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (memories.isNotEmpty()) {
                Text(
                    text = "Memories (${memories.size})",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(4.dp))
                LazyColumn {
                    items(memories) { mem ->
                        RecallRow(
                            icon = Icons.Filled.Memory,
                            title = "[${mem.category}] ${mem.content.take(80)}",
                            subtitle = mem.content.drop(80).take(120),
                        )
                    }
                }
            }
            if (hands.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Hands (${hands.size})",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(4.dp))
                LazyColumn {
                    items(hands) { hand ->
                        RecallRow(
                            icon = Icons.Filled.QuestionMark,
                            title = hand.name,
                            subtitle = hand.description,
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(20.dp))
        }
    }
}

@Composable
private fun RecallRow(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String?,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(16.dp).padding(top = 2.dp),
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
            )
            if (!subtitle.isNullOrBlank()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

/** Lightweight DTO for memory content. Caller (ChatScreen) builds this list from the memory store. */
data class RecalledMemory(val id: String, val category: String, val content: String)
data class RecalledHand(val id: String, val name: String, val description: String)

private fun RecallSummary.summary(): String {
    val m = memoryIds.size
    val h = handIds.size
    return when {
        m == 0 && h == 0 -> "No memories used for this answer"
        h == 0 -> "Used $m memor${if (m == 1) "y" else "ies"}"
        m == 0 -> "Triggered $h hand${if (h == 1) "" else "s"}"
        else -> "Used $m memor${if (m == 1) "y" else "ies"} · $h hand${if (h == 1) "" else "s"}"
    }
}
