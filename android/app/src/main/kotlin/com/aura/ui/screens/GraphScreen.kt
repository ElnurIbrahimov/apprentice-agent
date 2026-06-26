package com.aura.ui.screens

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Route
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.kg.KgNode
import com.aura.kg.KnowledgeGraphRepository
import com.aura.ui.viewmodel.GraphViewModel
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

private val jsonPretty = Json { prettyPrint = true }

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun GraphScreen(viewModel: GraphViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    var showSheet by remember { mutableStateOf(false) }
    var selectedNodeSet by remember { mutableStateOf(setOf<String>()) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    // Open sheet when selectedNode changes
    val currentSelected = state.selectedNode
    if (currentSelected != null && !showSheet) {
        showSheet = true
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
        ) {
            Text(
                text = "Knowledge Graph",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                text = "${state.nodes.size} nodes in view",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(12.dp))

            // Search bar
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = state.query,
                    onValueChange = { viewModel.onQueryChange(it) },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Search nodes…") },
                    singleLine = true,
                    leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
                    keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                        imeAction = androidx.compose.ui.text.input.ImeAction.Search,
                    ),
                    keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                        onSearch = { viewModel.search() },
                    ),
                )
                Spacer(modifier = Modifier.size(8.dp))
                IconButton(
                    onClick = { viewModel.search() },
                    modifier = Modifier
                        .size(48.dp)
                        .background(MaterialTheme.colorScheme.primary, CircleShape),
                ) {
                    Icon(
                        Icons.Filled.Search,
                        contentDescription = "Search",
                        tint = MaterialTheme.colorScheme.onPrimary,
                    )
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
            HorizontalDivider()
            Spacer(modifier = Modifier.height(8.dp))

            // Loading
            if (state.loading) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            } else if (state.nodes.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "🗂️",
                            style = MaterialTheme.typography.displayLarge,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "No nodes yet",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onBackground,
                        )
                        Text(
                            text = "The knowledge graph populates as you chat with Aura.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                        )
                    }
                }
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(state.nodes, key = { it.id }) { node ->
                        val isSelectedForPath = node.id in selectedNodeSet
                        NodeCard(
                            node = node,
                            isSelectedForPath = isSelectedForPath,
                            selectedCount = selectedNodeSet.size,
                            onClick = {
                                selectedNodeSet = emptySet()
                                viewModel.selectNode(node.id)
                            },
                            onLongClick = {
                                selectedNodeSet = if (node.id in selectedNodeSet) {
                                    selectedNodeSet - node.id
                                } else if (selectedNodeSet.size >= 2) {
                                    setOf(node.id)
                                } else {
                                    selectedNodeSet + node.id
                                }
                            },
                        )
                    }
                }
            }
        }

        // Error banner
        state.error?.let { err ->
            Surface(
                color = MaterialTheme.colorScheme.errorContainer,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                shape = RoundedCornerShape(8.dp),
            ) {
                Text(
                    text = err,
                    modifier = Modifier.padding(12.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        // Find Path FAB when 2 nodes selected
        if (selectedNodeSet.size == 2) {
            FloatingActionButton(
                onClick = {
                    val ids = selectedNodeSet.toList()
                    viewModel.findPath(ids[0], ids[1])
                    selectedNodeSet = emptySet()
                    showSheet = true
                },
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(16.dp),
                containerColor = MaterialTheme.colorScheme.primary,
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Filled.Route,
                        contentDescription = "Find path",
                        tint = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        "Find Path",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                }
            }
        }

        // Selected node count chip
        if (selectedNodeSet.isNotEmpty()) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(end = 16.dp, top = 104.dp),
                shape = RoundedCornerShape(16.dp),
                color = MaterialTheme.colorScheme.secondaryContainer,
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "${selectedNodeSet.size} selected",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    IconButton(
                        onClick = { selectedNodeSet = emptySet() },
                        modifier = Modifier.size(20.dp),
                    ) {
                        Icon(
                            Icons.Filled.Close,
                            contentDescription = "Clear selection",
                            modifier = Modifier.size(14.dp),
                            tint = MaterialTheme.colorScheme.onSecondaryContainer,
                        )
                    }
                }
            }
        }
    }

    // Bottom sheet
    if (showSheet && (state.selectedNode != null || state.path != null)) {
        ModalBottomSheet(
            onDismissRequest = {
                showSheet = false
                viewModel.clearSelection()
            },
            sheetState = sheetState,
            containerColor = MaterialTheme.colorScheme.surface,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 12.dp),
            ) {
                // Path result section
                if (state.path != null) {
                    PathSection(path = state.path!!, onNodeClick = { node ->
                        viewModel.selectNode(node.id)
                    })
                    HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                }

                // Node detail section
                state.selectedNode?.let { node ->
                    NodeDetailSheet(
                        node = node,
                        neighbors = state.neighbors,
                        onRelatedNodeClick = { relatedId ->
                            viewModel.selectNode(relatedId)
                        },
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

// ── Node Card ──────────────────────────────────────────────────────────────

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun NodeCard(
    node: KgNode,
    isSelectedForPath: Boolean,
    selectedCount: Int,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
) {
    val containerColor = when {
        isSelectedForPath -> MaterialTheme.colorScheme.primaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }
    val borderModifier = if (isSelectedForPath) {
        Modifier.padding(2.dp)
    } else {
        Modifier
    }

    Surface(
        color = containerColor,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier
            .fillMaxWidth()
            .then(borderModifier)
            .combinedClickable(
                onClick = onClick,
                onLongClick = onLongClick,
            ),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Type dot
            TypeDot(node.type.name)
            Spacer(modifier = Modifier.size(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = node.label,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(modifier = Modifier.height(2.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    AssistChip(
                        onClick = {},
                        label = {
                            Text(
                                node.type.name.lowercase(),
                                style = MaterialTheme.typography.labelSmall,
                            )
                        },
                        colors = AssistChipDefaults.assistChipColors(
                            containerColor = MaterialTheme.colorScheme.secondaryContainer,
                        ),
                        modifier = Modifier.height(24.dp),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "%.0f%%".format(node.confidence * 100),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                    Text(
                        text = "  ·  ${friendlyTimestamp(node.updatedAt)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                    )
                }
            }
            if (isSelectedForPath) {
                Text(
                    text = "#${selectedCount}",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

// ── Bottom Sheet: Node Detail ───────────────────────────────────────────────

@Composable
private fun NodeDetailSheet(
    node: KgNode,
    neighbors: KnowledgeGraphRepository.Neighbors?,
    onRelatedNodeClick: (String) -> Unit,
) {
    // Label
    Text(
        text = node.label,
        style = MaterialTheme.typography.titleLarge,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.onSurface,
    )
    Spacer(modifier = Modifier.height(4.dp))

    // Type + confidence row
    Row(verticalAlignment = Alignment.CenterVertically) {
        AssistChip(
            onClick = {},
            label = { Text(node.type.name.lowercase()) },
            colors = AssistChipDefaults.assistChipColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
            ),
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "Confidence: %.0f%%".format(node.confidence * 100),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
        )
    }
    Spacer(modifier = Modifier.height(8.dp))

    // Source turn
    if (node.sourceTurnId.isNotBlank()) {
        Text(
            text = "From turn: ${node.sourceTurnId.take(12)}…",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
        )
        Spacer(modifier = Modifier.height(8.dp))
    }

    // Properties
    if (node.properties.isNotEmpty()) {
        Text(
            text = "Properties",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                text = jsonPretty.encodeToString(JsonObject.serializer(), node.properties),
                modifier = Modifier.padding(10.dp),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
    }

    // Edges
    neighbors?.let { n ->
        if (n.outgoing.isNotEmpty()) {
            Text(
                text = "Outgoing edges (${n.outgoing.size})",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Spacer(modifier = Modifier.height(4.dp))
            n.outgoing.groupBy { it.type.name.lowercase() }.forEach { (type, edges) ->
                Text(
                    text = type,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(start = 8.dp, top = 4.dp),
                )
                edges.forEach { edge ->
                    Text(
                        text = "  → ${edge.targetId.take(16)}…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                        modifier = Modifier
                            .clickable { onRelatedNodeClick(edge.targetId) }
                            .padding(start = 16.dp, top = 2.dp, bottom = 2.dp),
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        if (n.incoming.isNotEmpty()) {
            Text(
                text = "Incoming edges (${n.incoming.size})",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Spacer(modifier = Modifier.height(4.dp))
            n.incoming.groupBy { it.type.name.lowercase() }.forEach { (type, edges) ->
                Text(
                    text = type,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(start = 8.dp, top = 4.dp),
                )
                edges.forEach { edge ->
                    Text(
                        text = "  ← ${edge.sourceId.take(16)}…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                        modifier = Modifier
                            .clickable { onRelatedNodeClick(edge.sourceId) }
                            .padding(start = 16.dp, top = 2.dp, bottom = 2.dp),
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

// ── Bottom Sheet: Path Section ──────────────────────────────────────────────

@Composable
private fun PathSection(
    path: List<KgNode>,
    onNodeClick: (KgNode) -> Unit,
) {
    Text(
        text = "Path (${path.size} nodes)",
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.SemiBold,
        color = MaterialTheme.colorScheme.onSurface,
    )
    Spacer(modifier = Modifier.height(8.dp))
    path.forEachIndexed { index, node ->
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .clickable { onNodeClick(node) }
                .padding(vertical = 2.dp),
        ) {
            Text(
                text = "${index + 1}.",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.width(24.dp),
            )
            TypeDot(node.type.name)
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = node.label,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
        if (index < path.lastIndex) {
            Text(
                text = "  ↓",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                modifier = Modifier.padding(start = 10.dp),
            )
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

@Composable
private fun TypeDot(type: String) {
    val color = when (type.lowercase()) {
        "concept" -> MaterialTheme.colorScheme.primary
        "entity" -> MaterialTheme.colorScheme.secondary
        "person" -> MaterialTheme.colorScheme.tertiary
        "project" -> MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
        "tool" -> MaterialTheme.colorScheme.error
        else -> MaterialTheme.colorScheme.outline
    }
    Box(
        modifier = Modifier
            .size(10.dp)
            .background(color, CircleShape),
    )
}

private fun friendlyTimestamp(millis: Long): String {
    val age = (System.currentTimeMillis() - millis) / 1000
    return when {
        age < 60 -> "just now"
        age < 3600 -> "${age / 60}m ago"
        age < 86400 -> "${age / 3600}h ago"
        else -> "${age / 86400}d ago"
    }
}
