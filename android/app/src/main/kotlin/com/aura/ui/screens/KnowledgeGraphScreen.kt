package com.aura.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Merge
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.kg.KgNode
import com.aura.kg.NodeType
import com.aura.ui.viewmodel.KnowledgeGraphViewModel
import com.aura.ui.viewmodel.ResolvedKgRelation
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

import com.aura.ui.theme.AuraThemeTokens
private val graphJson = Json { prettyPrint = true }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun KnowledgeGraphScreen(
    onBack: () -> Unit,
    onOpenSourceConversation: (String, Long) -> Unit = { _, _ -> },
    viewModel: KnowledgeGraphViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    var editing by remember { mutableStateOf<KgNode?>(null) }
    var deleting by remember { mutableStateOf<KgNode?>(null) }
    var merging by remember { mutableStateOf<KgNode?>(null) }

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
                    "Knowledge graph",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Inspect and correct what Aura connects",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.62f),
                )
            }
            IconButton(onClick = viewModel::refresh, enabled = !state.loading) {
                Icon(Icons.Filled.Refresh, contentDescription = "Refresh graph")
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            GraphStatCard(
                value = state.stats.nodeCount.toString(),
                label = "entities",
                color = AuraThemeTokens.colors.actionPrimary,
                modifier = Modifier.weight(1f),
            )
            GraphStatCard(
                value = state.stats.edgeCount.toString(),
                label = "relations",
                color = AuraThemeTokens.colors.assistantAccent,
                modifier = Modifier.weight(1f),
            )
        }
        Spacer(Modifier.height(10.dp))

        OutlinedTextField(
            value = state.query,
            onValueChange = viewModel::setQuery,
            leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
            trailingIcon = if (state.query.isNotEmpty()) {
                {
                    IconButton(onClick = { viewModel.setQuery("") }) {
                        Icon(Icons.Filled.Close, contentDescription = "Clear search")
                    }
                }
            } else null,
            placeholder = { Text("Search labels, types, or properties") },
            singleLine = true,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            TypeChip(null, "All", state.typeFilter == null, viewModel::setTypeFilter)
            NodeType.entries.filterNot { it == NodeType.UNKNOWN }.forEach { type ->
                TypeChip(type, type.displayLabel(), state.typeFilter == type, viewModel::setTypeFilter)
            }
        }
        Spacer(Modifier.height(8.dp))

        state.error?.let { message ->
            Surface(
                color = AuraThemeTokens.colors.error,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
            ) {
                Row(
                    modifier = Modifier.padding(start = 12.dp, end = 4.dp, top = 6.dp, bottom = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        message,
                        style = MaterialTheme.typography.bodySmall,
                        color = AuraThemeTokens.colors.textPrimary,
                        modifier = Modifier.weight(1f),
                    )
                    IconButton(onClick = viewModel::clearError) {
                        Icon(Icons.Filled.Close, contentDescription = "Dismiss error")
                    }
                }
            }
        }

        when {
            state.loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            state.nodes.isEmpty() -> GraphEmptyState(filtered = state.query.isNotBlank() || state.typeFilter != null)
            else -> LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxSize(),
            ) {
                items(state.nodes, key = { it.id }) { node ->
                    GraphNodeCard(node = node, onClick = { viewModel.selectNode(node) })
                }
                item { Spacer(Modifier.height(12.dp)) }
            }
        }
    }

    state.selected?.let { selected ->
        ModalBottomSheet(onDismissRequest = viewModel::dismissNode) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TypeDot(selected.node.type)
                    Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) {
                        Text(
                            selected.node.label,
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            selected.node.type.displayLabel(),
                            style = MaterialTheme.typography.labelMedium,
                            color = AuraThemeTokens.colors.actionPrimary,
                        )
                    }
                }

                if (selected.node.properties.isNotEmpty()) {
                    DetailSection("Properties") {
                        Text(
                            graphJson.encodeToString(JsonObject.serializer(), selected.node.properties),
                            style = MaterialTheme.typography.bodySmall,
                            color = AuraThemeTokens.colors.textPrimary,
                        )
                    }
                }
                if (selected.node.sourceConversationId.isNotBlank()) {
                    OutlinedButton(
                        onClick = {
                            viewModel.dismissNode()
                            onOpenSourceConversation(
                                selected.node.sourceConversationId,
                                selected.node.sourceTurnTimestamp,
                            )
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("Open source conversation")
                    }
                }
                RelationSection("Incoming", selected.incoming)
                RelationSection("Outgoing", selected.outgoing)

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    OutlinedButton(
                        onClick = { editing = selected.node },
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Filled.Edit, contentDescription = null, modifier = Modifier.size(17.dp))
                        Spacer(Modifier.width(5.dp))
                        Text("Edit")
                    }
                    OutlinedButton(
                        onClick = { merging = selected.node },
                        enabled = state.allNodes.size > 1,
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Filled.Merge, contentDescription = null, modifier = Modifier.size(17.dp))
                        Spacer(Modifier.width(5.dp))
                        Text("Merge")
                    }
                    OutlinedButton(
                        onClick = { deleting = selected.node },
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(
                            Icons.Filled.Delete,
                            contentDescription = null,
                            tint = AuraThemeTokens.colors.error,
                            modifier = Modifier.size(17.dp),
                        )
                    }
                }
            }
        }
    }

    editing?.let { node ->
        EditGraphNodeDialog(
            node = node,
            busy = state.mutating,
            onDismiss = { editing = null },
            onSave = { label, type, properties ->
                viewModel.updateNode(node.id, label, type, properties)
                editing = null
            },
        )
    }

    merging?.let { source ->
        MergeGraphNodeDialog(
            source = source,
            candidates = state.allNodes.filterNot { it.id == source.id },
            busy = state.mutating,
            onDismiss = { merging = null },
            onMerge = { target ->
                viewModel.mergeNode(source.id, target.id)
                merging = null
            },
        )
    }

    deleting?.let { node ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete ${node.label}?") },
            text = { Text("The entity and every relation connected to it will be permanently removed.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteNode(node.id)
                        deleting = null
                    },
                    enabled = !state.mutating,
                ) { Text("Delete", color = AuraThemeTokens.colors.error) }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun GraphStatCard(value: String, label: String, color: Color, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        color = color.copy(alpha = 0.10f),
        shape = RoundedCornerShape(16.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, color = color)
            Spacer(Modifier.width(8.dp))
            Text(label, style = MaterialTheme.typography.bodySmall, color = AuraThemeTokens.colors.textPrimary)
        }
    }
}

@Composable
private fun TypeChip(type: NodeType?, label: String, selected: Boolean, onSelect: (NodeType?) -> Unit) {
    AssistChip(
        onClick = { onSelect(type) },
        label = { Text(label) },
        colors = if (selected) {
            AssistChipDefaults.assistChipColors(
                containerColor = AuraThemeTokens.colors.actionPrimary,
                labelColor = AuraThemeTokens.colors.onActionPrimary,
            )
        } else AssistChipDefaults.assistChipColors(),
    )
}

@Composable
private fun GraphNodeCard(node: KgNode, onClick: () -> Unit) {
    Surface(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        color = AuraThemeTokens.colors.surface1.copy(alpha = 0.34f),
        shape = RoundedCornerShape(16.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 13.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TypeDot(node.type)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    node.label,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    buildString {
                        append(node.type.displayLabel())
                        if (node.properties.isNotEmpty()) append(" · ${node.properties.size} properties")
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.72f),
                )
            }
            Text(
                "${(node.confidence * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.62f),
            )
        }
    }
}

@Composable
private fun TypeDot(type: NodeType) {
    Box(
        modifier = Modifier
            .size(12.dp)
            .background(type.color(), CircleShape),
    )
}

@Composable
private fun GraphEmptyState(filtered: Boolean) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                if (filtered) "No matching entities" else "Your graph is still quiet",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                if (filtered) "Clear the search or choose another type."
                else "As you chat, Aura will connect people, projects, tools, and ideas here.",
                style = MaterialTheme.typography.bodyMedium,
                color = AuraThemeTokens.colors.textPrimary.copy(alpha = 0.62f),
            )
        }
    }
}

@Composable
private fun DetailSection(title: String, content: @Composable () -> Unit) {
    Surface(
        color = AuraThemeTokens.colors.surface1.copy(alpha = 0.38f),
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(title, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            content()
        }
    }
}

@Composable
private fun RelationSection(title: String, relations: List<ResolvedKgRelation>) {
    DetailSection("$title · ${relations.size}") {
        if (relations.isEmpty()) {
            Text("No ${title.lowercase()} relations", style = MaterialTheme.typography.bodySmall)
        } else {
            relations.forEachIndexed { index, relation ->
                if (index > 0) HorizontalDivider(Modifier.padding(vertical = 7.dp))
                Text(
                    "${relation.edge.type.name.lowercase().replace('_', ' ')}  →  ${relation.otherLabel}",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

@Composable
private fun EditGraphNodeDialog(
    node: KgNode,
    busy: Boolean,
    onDismiss: () -> Unit,
    onSave: (String, NodeType, JsonObject) -> Unit,
) {
    var label by remember(node.id) { mutableStateOf(node.label) }
    var type by remember(node.id) { mutableStateOf(node.type) }
    var propertiesText by remember(node.id) {
        mutableStateOf(graphJson.encodeToString(JsonObject.serializer(), node.properties))
    }
    var validationError by remember { mutableStateOf<String?>(null) }
    var typeMenu by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit entity") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = label,
                    onValueChange = { label = it },
                    label = { Text("Label") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Box {
                    OutlinedButton(onClick = { typeMenu = true }, modifier = Modifier.fillMaxWidth()) {
                        Text("Type · ${type.displayLabel()}")
                    }
                    androidx.compose.material3.DropdownMenu(
                        expanded = typeMenu,
                        onDismissRequest = { typeMenu = false },
                    ) {
                        NodeType.entries.forEach { option ->
                            androidx.compose.material3.DropdownMenuItem(
                                text = { Text(option.displayLabel()) },
                                onClick = { type = option; typeMenu = false },
                            )
                        }
                    }
                }
                OutlinedTextField(
                    value = propertiesText,
                    onValueChange = { propertiesText = it; validationError = null },
                    label = { Text("Properties JSON") },
                    minLines = 4,
                    maxLines = 8,
                    isError = validationError != null,
                    supportingText = validationError?.let { message -> { Text(message) } },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    val parsed = runCatching {
                        Json.parseToJsonElement(propertiesText) as? JsonObject
                            ?: error("Properties must be a JSON object")
                    }
                    parsed.onSuccess { properties ->
                        if (label.isBlank()) validationError = "Label cannot be blank"
                        else onSave(label.trim(), type, properties)
                    }.onFailure { validationError = it.message ?: "Invalid JSON" }
                },
                enabled = !busy,
            ) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun MergeGraphNodeDialog(
    source: KgNode,
    candidates: List<KgNode>,
    busy: Boolean,
    onDismiss: () -> Unit,
    onMerge: (KgNode) -> Unit,
) {
    var query by remember { mutableStateOf("") }
    var target by remember { mutableStateOf<KgNode?>(null) }
    val visible = candidates.filter {
        query.isBlank() || it.label.contains(query, ignoreCase = true) ||
            it.type.name.contains(query, ignoreCase = true)
    }.take(20)

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Merge ${source.label}") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Choose the entity to keep. Properties and relations from ${source.label} will move into it.",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedTextField(
                    value = query,
                    onValueChange = { query = it },
                    placeholder = { Text("Find target") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                LazyColumn(modifier = Modifier.height(220.dp)) {
                    items(visible, key = { it.id }) { candidate ->
                        Surface(
                            color = if (target?.id == candidate.id) AuraThemeTokens.colors.actionPrimary
                            else Color.Transparent,
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier.fillMaxWidth().clickable { target = candidate },
                        ) {
                            Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
                                TypeDot(candidate.type)
                                Spacer(Modifier.width(8.dp))
                                Column {
                                    Text(candidate.label, fontWeight = FontWeight.Medium)
                                    Text(candidate.type.displayLabel(), style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = { target?.let(onMerge) }, enabled = target != null && !busy) {
                Text("Merge into ${target?.label ?: "target"}")
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

private fun NodeType.displayLabel(): String = name.lowercase().replace('_', ' ').replaceFirstChar { it.uppercase() }

@Composable
private fun NodeType.color(): Color = when (this) {
    NodeType.PERSON -> AuraThemeTokens.colors.actionPrimary
    NodeType.PROJECT -> AuraThemeTokens.colors.assistantAccent
    NodeType.TOOL, NodeType.SKILL -> AuraThemeTokens.colors.assistantAccent
    NodeType.EVENT -> AuraThemeTokens.colors.error
    NodeType.LOCATION -> Color(0xFF4DB6AC)
    NodeType.EMOTION -> Color(0xFFE573A9)
    else -> AuraThemeTokens.colors.textPrimary.copy(alpha = 0.68f)
}
