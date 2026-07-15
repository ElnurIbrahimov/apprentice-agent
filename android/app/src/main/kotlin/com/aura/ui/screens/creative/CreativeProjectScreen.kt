package com.aura.ui.screens.creative

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircleOutline
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.creative.CreativeMode
import com.aura.creative.CreativeProject
import com.aura.creative.WritingTemplates
import com.aura.ui.components.AuraScreenShell
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens
import com.aura.ui.viewmodel.CreativeStudioViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreativeProjectScreen(
    projectId: String,
    onBack: () -> Unit,
    viewModel: CreativeStudioViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val project = state.selectedProject
    var selectedTab by remember { mutableIntStateOf(0) }
    var showEdit by remember { mutableStateOf(false) }

    LaunchedEffect(projectId) { viewModel.loadProject(projectId) }

    if (state.loading || project == null) {
        Column(
            Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (state.loading) CircularProgressIndicator() else Text(state.error ?: "Project not found")
            TextButton(onClick = onBack) { Text("Back to Studio") }
        }
        return
    }

    AuraScreenShell(
        title = project.name,
        subtitle = listOfNotNull(
            project.genre.takeIf(String::isNotBlank),
            project.tone.takeIf(String::isNotBlank),
            WritingTemplates.byId(project.templateId)?.name,
        ).joinToString(" · ").ifBlank { "Open creative project" },
        action = {
            IconButton(onClick = { showEdit = true }) {
                Icon(Icons.Filled.Edit, contentDescription = "Edit project")
            }
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back to Studio")
            }
        },
    ) { padding ->
        Column(Modifier.fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = selectedTab) {
                listOf("World", "Write", "Simulate").forEachIndexed { index, label ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(label) },
                    )
                }
            }
            if (state.error != null || state.message != null) {
                Surface(
                    color = if (state.error != null) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.primaryContainer,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Row(Modifier.padding(AuraSpacing.sm), verticalAlignment = Alignment.CenterVertically) {
                        Text(state.error ?: state.message.orEmpty(), modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                        TextButton(onClick = viewModel::clearNotice) { Text("Dismiss") }
                    }
                }
            }
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    top = AuraSpacing.md,
                    bottom = padding.calculateBottomPadding() + AuraSpacing.xl,
                ),
                verticalArrangement = Arrangement.spacedBy(AuraSpacing.md),
            ) {
                item {
                    when (selectedTab) {
                        0 -> WorldBibleEditor(project = project, onSave = viewModel::saveWorld)
                        1 -> WritingRoom(state.output, state.generating, viewModel::generate, viewModel::cancelGeneration)
                        else -> SimulationRoom(project, state.output, state.generating, viewModel::generate, viewModel::cancelGeneration, viewModel::canonizeSimulation)
                    }
                }
            }
        }
    }

    if (showEdit) {
        ProjectMetadataDialog(
            project = project,
            onDismiss = { showEdit = false },
            onSave = { name, description, genre, tone, template ->
                showEdit = false
                viewModel.saveMetadata(name, description, genre, tone, template)
            },
        )
    }
}

@Composable
private fun WritingRoom(
    output: String,
    generating: Boolean,
    onGenerate: (CreativeMode, String, String) -> Unit,
    onCancel: () -> Unit,
) {
    var mode by remember { mutableStateOf(CreativeMode.DRAFT) }
    var prompt by remember { mutableStateOf("") }
    Column(verticalArrangement = Arrangement.spacedBy(AuraSpacing.md)) {
        Text("Writing room", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
        Text(
            "The selected model receives this project's current canon every time. Your draft stays exploratory until you add facts to the World tab.",
            style = MaterialTheme.typography.bodySmall,
            color = AuraThemeTokens.colors.textSecondary,
        )
        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            items(CreativeMode.entries.filter { it != CreativeMode.SIMULATE }) { item ->
                FilterChip(selected = mode == item, onClick = { mode = item }, label = { Text(item.label) })
            }
        }
        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            modifier = Modifier.fillMaxWidth(),
            minLines = 5,
            label = { Text(promptLabel(mode)) },
            supportingText = { Text(mode.instruction) },
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.align(Alignment.End)) {
            if (generating) {
                OutlinedButton(onClick = onCancel) {
                    Icon(Icons.Filled.Stop, contentDescription = null)
                    Text("Stop")
                }
            }
            Button(enabled = prompt.isNotBlank() && !generating, onClick = { onGenerate(mode, prompt, "") }) {
                Text(if (mode == CreativeMode.CONTINUITY) "Check continuity" else mode.label)
            }
        }
        GenerationOutput(output = output, generating = generating)
    }
}

@Composable
private fun SimulationRoom(
    project: CreativeProject,
    output: String,
    generating: Boolean,
    onGenerate: (CreativeMode, String, String) -> Unit,
    onCancel: () -> Unit,
    onCanonize: (String) -> Unit,
) {
    var premise by remember { mutableStateOf("") }
    var perspective by remember { mutableStateOf("") }
    Column(verticalArrangement = Arrangement.spacedBy(AuraSpacing.md)) {
        Text("Scenario simulator", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
        Text(
            "Explore alternate decisions without contaminating canon. Aura traces first- and second-order consequences against the world bible.",
            style = MaterialTheme.typography.bodySmall,
            color = AuraThemeTokens.colors.textSecondary,
        )
        OutlinedTextField(
            value = premise,
            onValueChange = { premise = it },
            modifier = Modifier.fillMaxWidth(),
            minLines = 4,
            label = { Text("What if…") },
            placeholder = { Text("What if the protagonist accepts the antagonist's offer?") },
        )
        OutlinedTextField(
            value = perspective,
            onValueChange = { perspective = it },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            label = { Text("Perspective character (optional)") },
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.align(Alignment.End)) {
            if (generating) OutlinedButton(onClick = onCancel) { Text("Stop") }
            Button(enabled = premise.isNotBlank() && !generating, onClick = { onGenerate(CreativeMode.SIMULATE, premise, perspective) }) {
                Text("Run scenario")
            }
        }
        GenerationOutput(output = output, generating = generating)
        if (project.world.simulations.isNotEmpty()) {
            Text("Simulation history", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            project.world.simulations.forEach { simulation ->
                val colors = AuraThemeTokens.colors
                Surface(
                    color = colors.surface1,
                    shape = MaterialTheme.shapes.medium,
                    border = BorderStroke(1.dp, colors.borderSubtle),
                ) {
                    Column(Modifier.padding(AuraSpacing.md), verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs)) {
                        Text(simulation.premise, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                        Text(simulation.outcome, style = MaterialTheme.typography.bodySmall, maxLines = 8)
                        if (simulation.canonized) {
                            Text("CANON", color = colors.actionPrimary, style = MaterialTheme.typography.labelSmall)
                        } else {
                            TextButton(onClick = { onCanonize(simulation.id) }, modifier = Modifier.align(Alignment.End)) {
                                Icon(Icons.Filled.CheckCircleOutline, contentDescription = null)
                                Text("Add outcome to canon timeline")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun GenerationOutput(output: String, generating: Boolean) {
    if (output.isBlank() && !generating) return
    val colors = AuraThemeTokens.colors
    Surface(
        color = colors.surface1,
        shape = MaterialTheme.shapes.medium,
        border = BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Column(Modifier.padding(AuraSpacing.md), verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs)) {
            Text(if (generating) "Aura is writing…" else "Result", style = MaterialTheme.typography.labelLarge, color = colors.actionPrimary)
            if (generating && output.isBlank()) CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally))
            SelectionContainer { Text(output, style = MaterialTheme.typography.bodyMedium) }
        }
    }
}

private fun promptLabel(mode: CreativeMode): String = when (mode) {
    CreativeMode.BRAINSTORM -> "Question, problem, or possibility space"
    CreativeMode.OUTLINE -> "Story premise, arc, or section to outline"
    CreativeMode.DRAFT -> "Scene goal, viewpoint, conflict, and desired length"
    CreativeMode.REWRITE -> "Paste text and describe what should improve"
    CreativeMode.CONTINUITY -> "Paste the passage to audit against canon"
    CreativeMode.SIMULATE -> "Scenario premise"
}

@Composable
private fun ProjectMetadataDialog(
    project: CreativeProject,
    onDismiss: () -> Unit,
    onSave: (String, String, String, String, String) -> Unit,
) {
    var name by remember { mutableStateOf(project.name) }
    var description by remember { mutableStateOf(project.description) }
    var genre by remember { mutableStateOf(project.genre) }
    var tone by remember { mutableStateOf(project.tone) }
    var template by remember { mutableStateOf(project.templateId) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Project details") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(name, { name = it }, label = { Text("Name") }, singleLine = true)
                OutlinedTextField(description, { description = it }, label = { Text("Premise") }, minLines = 2)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(genre, { genre = it }, label = { Text("Genre") }, modifier = Modifier.weight(1f))
                    OutlinedTextField(tone, { tone = it }, label = { Text("Tone") }, modifier = Modifier.weight(1f))
                }
                WritingTemplates.all.forEach { form ->
                    FilterChip(selected = template == form.id, onClick = { template = form.id }, label = { Text("${form.icon} ${form.name}") })
                }
            }
        },
        confirmButton = { Button(enabled = name.isNotBlank(), onClick = { onSave(name, description, genre, tone, template) }) { Text("Save") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}