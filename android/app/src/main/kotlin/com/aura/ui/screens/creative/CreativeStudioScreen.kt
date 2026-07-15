package com.aura.ui.screens.creative

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AutoStories
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.aura.creative.CreativeProject
import com.aura.creative.WritingTemplates
import com.aura.ui.components.AuraScreenShell
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens
import com.aura.ui.viewmodel.CreativeStudioViewModel

@Composable
fun CreativeStudioScreen(
    onOpenProject: (String) -> Unit,
    viewModel: CreativeStudioViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    var showCreate by remember { mutableStateOf(false) }
    var deleteProject by remember { mutableStateOf<CreativeProject?>(null) }

    LaunchedEffect(state.createdProjectId) {
        state.createdProjectId?.let { id ->
            viewModel.consumeCreatedProject()
            onOpenProject(id)
        }
    }

    AuraScreenShell(
        title = "Creative Studio",
        subtitle = "World bibles, writing rooms, simulations, and continuity",
        action = {
            IconButton(onClick = { showCreate = true }) {
                Icon(Icons.Filled.Add, contentDescription = "New creative project")
            }
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                top = padding.calculateTopPadding() + AuraSpacing.sm,
                bottom = padding.calculateBottomPadding() + AuraSpacing.xl,
            ),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            item {
                CreativeHero(
                    projectCount = state.projects.size,
                    onCreate = { showCreate = true },
                )
            }
            if (!state.loading && state.projects.isEmpty()) {
                item {
                    Text(
                        "No projects yet. Start with a form, then let the world grow around your own ideas.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = AuraThemeTokens.colors.textSecondary,
                        modifier = Modifier.padding(vertical = AuraSpacing.md),
                    )
                }
            }
            items(state.projects, key = { it.id }) { project ->
                CreativeProjectCard(
                    project = project,
                    onOpen = { onOpenProject(project.id) },
                    onDelete = { deleteProject = project },
                )
            }
        }
    }

    if (showCreate) {
        NewCreativeProjectDialog(
            onDismiss = { showCreate = false },
            onCreate = { name, description, genre, tone, template ->
                showCreate = false
                viewModel.createProject(name, description, genre, tone, template)
            },
        )
    }
    deleteProject?.let { project ->
        AlertDialog(
            onDismissRequest = { deleteProject = null },
            title = { Text("Delete ${project.name}?") },
            text = { Text("The project world bible, outline, and simulations will be removed. This cannot be undone.") },
            confirmButton = {
                Button(onClick = {
                    viewModel.deleteProject(project.id)
                    deleteProject = null
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleteProject = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun CreativeHero(projectCount: Int, onCreate: () -> Unit) {
    val colors = AuraThemeTokens.colors
    Surface(
        color = colors.surface1,
        shape = MaterialTheme.shapes.large,
        border = BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Column(
            Modifier.padding(AuraSpacing.lg),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            Icon(
                Icons.Filled.AutoStories,
                contentDescription = null,
                tint = colors.actionPrimary,
                modifier = Modifier.size(30.dp),
            )
            Text(
                "Build a world that remembers itself.",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                color = colors.textPrimary,
            )
            Text(
                "Keep canon, characters, places, rules, and story beats together. Draft against them, test alternate futures, then canonize only what earns its place.",
                style = MaterialTheme.typography.bodyMedium,
                color = colors.textSecondary,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    if (projectCount == 1) "1 active world" else "$projectCount active worlds",
                    style = MaterialTheme.typography.labelMedium,
                    color = colors.textTertiary,
                )
                Button(onClick = onCreate) {
                    Icon(Icons.Filled.Add, contentDescription = null)
                    Spacer(Modifier.size(6.dp))
                    Text("New project")
                }
            }
        }
    }
}

@Composable
private fun CreativeProjectCard(
    project: CreativeProject,
    onOpen: () -> Unit,
    onDelete: () -> Unit,
) {
    val colors = AuraThemeTokens.colors
    val world = project.world
    val template = WritingTemplates.byId(project.templateId)
    Surface(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onOpen),
        color = colors.surface1,
        shape = MaterialTheme.shapes.medium,
        border = BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Row(
            Modifier.padding(AuraSpacing.md),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            Surface(
                color = colors.actionPrimary.copy(alpha = 0.12f),
                shape = MaterialTheme.shapes.medium,
            ) {
                Text(template?.icon ?: "✍️", modifier = Modifier.padding(12.dp))
            }
            Column(Modifier.weight(1f)) {
                Text(
                    project.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = colors.textPrimary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    listOfNotNull(project.genre.takeIf(String::isNotBlank), project.tone.takeIf(String::isNotBlank)).joinToString(" · ").ifBlank { template?.name ?: "Open canvas" },
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.textSecondary,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "${world.characters.size} characters · ${world.locations.size} places · ${world.rules.size} rules · ${project.turnCount} turns",
                    style = MaterialTheme.typography.labelSmall,
                    color = colors.textTertiary,
                )
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Filled.DeleteOutline, contentDescription = "Delete project", tint = colors.textTertiary)
            }
        }
    }
}

@Composable
private fun NewCreativeProjectDialog(
    onDismiss: () -> Unit,
    onCreate: (String, String, String, String, String) -> Unit,
) {
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var genre by remember { mutableStateOf("") }
    var tone by remember { mutableStateOf("") }
    var templateId by remember { mutableStateOf(WritingTemplates.all.first().id) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New creative project") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(name, { name = it }, label = { Text("Project name") }, singleLine = true)
                OutlinedTextField(description, { description = it }, label = { Text("Premise") }, minLines = 2)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(genre, { genre = it }, label = { Text("Genre") }, modifier = Modifier.weight(1f), singleLine = true)
                    OutlinedTextField(tone, { tone = it }, label = { Text("Tone") }, modifier = Modifier.weight(1f), singleLine = true)
                }
                Text("Form", style = MaterialTheme.typography.labelLarge)
                WritingTemplates.all.forEach { template ->
                    FilterChip(
                        selected = template.id == templateId,
                        onClick = { templateId = template.id },
                        label = { Text("${template.icon}  ${template.name}") },
                    )
                }
            }
        },
        confirmButton = {
            Button(
                enabled = name.isNotBlank(),
                onClick = { onCreate(name, description, genre, tone, templateId) },
            ) { Text("Create") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}