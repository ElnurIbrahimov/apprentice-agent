package com.aura.ui.screens.skills

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
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
import com.aura.skills.Skill
import com.aura.ui.viewmodel.SkillsViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SkillsScreen(
    onBack: () -> Unit,
    vm: SkillsViewModel = hiltViewModel(),
) {
    val skills by vm.skills.collectAsState()
    val selectedId by vm.selectedId.collectAsState()
    val selected = skills.firstOrNull { it.id == selectedId }

    var showNew by remember { mutableStateOf(false) }
    var confirmDeleteId by remember { mutableStateOf<String?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Skills") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { showNew = true }) {
                Icon(Icons.Filled.Add, contentDescription = "Add skill")
            }
        },
    ) { padding ->
        if (skills.isEmpty()) {
            EmptyState(padding = padding)
        } else if (selected == null) {
            SkillList(
                padding = padding,
                skills = skills,
                onClick = { vm.select(it.id) },
            )
        } else {
            SkillDetail(
                padding = padding,
                skill = selected,
                onBack = { vm.select(null) },
                onSave = { vm.update(it) },
                onDelete = { confirmDeleteId = it.id },
            )
        }
    }

    if (showNew) {
        NewSkillDialog(
            onDismiss = { showNew = false },
            onCreate = { name, description, body ->
                vm.add(name, description, body)
                showNew = false
            },
        )
    }

    val pendingDelete = confirmDeleteId
    if (pendingDelete != null) {
        AlertDialog(
            onDismissRequest = { confirmDeleteId = null },
            title = { Text("Delete skill?") },
            text = { Text("This removes the skill and its body permanently.") },
            confirmButton = {
                TextButton(onClick = {
                    vm.remove(pendingDelete)
                    confirmDeleteId = null
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { confirmDeleteId = null }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun EmptyState(padding: PaddingValues) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .padding(24.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("No skills yet", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(
                "Skills are reusable instruction modules. The agent can invoke them by name via the use_skill tool.",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}

@Composable
private fun SkillList(
    padding: PaddingValues,
    skills: List<Skill>,
    onClick: (Skill) -> Unit,
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items(items = skills, key = { it.id }) { skill ->
            Card(
                modifier = Modifier
                    .fillMaxWidth(),
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        skill.name,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    if (skill.description.isNotBlank()) {
                        Text(
                            skill.description,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(top = 4.dp),
                        )
                    }
                    Text(
                        skill.preview(),
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 8.dp),
                        maxLines = 2,
                    )
                    FilledTonalButton(
                        onClick = { onClick(skill) },
                        modifier = Modifier
                            .padding(top = 8.dp)
                            .align(Alignment.End),
                    ) { Text("Edit") }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SkillDetail(
    padding: PaddingValues,
    skill: Skill,
    onBack: () -> Unit,
    onSave: (Skill) -> Unit,
    onDelete: (Skill) -> Unit,
) {
    var name by remember(skill.id) { mutableStateOf(skill.name) }
    var description by remember(skill.id) { mutableStateOf(skill.description) }
    var body by remember(skill.id) { mutableStateOf(skill.body) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedTextField(
            value = name,
            onValueChange = { name = it.take(80) },
            label = { Text("Name") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = description,
            onValueChange = { description = it.take(240) },
            label = { Text("Description (optional)") },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = body,
            onValueChange = { body = it },
            label = { Text("Body (markdown)") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 8,
        )
        HorizontalDivider()
        androidx.compose.foundation.layout.Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Button(
                onClick = {
                    val updated = skill.renamed(name).withDescription(description).withBody(body)
                    onSave(updated)
                    onBack()
                },
                modifier = Modifier.weight(1f),
            ) { Text("Save") }
            TextButton(
                onClick = { onDelete(skill) },
                modifier = Modifier.weight(1f),
            ) { Text("Delete") }
        }
    }
}

@Composable
private fun NewSkillDialog(
    onDismiss: () -> Unit,
    onCreate: (name: String, description: String, body: String) -> Unit,
) {
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var body by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New skill") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(value = name, onValueChange = { name = it.take(80) }, label = { Text("Name") }, singleLine = true)
                OutlinedTextField(value = description, onValueChange = { description = it.take(240) }, label = { Text("Description (optional)") }, singleLine = true)
                OutlinedTextField(value = body, onValueChange = { body = it }, label = { Text("Body (markdown)") }, minLines = 6, maxLines = 12)
            }
        },
        confirmButton = {
            TextButton(
                enabled = name.isNotBlank(),
                onClick = { onCreate(name, description, body) },
            ) { Text("Create") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}
