package com.aura.ui.screens.creative

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aura.creative.CreativeProject
import com.aura.creative.StoryBeat
import com.aura.creative.WorldBible
import com.aura.creative.WorldCharacter
import com.aura.creative.WorldEvent
import com.aura.creative.WorldFaction
import com.aura.creative.WorldLocation
import com.aura.creative.WorldRule
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

private enum class WorldItemType(val label: String, val detailLabel: String, val extraLabel: String) {
    CHARACTER("Character", "Role", "Motivation"),
    LOCATION("Location", "Type", "Significance"),
    FACTION("Faction", "Alignment / power", "Key members"),
    RULE("Rule", "Category", "Story impact"),
    EVENT("Timeline event", "Date / era", "Participants"),
    BEAT("Story beat", "Status", "Purpose / payoff"),
}

@Composable
fun WorldBibleEditor(
    project: CreativeProject,
    onSave: (WorldBible) -> Unit,
) {
    var world by remember(project.world) { mutableStateOf(project.world) }
    var adding by remember { mutableStateOf<WorldItemType?>(null) }
    val colors = AuraThemeTokens.colors

    Column(verticalArrangement = Arrangement.spacedBy(AuraSpacing.md)) {
        Surface(
            color = colors.surface1,
            shape = MaterialTheme.shapes.medium,
            border = BorderStroke(1.dp, colors.borderSubtle),
        ) {
            Column(Modifier.padding(AuraSpacing.md), verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm)) {
                Text("World foundation", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                OutlinedTextField(
                    value = world.overview,
                    onValueChange = { world = world.copy(overview = it) },
                    label = { Text("What makes this world distinct?") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 3,
                )
                OutlinedTextField(
                    value = world.notes,
                    onValueChange = { world = world.copy(notes = it) },
                    label = { Text("Loose lore and private notes") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                )
                Button(onClick = { onSave(world) }, modifier = Modifier.align(Alignment.End)) {
                    Text("Save foundation")
                }
            }
        }

        WorldSection(
            title = "Characters",
            emptyText = "Who wants something badly enough to change the world?",
            items = world.characters.map { it.id to "${it.name}${it.role.takeIf(String::isNotBlank)?.let { role -> " · $role" }.orEmpty()}" },
            onAdd = { adding = WorldItemType.CHARACTER },
            onRemove = { id -> world = world.copy(characters = world.characters.filterNot { it.id == id }); onSave(world) },
        )
        WorldSection(
            title = "Locations",
            emptyText = "Give scenes places with pressure, history, and consequence.",
            items = world.locations.map { it.id to "${it.name}${it.type.takeIf(String::isNotBlank)?.let { type -> " · $type" }.orEmpty()}" },
            onAdd = { adding = WorldItemType.LOCATION },
            onRemove = { id -> world = world.copy(locations = world.locations.filterNot { it.id == id }); onSave(world) },
        )
        WorldSection(
            title = "Factions",
            emptyText = "Institutions and groups turn private desires into public conflict.",
            items = world.factions.map { it.id to it.name },
            onAdd = { adding = WorldItemType.FACTION },
            onRemove = { id -> world = world.copy(factions = world.factions.filterNot { it.id == id }); onSave(world) },
        )
        WorldSection(
            title = "Rules",
            emptyText = "The best worlds have constraints that create story, not trivia.",
            items = world.rules.map { it.id to "${it.name} · ${it.category}" },
            onAdd = { adding = WorldItemType.RULE },
            onRemove = { id -> world = world.copy(rules = world.rules.filterNot { it.id == id }); onSave(world) },
        )
        WorldSection(
            title = "Timeline",
            emptyText = "Anchor cause and effect before continuity drifts.",
            items = world.timeline.map { it.id to "${it.date.takeIf(String::isNotBlank)?.plus(" · ").orEmpty()}${it.title}" },
            onAdd = { adding = WorldItemType.EVENT },
            onRemove = { id -> world = world.copy(timeline = world.timeline.filterNot { it.id == id }); onSave(world) },
        )
        WorldSection(
            title = "Outline",
            emptyText = "Turn ideas into beats you can actually draft.",
            items = world.outline.map { it.id to "${it.title} · ${it.status}" },
            onAdd = { adding = WorldItemType.BEAT },
            onRemove = { id -> world = world.copy(outline = world.outline.filterNot { it.id == id }); onSave(world) },
        )
    }

    adding?.let { type ->
        AddWorldItemDialog(
            type = type,
            onDismiss = { adding = null },
            onAdd = { name, description, details, extra ->
                world = when (type) {
                    WorldItemType.CHARACTER -> world.copy(
                        characters = world.characters + WorldCharacter(name = name, role = details, backstory = description, motivation = extra),
                    )
                    WorldItemType.LOCATION -> world.copy(
                        locations = world.locations + WorldLocation(name = name, type = details, description = description, significance = extra),
                    )
                    WorldItemType.FACTION -> world.copy(
                        factions = world.factions + WorldFaction(name = name, ideology = description, members = extra.split(',').map(String::trim).filter(String::isNotBlank)),
                    )
                    WorldItemType.RULE -> world.copy(
                        rules = world.rules + WorldRule(name = name, description = description, category = details.ifBlank { "world" }, impact = extra),
                    )
                    WorldItemType.EVENT -> world.copy(
                        timeline = world.timeline + WorldEvent(title = name, date = details, description = description, participants = extra.split(',').map(String::trim).filter(String::isNotBlank)),
                    )
                    WorldItemType.BEAT -> world.copy(
                        outline = world.outline + StoryBeat(title = name, summary = description, status = details.ifBlank { "planned" }),
                    )
                }
                adding = null
                onSave(world)
            },
        )
    }
}

@Composable
private fun WorldSection(
    title: String,
    emptyText: String,
    items: List<Pair<String, String>>,
    onAdd: () -> Unit,
    onRemove: (String) -> Unit,
) {
    val colors = AuraThemeTokens.colors
    Surface(
        color = colors.surface1,
        shape = MaterialTheme.shapes.medium,
        border = BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Column(Modifier.padding(AuraSpacing.md), verticalArrangement = Arrangement.spacedBy(AuraSpacing.xs)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                OutlinedButton(onClick = onAdd) {
                    Icon(Icons.Filled.Add, contentDescription = null)
                    Text("Add")
                }
            }
            if (items.isEmpty()) {
                Text(emptyText, style = MaterialTheme.typography.bodySmall, color = colors.textSecondary)
            } else {
                items.forEach { (id, label) ->
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
                        IconButton(onClick = { onRemove(id) }) {
                            Icon(Icons.Filled.Close, contentDescription = "Remove $label", tint = colors.textTertiary)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AddWorldItemDialog(
    type: WorldItemType,
    onDismiss: () -> Unit,
    onAdd: (String, String, String, String) -> Unit,
) {
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var details by remember { mutableStateOf("") }
    var extra by remember { mutableStateOf("") }
    androidx.compose.material3.AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add ${type.label.lowercase()}") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(name, { name = it }, label = { Text("Name / title") }, singleLine = true)
                OutlinedTextField(description, { description = it }, label = { Text("Description") }, minLines = 3)
                OutlinedTextField(details, { details = it }, label = { Text(type.detailLabel) }, singleLine = true)
                if (type != WorldItemType.BEAT) {
                    OutlinedTextField(extra, { extra = it }, label = { Text(type.extraLabel) }, minLines = 2)
                }
            }
        },
        confirmButton = {
            Button(enabled = name.isNotBlank() && description.isNotBlank(), onClick = { onAdd(name, description, details, extra) }) {
                Text("Add to canon")
            }
        },
        dismissButton = { androidx.compose.material3.TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}