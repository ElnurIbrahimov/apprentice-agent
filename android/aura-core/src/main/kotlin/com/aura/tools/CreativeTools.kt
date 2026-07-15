package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.creative.CreativeProjectStore
import com.aura.creative.StoryBeat
import com.aura.creative.WorldCharacter
import com.aura.creative.WorldEvent
import com.aura.creative.WorldFaction
import com.aura.creative.WorldLocation
import com.aura.creative.WorldRule
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class CreativeReadProjectTool @Inject constructor(
    private val store: CreativeProjectStore,
) {
    fun definition() = ToolDefinition(
        name = "creative_read_project",
        description = "Read a Creative Studio project's canon: overview, characters, places, factions, rules, timeline, and outline.",
        parameters = ToolParameters(
            properties = mapOf(
                "projectId" to ToolProperty("string", "Creative project ID"),
            ),
            required = listOf("projectId"),
        ),
        category = ToolCategories.CREATIVE,
    )

    val tool = Tool(
        name = definition().name,
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        category = ToolCategories.CREATIVE,
        execute = { call, _ ->
            val id = call.arguments["projectId"] as? String
                ?: return@Tool ToolResult.Error("missing 'projectId'", "bad_args")
            val project = store.get(id)
                ?: return@Tool ToolResult.Error("Creative project not found", "not_found")
            val world = project.world
            val output = buildString {
                appendLine("PROJECT: ${project.name}")
                appendLine("Genre: ${project.genre.ifBlank { "unspecified" }}")
                appendLine("Tone: ${project.tone.ifBlank { "unspecified" }}")
                appendLine("Overview: ${world.overview.ifBlank { "not written yet" }}")
                if (world.characters.isNotEmpty()) {
                    appendLine("CHARACTERS:")
                    world.characters.forEach { appendLine("- ${it.name} [${it.role}]: ${it.backstory} Motivation: ${it.motivation}") }
                }
                if (world.locations.isNotEmpty()) {
                    appendLine("LOCATIONS:")
                    world.locations.forEach { appendLine("- ${it.name} [${it.type}]: ${it.description}") }
                }
                if (world.factions.isNotEmpty()) {
                    appendLine("FACTIONS:")
                    world.factions.forEach { appendLine("- ${it.name}: ${it.ideology}") }
                }
                if (world.rules.isNotEmpty()) {
                    appendLine("RULES:")
                    world.rules.forEach { appendLine("- ${it.name} [${it.category}]: ${it.description} Impact: ${it.impact}") }
                }
                if (world.timeline.isNotEmpty()) {
                    appendLine("TIMELINE:")
                    world.timeline.forEach { appendLine("- ${it.date} ${it.title}: ${it.description}") }
                }
                if (world.outline.isNotEmpty()) {
                    appendLine("OUTLINE:")
                    world.outline.forEachIndexed { index, beat -> appendLine("${index + 1}. ${beat.title}: ${beat.summary}") }
                }
            }
            ToolResult.Ok(output.trim())
        },
    )
}

@Singleton
class CreativeAddWorldItemTool @Inject constructor(
    private val store: CreativeProjectStore,
) {
    fun definition() = ToolDefinition(
        name = "creative_add_world_item",
        description = "Add one explicitly requested item to project canon: character, location, faction, rule, event, or story beat.",
        parameters = ToolParameters(
            properties = mapOf(
                "projectId" to ToolProperty("string", "Creative project ID"),
                "type" to ToolProperty("string", "character, location, faction, rule, event, or beat"),
                "name" to ToolProperty("string", "Name or title"),
                "description" to ToolProperty("string", "Core description, backstory, ideology, rule, event, or beat summary"),
                "details" to ToolProperty("string", "Optional role, location type, rule category, event date, or beat status"),
            ),
            required = listOf("projectId", "type", "name", "description"),
        ),
        category = ToolCategories.CREATIVE,
    )

    val tool = Tool(
        name = definition().name,
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        category = ToolCategories.CREATIVE,
        execute = { call, _ ->
            val id = call.arguments["projectId"] as? String
                ?: return@Tool ToolResult.Error("missing 'projectId'", "bad_args")
            val type = (call.arguments["type"] as? String)?.trim()?.lowercase()
                ?: return@Tool ToolResult.Error("missing 'type'", "bad_args")
            val name = (call.arguments["name"] as? String)?.trim()?.take(120)
                ?.takeIf(String::isNotBlank)
                ?: return@Tool ToolResult.Error("missing 'name'", "bad_args")
            val description = (call.arguments["description"] as? String)?.trim()?.take(4_000)
                ?.takeIf(String::isNotBlank)
                ?: return@Tool ToolResult.Error("missing 'description'", "bad_args")
            val details = (call.arguments["details"] as? String)?.trim()?.take(200).orEmpty()
            val project = store.get(id)
                ?: return@Tool ToolResult.Error("Creative project not found", "not_found")
            val world = project.world
            val updated = when (type) {
                "character" -> world.copy(
                    characters = world.characters + WorldCharacter(name = name, role = details, backstory = description),
                )
                "location" -> world.copy(
                    locations = world.locations + WorldLocation(name = name, type = details, description = description),
                )
                "faction" -> world.copy(
                    factions = world.factions + WorldFaction(name = name, ideology = description),
                )
                "rule" -> world.copy(
                    rules = world.rules + WorldRule(name = name, description = description, category = details.ifBlank { "world" }),
                )
                "event" -> world.copy(
                    timeline = world.timeline + WorldEvent(title = name, date = details, description = description),
                )
                "beat", "story_beat" -> world.copy(
                    outline = world.outline + StoryBeat(title = name, summary = description, status = details.ifBlank { "planned" }),
                )
                else -> return@Tool ToolResult.Error(
                    "unsupported type '$type'; use character, location, faction, rule, event, or beat",
                    "bad_args",
                )
            }
            store.updateWorld(id, updated)
                ?: return@Tool ToolResult.Error("Creative project disappeared during update", "not_found")
            ToolResult.Ok("Added $type '$name' to ${project.name} canon.")
        },
    )
}