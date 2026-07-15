package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import com.aura.skills.Skill
import com.aura.skills.SkillsStore
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Invokes a user-authored skill by name and returns the skill body for
 * inclusion in the next LLM context. The agent sees the body and follows
 * whatever instructions the user wrote.
 *
 * Risk: READ_ONLY (no local mutations, no network egress).
 *
 * Note: the body is the *raw markdown* the user wrote. Tools should not
 * try to render or summarize it; the agent loop will surface it as
 * additional context for the next assistant turn.
 */
@Singleton
class UseSkillTool @Inject constructor(
    private val skillsStore: SkillsStore,
) {
    fun definition() = ToolParameters(
        properties = mapOf(
            "name" to ToolProperty(
                type = "string",
                description = "The skill name (case-insensitive) to invoke.",
            ),
        ),
        required = listOf("name"),
    )

    val tool = Tool(
        name = "use_skill",
        description = "Invoke a user-authored skill by name. Returns the skill body " +
            "(free-form markdown instructions) which the agent should follow for the next turn.",
        risk = ToolRisk.READ_ONLY,
        parameters = definition(),
        execute = { call, _ ->
            val name = call.arguments["name"] as? String
                ?: return@Tool ToolResult.Error("missing 'name' argument", "bad_args")
            skillsStore.awaitLoaded()
            val skill: Skill = skillsStore.findByName(name.trim())
                ?: return@Tool ToolResult.Error(
                    "No skill named '$name'. Available: " +
                        skillsStore.skills.value.joinToString { it.name }.ifBlank { "(none yet)" },
                    "skill_not_found",
                )
            ToolResult.Ok(
                buildString {
                    appendLine("# Skill: ${skill.name}")
                    if (skill.description.isNotBlank()) appendLine(skill.description)
                    appendLine()
                    append(skill.body)
                }
            )
        },
        category = "skills",
    )
}
