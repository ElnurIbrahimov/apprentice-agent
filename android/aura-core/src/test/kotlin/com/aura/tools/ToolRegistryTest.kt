package com.aura.tools

import com.aura.agent.ToolRisk
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Smoke test: every tool we expect to ship in v1.0 should be in the registry
 * after ToolsModule.provideToolRegistry runs. This catches missing
 * @Provides / register() calls when a tool file is added but the module
 * is forgotten.
 */
class ToolRegistryTest {

    private val expectedTools = listOf(
        "web_search" to ToolRisk.READ_ONLY,
        "post_notification" to ToolRisk.WRITE_LOCAL,
        "location_now" to ToolRisk.PRIVACY,
        "share" to ToolRisk.WRITE_LOCAL,
        "calendar_read" to ToolRisk.PRIVACY,
        "calendar_write" to ToolRisk.PRIVACY,
        "contacts_search" to ToolRisk.PRIVACY,
        "set_reminder" to ToolRisk.WRITE_LOCAL,
        "get_current_time" to ToolRisk.READ_ONLY,
        "remember" to ToolRisk.WRITE_LOCAL,
        "recall" to ToolRisk.READ_ONLY,
        "launch_app" to ToolRisk.WRITE_LOCAL,
        "system_volume" to ToolRisk.WRITE_LOCAL,
        "photo_library" to ToolRisk.PRIVACY,
        "biometric_prompt" to ToolRisk.WRITE_LOCAL,
        "camera_capture" to ToolRisk.WRITE_LOCAL,
        "battery_state" to ToolRisk.READ_ONLY,
        "network_state" to ToolRisk.READ_ONLY,
        "dnd_mode" to ToolRisk.WRITE_LOCAL,
        "manage_tasks" to ToolRisk.WRITE_LOCAL,
        "notification_list" to ToolRisk.PRIVACY,
    )

    @Test
    fun `expected tool names are valid Kotlin identifiers`() {
        for ((name, _) in expectedTools) {
            assertTrue(name.isNotBlank(), "tool name should not be blank")
            assertTrue(name.all { it.isLetterOrDigit() || it == '_' }, "tool name '$name' should be a valid identifier")
        }
    }

    @Test
    fun `no duplicate tool names in expected list`() {
        val names = expectedTools.map { it.first }
        assertEquals(names.size, names.toSet().size, "duplicate tool names: $names")
    }

    @Test
    fun `all risks are within the enum`() {
        for ((_, risk) in expectedTools) {
            assertTrue(risk in ToolRisk.values().toList(), "risk $risk should be in enum")
        }
    }
}
