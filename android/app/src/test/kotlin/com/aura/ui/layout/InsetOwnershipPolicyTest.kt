package com.aura.ui.layout

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

class InsetOwnershipPolicyTest {

    private fun source(relativePath: String): String {
        val candidates = listOf(
            File("src/main/kotlin/$relativePath"),
            File("app/src/main/kotlin/$relativePath"),
        )
        return candidates.firstOrNull(File::isFile)?.readText()
            ?: error("Missing source file $relativePath")
    }

    @Test
    fun `nested scaffolds explicitly opt out of system insets`() {
        val screens = listOf(
            "com/aura/ui/screens/HandsScreen.kt",
            "com/aura/ui/screens/TasksScreen.kt",
            "com/aura/ui/screens/ProactiveHistoryScreen.kt",
            "com/aura/ui/screens/ProfileScreen.kt",
            "com/aura/ui/screens/RemindersScreen.kt",
            "com/aura/ui/screens/IdentityEditorScreen.kt",
        )

        val violations = screens.filterNot { path ->
            source(path).contains("contentWindowInsets = WindowInsets(0)")
        }

        assertTrue(
            violations.isEmpty(),
            "Nested Scaffolds must opt out of root-owned system insets: ${violations.joinToString()}",
        )
    }

    @Test
    fun `root navigation consumes scaffold padding at composition boundary`() {
        val nav = source("com/aura/ui/nav/NavGraph.kt")
        assertTrue(
            nav.contains("consumeWindowInsets(padding)"),
            "NavHost must consume root Scaffold padding before child Scaffolds",
        )
    }
}
