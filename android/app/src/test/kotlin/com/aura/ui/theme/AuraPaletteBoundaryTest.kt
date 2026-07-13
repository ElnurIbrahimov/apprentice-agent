package com.aura.ui.theme

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

class AuraPaletteBoundaryTest {

    @Test
    fun `production code cannot read raw light or dark palettes outside theme`() {
        val sourceRoot = listOf(
            File("src/main/kotlin"),
            File("app/src/main/kotlin"),
        ).firstOrNull(File::isDirectory)
            ?: error("Could not locate app production sources from ${File(".").absolutePath}")

        val violations = sourceRoot.walkTopDown()
            .filter { it.isFile && it.extension == "kt" }
            .filterNot { it.invariantSeparatorsPath.contains("/ui/theme/") }
            .flatMap { file ->
                file.readLines().asSequence().mapIndexedNotNull { index, line ->
                    if (Regex("AuraTokens\\.(Dark|Light)").containsMatchIn(line)) {
                        "${file.relativeTo(sourceRoot).invariantSeparatorsPath}:${index + 1}"
                    } else {
                        null
                    }
                }
            }
            .toList()

        assertTrue(
            violations.isEmpty(),
            "Raw palette access must stay inside ui/theme; violations: ${violations.joinToString()}",
        )
    }
}
