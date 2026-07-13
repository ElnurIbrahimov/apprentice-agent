package com.aura


import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule

import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import com.aura.ui.components.ModelPickerSheet
import com.aura.ui.theme.AuraTheme
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.assertEquals

class ModelPickerSheetTest {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun groupsModelsByProvider() {
        show(models = listOf("alpha:model-a", "beta:model-b"))
        composeRule.onNodeWithText("Alpha").assertIsDisplayed()
        composeRule.onNodeWithText("Beta").assertIsDisplayed()
        composeRule.onNodeWithTag("model-row-alpha:model-a").assertIsDisplayed()
    }

    @Test
    fun showsLoadingState() {
        show(models = emptyList(), loading = true)
        composeRule.onNodeWithText("Loading models from your providers…").assertIsDisplayed()
    }

    @Test
    fun showsVerifiedEmptyState() {
        show(models = emptyList())
        composeRule.onNodeWithText(
            "No verified models available. Save & Test a provider in Settings.",
        ).assertIsDisplayed()
    }

    @Test
    fun showsTypedErrorState() {
        show(models = emptyList(), error = "Authentication failed")
        composeRule.onNodeWithText("Couldn't load models").assertIsDisplayed()
        composeRule.onNodeWithText("Authentication failed").assertIsDisplayed()
    }

    @Test
    fun marksUnavailableCurrentModelWithoutMakingItSelectable() {
        show(current = "alpha:removed", models = listOf("alpha:model-a"))
        composeRule.onNodeWithText("Unavailable").assertIsDisplayed()
        composeRule.onNodeWithTag("model-row-alpha:removed").assertIsDisplayed()
    }

    @Test
    fun marksCachedProviderModels() {
        show(
            current = "alpha:model-a",
            models = listOf("alpha:model-a"),
            stale = setOf("alpha"),
        )
        composeRule.onNodeWithText("Cached").assertIsDisplayed()
    }

    @Test
    fun arbitraryPastedModelIsNotOfferedAndCatalogSelectionWorks() {
        var picked = ""
        show(models = listOf("alpha:model-a"), onPick = { picked = it })
        val search = composeRule.onNodeWithTag("model-search")
        search.performTextInput("alpha:made-up")
        composeRule.onNodeWithText("Use custom model ID").assertDoesNotExist()
        composeRule.onNodeWithTag("model-search").performTextClearance()
        composeRule.onNodeWithTag("model-row-alpha:model-a").performClick()
        assertEquals("alpha:model-a", picked)
    }

    private fun show(
        current: String = "",
        models: List<String>,
        loading: Boolean = false,
        error: String? = null,
        stale: Set<String> = emptySet(),
        onPick: (String) -> Unit = {},
    ) {
        composeRule.setContent {
            AuraTheme {
                ModelPickerSheet(
                    currentModel = current,
                    models = models,
                    isLoading = loading,
                    errorMessage = error,
                    staleProviderPrefixes = stale,
                    onPick = onPick,
                    onDismiss = {},
                )
            }
        }
        composeRule.waitForIdle()
    }
}
