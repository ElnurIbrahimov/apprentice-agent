package com.aura.ui.components

import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.aura.ui.theme.AuraTheme
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class AuraScreenStatesTest {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun loadingStateRendersNamedGeometrySkeletons() {
        composeRule.setContent {
            AuraTheme { AuraLoadingState(rows = 3) }
        }

        composeRule.onNodeWithTag("skeleton-row-0").assertExists()
        composeRule.onNodeWithTag("skeleton-row-1").assertExists()
        composeRule.onNodeWithTag("skeleton-row-2").assertExists()
    }

    @Test
    fun emptyStateExposesUsefulAction() {
        var acted = false
        composeRule.setContent {
            AuraTheme {
                AuraEmptyState(
                    title = "No memories",
                    message = "Save the first fact you want Aura to recall.",
                    actionLabel = "Add memory",
                    onAction = { acted = true },
                )
            }
        }

        composeRule.onNodeWithText("Add memory").performClick()
        assertTrue(acted)
    }

    @Test
    fun errorStateExposesRecoveryAction() {
        var retried = false
        composeRule.setContent {
            AuraTheme {
                AuraErrorState(
                    title = "Could not load memories",
                    message = "The local database did not respond.",
                    onRetry = { retried = true },
                )
            }
        }

        composeRule.onNodeWithText("Try again").performClick()
        assertTrue(retried)
    }
}
