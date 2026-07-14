package com.aura.ui.screens.home

import androidx.compose.ui.test.assertIsDisplayed

import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToIndex
import androidx.compose.ui.test.performTextInput
import com.aura.ui.theme.AuraTheme
import com.aura.ui.viewmodel.HomeLoadState
import com.aura.ui.viewmodel.HomeUiState
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class HomeContentTest {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun loading_never_flashes_empty_content() {
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                HomeContent(
                    state = HomeUiState(loadState = HomeLoadState.Loading),
                    greeting = "Good morning",
                    dateLabel = "Tuesday, July 14",
                )
            }
        }

        composeRule.onNodeWithTag("home-loading").assertIsDisplayed()
        composeRule.onNodeWithTag("home-empty").assertDoesNotExist()
    }

    @Test
    fun empty_home_keeps_primary_composer_and_turns_zero_counts_into_actions() {
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                HomeContent(
                    state = HomeUiState(loadState = HomeLoadState.Empty),
                    greeting = "Good morning",
                    dateLabel = "Tuesday, July 14",
                )
            }
        }

        composeRule.onNodeWithTag("home-empty").assertIsDisplayed()
        composeRule.onNodeWithText("Ask Aura").assertIsDisplayed()
        composeRule.onNodeWithText("Add memory").assertIsDisplayed()
        composeRule.onNodeWithText("Create task").assertIsDisplayed()
        assertEquals(0, composeRule.onAllNodesWithText("0").fetchSemanticsNodes().size)
    }

    @Test
    fun populated_home_renders_one_current_priority_and_count_metadata() {
        composeRule.setContent {
            AuraTheme(themeMode = "dark") {
                HomeContent(
                    state = HomeUiState(
                        loadState = HomeLoadState.Content,
                        pendingTasks = listOf("Ship the build", "Review release notes"),
                        toolsCount = 38,
                    ),
                    greeting = "Good evening",
                    dateLabel = "Tuesday, July 14",
                )
            }
        }

        composeRule.onNodeWithText("Current priority").assertIsDisplayed()
        composeRule.onNodeWithText("Ship the build").assertIsDisplayed()
        composeRule.onNodeWithText("Tasks").assertIsDisplayed()
        composeRule.onNodeWithText("2 open").assertIsDisplayed()
        composeRule.onNodeWithTag("home-destinations").performScrollToIndex(4)
        composeRule.onNodeWithText("38 available").assertIsDisplayed()
    }

    @Test
    fun partial_error_keeps_priority_visible_with_retryable_status() {
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                HomeContent(
                    state = HomeUiState(
                        loadState = HomeLoadState.Error(
                            message = "Calendar is unavailable",
                            hasPartialContent = true,
                        ),
                        pendingTasks = listOf("Keep working"),
                    ),
                    greeting = "Good morning",
                    dateLabel = "Tuesday, July 14",
                )
            }
        }

        composeRule.onNodeWithText("Calendar is unavailable").assertIsDisplayed()
        composeRule.onNodeWithText("Keep working").assertIsDisplayed()
        composeRule.onNodeWithText("Retry").assertIsDisplayed()
    }

    @Test
    fun ask_aura_composer_sends_typed_draft() {
        var sent = ""
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                HomeContent(
                    state = HomeUiState(loadState = HomeLoadState.Empty),
                    greeting = "Good morning",
                    dateLabel = "Tuesday, July 14",
                    onAskAura = { sent = it },
                )
            }
        }

        composeRule.onNodeWithTag("home-ask-input").performTextInput("Plan my day")
        composeRule.onNodeWithTag("home-ask-send").performClick()

        composeRule.runOnIdle { assertEquals("Plan my day", sent) }
    }
}
