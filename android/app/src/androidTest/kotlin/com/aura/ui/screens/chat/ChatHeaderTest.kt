package com.aura.ui.screens.chat

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.assertHeightIsEqualTo
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraTheme
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.assertTrue

class ChatHeaderTest {
    @get:Rule val composeRule = createComposeRule()

    @Test
    fun compact_width_keeps_model_new_chat_and_overflow_visible() {
        composeRule.setContent {
            AuraTheme {
                Box(Modifier.width(320.dp)) {
                    ChatHeader(
                        activeModel = "provider:an-extremely-long-model-name-that-must-ellipsis",
                        conversationModel = null,
                    )
                }
            }
        }

        composeRule.onNodeWithTag("chat-header").assertHeightIsEqualTo(56.dp)
        val pillWidth = composeRule.onNodeWithTag("chat-model-pill").fetchSemanticsNode().boundsInRoot.width
        val maxPillWidth = with(composeRule.density) { 176.dp.toPx() }
        assertTrue("model pill width $pillWidth exceeds $maxPillWidth", pillWidth <= maxPillWidth + 1f)
        composeRule.onNodeWithContentDescription("New conversation").assertIsDisplayed()
        composeRule.onNodeWithContentDescription("More chat actions").assertIsDisplayed()
    }

    @Test
    fun missing_model_is_an_explicit_picker_state() {
        composeRule.setContent { AuraTheme { ChatHeader(activeModel = "", conversationModel = null) } }

        composeRule.onNodeWithText("Choose model").assertIsDisplayed()
        composeRule.onNodeWithTag("chat-model-missing", useUnmergedTree = true).assertIsDisplayed()
    }

    @Test
    fun session_override_is_visibly_scoped() {
        composeRule.setContent {
            AuraTheme {
                ChatHeader(
                    activeModel = "ollama:default-model",
                    conversationModel = "anthropic:session-model",
                )
            }
        }

        composeRule.onNodeWithTag("chat-session-override", useUnmergedTree = true).assertIsDisplayed()
    }

    @Test
    fun streaming_keeps_essential_actions_visible() {
        composeRule.setContent {
            AuraTheme {
                Box(Modifier.width(320.dp)) {
                    ChatHeader(activeModel = "ollama:model", streaming = true)
                }
            }
        }

        composeRule.onNodeWithContentDescription("New conversation").assertIsDisplayed()
        composeRule.onNodeWithContentDescription("More chat actions").assertIsDisplayed()
    }
}
