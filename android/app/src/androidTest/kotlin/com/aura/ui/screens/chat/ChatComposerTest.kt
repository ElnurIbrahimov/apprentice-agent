package com.aura.ui.screens.chat

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsNotDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraTheme
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class ChatComposerTest {
    @get:Rule val composeRule = createComposeRule()

    @Test
    fun sends_typed_draft_and_is_at_least_52dp() {
        var sent = false
        composeRule.setContent {
            AuraTheme {
                var draft by remember { mutableStateOf("") }
                Box(Modifier.width(360.dp)) {
                    ChatComposer(
                        draft = draft,
                        streaming = false,
                        sendEnabled = true,
                        onDraftChange = { draft = it },
                        onSend = { sent = true },
                    )
                }
            }
        }

        val height = composeRule.onNodeWithTag("chat-composer").fetchSemanticsNode().boundsInRoot.height
        val minHeight = with(composeRule.density) { 52.dp.toPx() }
        assertTrue("composer height $height is below $minHeight", height >= minHeight - 1f)
        composeRule.onNodeWithTag("chat-composer-input").performTextInput("Hello Aura")
        composeRule.onNodeWithContentDescription("Send").performClick()
        assertTrue(sent)
    }

    @Test
    fun blank_or_unverified_model_disables_send() {
        composeRule.setContent {
            AuraTheme {
                ChatComposer(
                    draft = "",
                    streaming = false,
                    sendEnabled = false,
                    onDraftChange = {},
                    onSend = {},
                )
            }
        }

        composeRule.onNodeWithContentDescription("Send").assertIsNotEnabled()
    }

    @Test
    fun voice_button_exposes_all_three_modes() {
        var continuous = false
        composeRule.setContent {
            AuraTheme {
                ChatComposer(
                    draft = "",
                    streaming = false,
                    sendEnabled = true,
                    onDraftChange = {},
                    onSend = {},
                    onContinuousVoice = { continuous = true },
                )
            }
        }

        composeRule.onNodeWithContentDescription("Voice modes").performClick()
        composeRule.onNodeWithText("Tap to speak").assertIsDisplayed()
        composeRule.onNodeWithText("Hold to talk").assertIsDisplayed()
        composeRule.onNodeWithText("Continuous voice").performClick()
        assertTrue(continuous)
    }

    @Test
    fun attachment_button_exposes_camera_gallery_and_audio() {
        composeRule.setContent {
            AuraTheme {
                ChatComposer(
                    draft = "Draft",
                    streaming = false,
                    sendEnabled = true,
                    onDraftChange = {},
                    onSend = {},
                )
            }
        }

        composeRule.onNodeWithContentDescription("Attach").performClick()
        composeRule.onNodeWithText("Camera").assertIsDisplayed()
        composeRule.onNodeWithText("Gallery").assertIsDisplayed()
        composeRule.onNodeWithText("Audio").assertIsDisplayed()
    }

    @Test
    fun streaming_replaces_send_with_stop() {
        composeRule.setContent {
            AuraTheme {
                ChatComposer(
                    draft = "Question",
                    streaming = true,
                    sendEnabled = true,
                    onDraftChange = {},
                    onSend = {},
                )
            }
        }

        composeRule.onNodeWithContentDescription("Stop streaming").assertIsDisplayed()
    }
}
