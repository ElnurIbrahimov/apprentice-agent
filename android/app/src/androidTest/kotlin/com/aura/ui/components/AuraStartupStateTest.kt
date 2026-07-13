package com.aura.ui.components

import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.assertHeightIsAtLeast
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraTheme
import org.junit.Rule
import org.junit.Test

class AuraStartupStateTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun brandedStartupUsesBalancedStaticComposition() {
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                AuraStartupState(modifier = Modifier.testTag("startup"))
            }
        }

        composeRule.onNodeWithTag("aura-startup-mark").assertIsDisplayed()
        composeRule.onNodeWithTag("aura-startup-content").assertHeightIsAtLeast(96.dp)
    }

    @Test
    fun appLockContentIsCenteredAndActionable() {
        composeRule.setContent {
            AuraTheme(themeMode = "dark") {
                AuraAppLockContent(
                    statusMessage = null,
                    onUnlock = {},
                    modifier = Modifier.testTag("app-lock"),
                )
            }
        }

        composeRule.onNodeWithTag("app-lock-content").assertIsDisplayed()
        composeRule.onNodeWithTag("app-lock-action").assertHeightIsAtLeast(48.dp)
    }
}
