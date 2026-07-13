package com.aura.ui

import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.assertHeightIsAtLeast
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import com.aura.ui.components.AuraScreenHeader
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraTheme
import org.junit.Rule
import org.junit.Test

class InsetLayoutTest {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun sharedScreenHeaderKeepsTokenizedBaselineHeight() {
        composeRule.setContent {
            AuraTheme {
                AuraScreenHeader(
                    title = "Tasks",
                    subtitle = "Three pending",
                    modifier = Modifier.testTag("screen-header"),
                )
            }
        }

        composeRule.onNodeWithTag("screen-header")
            .assertHeightIsAtLeast(AuraDimensions.topAppBarHeight)
    }
}
