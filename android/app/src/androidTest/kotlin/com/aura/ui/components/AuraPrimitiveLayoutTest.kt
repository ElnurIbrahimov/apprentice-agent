package com.aura.ui.components

import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.assertHeightIsAtLeast
import androidx.compose.ui.test.assertWidthIsAtLeast
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.unit.dp
import com.aura.ui.theme.AuraTheme
import org.junit.Rule
import org.junit.Test

class AuraPrimitiveLayoutTest {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun iconButtonExposesFortyEightDpTouchTarget() {
        composeRule.setContent {
            AuraTheme {
                AuraIconButton(
                    onClick = {},
                    modifier = Modifier.testTag("icon-button"),
                ) {
                    Icon(Icons.Filled.Close, contentDescription = null)
                }
            }
        }

        composeRule.onNodeWithTag("icon-button")
            .assertWidthIsAtLeast(48.dp)
            .assertHeightIsAtLeast(48.dp)
    }

    @Test
    fun primaryButtonExposesFortyEightDpTouchTarget() {
        composeRule.setContent {
            AuraTheme {
                AuraPrimaryButton(
                    onClick = {},
                    modifier = Modifier.testTag("primary-button"),
                ) {
                    Text("Continue")
                }
            }
        }

        composeRule.onNodeWithTag("primary-button")
            .assertHeightIsAtLeast(48.dp)
    }
}
