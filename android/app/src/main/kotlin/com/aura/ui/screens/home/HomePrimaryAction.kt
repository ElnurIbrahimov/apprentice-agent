package com.aura.ui.screens.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.aura.ui.components.AuraFilterChip
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraSpacing
import com.aura.ui.theme.AuraThemeTokens

private val quickPrompts = listOf(
    "Plan my day" to "Help me plan my day",
    "What's next?" to "What should I focus on next?",
    "Remind me" to "Create a reminder",
)

@Composable
fun HomePrimaryAction(
    onAskAura: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = AuraThemeTokens.colors
    var draft by remember { mutableStateOf("") }

    fun send() {
        val prompt = draft.trim()
        if (prompt.isNotEmpty()) {
            onAskAura(prompt)
            draft = ""
        }
    }

    Surface(
        modifier = modifier.fillMaxWidth(),
        color = colors.surface0,
        shape = MaterialTheme.shapes.large,
        border = androidx.compose.foundation.BorderStroke(1.dp, colors.borderSubtle),
    ) {
        Column(
            modifier = Modifier.padding(AuraSpacing.md),
            verticalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
        ) {
            Text(
                text = "Ask Aura",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
                color = colors.textPrimary,
            )
            Text(
                text = "Start with what matters right now.",
                style = MaterialTheme.typography.bodyMedium,
                color = colors.textSecondary,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(AuraSpacing.sm),
            ) {
                OutlinedTextField(
                    value = draft,
                    onValueChange = { draft = it },
                    modifier = Modifier.weight(1f).testTag("home-ask-input"),
                    placeholder = { Text("What do you need?") },
                    singleLine = true,
                    shape = RoundedCornerShape(AuraDimensions.controlRadius),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                    keyboardActions = KeyboardActions(onSend = { send() }),
                )
                FilledIconButton(
                    onClick = { send() },
                    enabled = draft.isNotBlank(),
                    modifier = Modifier.testTag("home-ask-send"),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = colors.actionPrimary,
                        contentColor = colors.onActionPrimary,
                        disabledContainerColor = colors.actionDisabled,
                        disabledContentColor = colors.onActionDisabled,
                    ),
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Ask Aura")
                }
            }
            LazyRow(horizontalArrangement = Arrangement.spacedBy(AuraSpacing.xs)) {
                items(quickPrompts, key = { it.first }) { prompt ->
                    AuraFilterChip(
                        selected = false,
                        onClick = { onAskAura(prompt.second) },
                        label = { Text(prompt.first) },
                    )
                }
            }
        }
    }
}
