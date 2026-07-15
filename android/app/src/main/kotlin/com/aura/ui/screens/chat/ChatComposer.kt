package com.aura.ui.screens.chat

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.aura.skills.Skill
import com.aura.ui.components.AuraIconButton
import com.aura.ui.theme.AuraThemeTokens
import com.aura.ui.util.Haptics

@OptIn(ExperimentalFoundationApi::class, ExperimentalMaterial3Api::class)
@Composable
fun ChatComposer(
    draft: String,
    streaming: Boolean,
    sendEnabled: Boolean,
    onDraftChange: (String) -> Unit,
    onSend: () -> Unit,
    onCancel: () -> Unit = {},
    onTapToSpeak: () -> Unit = {},
    onHoldToTalk: () -> Unit = {},
    onContinuousVoice: () -> Unit = {},
    onCameraClick: () -> Unit = {},
    onGalleryClick: () -> Unit = {},
    onAudioClick: () -> Unit = {},
    onDocumentClick: () -> Unit = {},
    skills: List<Skill> = emptyList(),
    onUseSkill: (Skill) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    var attachmentOpen by remember { mutableStateOf(false) }
    var voiceMenuOpen by remember { mutableStateOf(false) }
    val attachmentState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val hapticView = LocalView.current
    val canSend = sendEnabled && draft.isNotBlank()

    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 6.dp)
            .defaultMinSize(minHeight = 52.dp)
            .testTag("chat-composer"),
        color = AuraThemeTokens.colors.surface1,
        contentColor = AuraThemeTokens.colors.textPrimary,
        shape = RoundedCornerShape(26.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, AuraThemeTokens.colors.borderSubtle),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 2.dp),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            AuraIconButton(
                onClick = { attachmentOpen = true },
                containerColor = Color.Transparent,
                modifier = Modifier.testTag("chat-composer-attach"),
            ) {
                Icon(
                    imageVector = Icons.Filled.Add,
                    contentDescription = "Attach",
                    tint = AuraThemeTokens.colors.textSecondary,
                    modifier = Modifier.size(20.dp),
                )
            }

            BasicTextField(
                value = draft,
                onValueChange = onDraftChange,
                textStyle = MaterialTheme.typography.bodyLarge.copy(
                    color = AuraThemeTokens.colors.textPrimary,
                ),
                cursorBrush = SolidColor(AuraThemeTokens.colors.actionPrimary),
                minLines = 1,
                maxLines = 6,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(
                    onSend = { if (canSend) onSend() },
                ),
                decorationBox = { inner ->
                    Box(contentAlignment = Alignment.CenterStart) {
                        if (draft.isEmpty()) {
                            Text(
                                text = "Message Aura…",
                                color = AuraThemeTokens.colors.textSecondary,
                                style = MaterialTheme.typography.bodyLarge,
                            )
                        }
                        inner()
                    }
                },
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 48.dp, max = 144.dp)
                    .padding(vertical = 12.dp)
                    .testTag("chat-composer-input"),
            )

            if (!streaming) {
                Box {
                    Box(
                        modifier = Modifier
                            .size(48.dp)
                            .combinedClickable(
                                onClick = { voiceMenuOpen = true },
                                onLongClick = onHoldToTalk,
                            )
                            .testTag("chat-composer-voice"),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Mic,
                            contentDescription = "Voice modes",
                            tint = AuraThemeTokens.colors.textSecondary,
                            modifier = Modifier.size(20.dp),
                        )
                    }
                    DropdownMenu(
                        expanded = voiceMenuOpen,
                        onDismissRequest = { voiceMenuOpen = false },
                    ) {
                        VoiceModeItem(
                            label = "Tap to speak",
                            supporting = "Speak one message and send",
                            icon = Icons.Filled.Mic,
                            onClick = {
                                voiceMenuOpen = false
                                onTapToSpeak()
                            },
                        )
                        VoiceModeItem(
                            label = "Hold to talk",
                            supporting = "Review your transcript before sending",
                            icon = Icons.Filled.GraphicEq,
                            onClick = {
                                voiceMenuOpen = false
                                onHoldToTalk()
                            },
                        )
                        VoiceModeItem(
                            label = "Continuous voice",
                            supporting = "Hands-free listening and replies",
                            icon = Icons.Filled.AudioFile,
                            onClick = {
                                voiceMenuOpen = false
                                onContinuousVoice()
                            },
                        )
                    }
                }
            }

            AuraIconButton(
                onClick = {
                    if (streaming) {
                        onCancel()
                    } else if (canSend) {
                        Haptics.send(hapticView)
                        onSend()
                    }
                },
                enabled = streaming || canSend,
                containerColor = when {
                    streaming -> AuraThemeTokens.colors.error.copy(alpha = 0.16f)
                    canSend -> AuraThemeTokens.colors.actionPrimary
                    else -> AuraThemeTokens.colors.surface2
                },
                modifier = Modifier.testTag(if (streaming) "chat-composer-stop" else "chat-composer-send"),
            ) {
                Icon(
                    imageVector = if (streaming) Icons.Filled.Stop else Icons.AutoMirrored.Filled.Send,
                    contentDescription = if (streaming) "Stop streaming" else "Send",
                    tint = when {
                        streaming -> AuraThemeTokens.colors.error
                        canSend -> Color.White
                        else -> AuraThemeTokens.colors.textTertiary
                    },
                    modifier = Modifier.size(if (streaming) 18.dp else 20.dp),
                )
            }
        }
    }

    if (attachmentOpen) {
        ModalBottomSheet(
            onDismissRequest = { attachmentOpen = false },
            sheetState = attachmentState,
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = "Attach",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 8.dp),
                )
                AttachmentOption(Icons.Filled.PhotoLibrary, "Gallery") {
                    attachmentOpen = false
                    onGalleryClick()
                }
                AttachmentOption(Icons.Filled.AddAPhoto, "Camera") {
                    attachmentOpen = false
                    onCameraClick()
                }
                AttachmentOption(Icons.Filled.AudioFile, "Audio") {
                    attachmentOpen = false
                    onAudioClick()
                }
                AttachmentOption(Icons.Filled.Description, "Document") {
                    attachmentOpen = false
                    onDocumentClick()
                }
                if (skills.isNotEmpty()) {
                    Text(
                        text = "Use a skill",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = AuraThemeTokens.colors.textSecondary,
                        modifier = Modifier.padding(top = 16.dp, bottom = 4.dp),
                    )
                    Text(
                        text = "Inserts /use_skill <name> into the composer — the agent will load the skill's body on the next turn.",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuraThemeTokens.colors.textSecondary,
                        modifier = Modifier.padding(bottom = 8.dp),
                    )
                    skills.take(8).forEach { skill ->
                        Surface(
                            onClick = {
                                attachmentOpen = false
                                onUseSkill(skill)
                            },
                            modifier = Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp),
                            color = Color.Transparent,
                            shape = RoundedCornerShape(10.dp),
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.AutoAwesome,
                                    contentDescription = null,
                                    tint = AuraThemeTokens.colors.actionPrimary,
                                )
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = skill.name,
                                        color = AuraThemeTokens.colors.textPrimary,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    if (skill.description.isNotBlank()) {
                                        Text(
                                            text = skill.description,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = AuraThemeTokens.colors.textSecondary,
                                            maxLines = 2,
                                        )
                                    }
                                }
                            }
                        }
                    }
                    if (skills.size > 8) {
                        Text(
                            text = "+${skills.size - 8} more — open the Skills tab to see all",
                            style = MaterialTheme.typography.labelSmall,
                            color = AuraThemeTokens.colors.textTertiary,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun VoiceModeItem(
    label: String,
    supporting: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    onClick: () -> Unit,
) {
    DropdownMenuItem(
        text = {
            Column {
                Text(label, color = AuraThemeTokens.colors.textPrimary)
                Text(
                    supporting,
                    color = AuraThemeTokens.colors.textSecondary,
                    style = MaterialTheme.typography.labelSmall,
                )
            }
        },
        leadingIcon = {
            Icon(icon, contentDescription = null, tint = AuraThemeTokens.colors.actionPrimary)
        },
        onClick = onClick,
    )
}

@Composable
private fun AttachmentOption(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
) {
    Surface(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp),
        color = Color.Transparent,
        shape = RoundedCornerShape(10.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Icon(icon, contentDescription = null, tint = AuraThemeTokens.colors.textSecondary)
            Text(label, color = AuraThemeTokens.colors.textPrimary)
        }
    }
}
