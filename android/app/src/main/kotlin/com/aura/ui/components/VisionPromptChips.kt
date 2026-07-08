package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.clickable

/**
 * Quick-prompt chips for a captured image. Shown above the input
 * bar right after the user captures or picks a photo. The chips
 * turn the image into a question — tapping one fires
 * [onPick] with the chosen prompt.
 *
 * Why this and not the default "Describe this image in detail":
 * - Most users want a specific outcome (text recognition,
 *   translation, identification), not a generic description.
 * - 3 well-chosen chips surface 80% of common vision intents.
 *
 * The dismiss X clears the staged image without sending.
 */
@Composable
fun VisionPromptChips(
    onPick: (String) -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val prompts = listOf(
        "Describe" to "Describe this image in detail",
        "Read text" to "Read all the text in this image",
        "Translate" to "Translate the text in this image to English",
    )
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for ((label, prompt) in prompts) {
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f),
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.clickable { onPick(prompt) },
            ) {
                Text(
                    text = label,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                )
            }
        }
        IconButton(
            onClick = onDismiss,
            modifier = Modifier.size(32.dp),
        ) {
            Icon(
                imageVector = Icons.Filled.Close,
                contentDescription = "Dismiss staged image",
                tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                modifier = Modifier.size(18.dp),
            )
        }
    }
}
