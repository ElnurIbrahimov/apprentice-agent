package com.aura.ui.components

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.aura.ui.theme.AuraDimensions

@Composable
fun ResponsiveContainer(
    modifier: Modifier = Modifier,
    includeHorizontalGutter: Boolean = true,
    content: @Composable BoxScope.() -> Unit,
) {
    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = Alignment.TopCenter,
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = AuraDimensions.contentMaxWidth)
                .then(
                    if (includeHorizontalGutter) {
                        Modifier.padding(horizontal = AuraDimensions.compactGutter)
                    } else {
                        Modifier
                    },
                ),
            content = content,
        )
    }
}
