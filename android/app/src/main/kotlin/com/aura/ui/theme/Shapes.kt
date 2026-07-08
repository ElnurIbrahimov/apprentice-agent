package com.aura.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

/**
 * Aura shapes — 1:1 with Aura Web's CSS radius scale.
 *   --radius-sm: 8
 *   --radius-md: 12
 *   --radius-lg: 24
 *   --radius-full: 9999
 */
val AuraShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(16.dp),
    extraLarge = RoundedCornerShape(24.dp),
)
