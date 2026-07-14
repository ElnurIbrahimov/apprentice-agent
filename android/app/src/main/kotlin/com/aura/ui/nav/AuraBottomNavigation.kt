package com.aura.ui.nav

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsBottomHeight
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens

sealed class TopLevelRoute(
    val route: String,
    val label: String,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector,
) {
    data object Home : TopLevelRoute("home", "Home", Icons.Filled.Home, Icons.Outlined.Home)
    data object Chat : TopLevelRoute(
        "chat",
        "Chat",
        Icons.AutoMirrored.Filled.Chat,
        Icons.AutoMirrored.Outlined.Chat,
    )
    data object Memory : TopLevelRoute("memory", "Memory", Icons.Filled.Memory, Icons.Outlined.Memory)
    data object Settings : TopLevelRoute("settings", "Settings", Icons.Filled.Settings, Icons.Outlined.Settings)
}

internal val topLevelRoutes = listOf(
    TopLevelRoute.Home,
    TopLevelRoute.Chat,
    TopLevelRoute.Memory,
    TopLevelRoute.Settings,
)

internal fun normalizedBaseRoute(route: String?): String? = route?.substringBefore('?')

@Composable
fun AuraBottomNavigation(
    currentRoute: String?,
    onRouteSelected: (TopLevelRoute) -> Unit,
    modifier: Modifier = Modifier,
    navigationBarInsets: WindowInsets = WindowInsets.navigationBars,
) {
    val colors = AuraThemeTokens.colors
    val baseRoute = normalizedBaseRoute(currentRoute)
    Surface(
        color = colors.surface0,
        shape = RectangleShape,
        shadowElevation = 0.dp,
        border = BorderStroke(1.dp, colors.borderSubtle),
        modifier = modifier
            .fillMaxWidth()
            .testTag("bottom-navigation"),
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(AuraDimensions.bottomNavigationHeight)
                    .testTag("bottom-navigation-row")
                    .padding(horizontal = 8.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                topLevelRoutes.forEach { route ->
                    val selected = baseRoute == route.route
                    val containerColor by animateColorAsState(
                        targetValue = if (selected) colors.selection else Color.Transparent,
                        animationSpec = tween(durationMillis = AuraDimensions.motionStandardMs),
                        label = "bar-item-bg",
                    )
                    val contentColor = if (selected) colors.textPrimary else colors.textTertiary
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = AuraDimensions.minimumTouchTarget)
                            .clip(RoundedCornerShape(AuraDimensions.controlRadius))
                            .background(containerColor)
                            .selectable(
                                selected = selected,
                                role = Role.Tab,
                                onClick = { if (!selected) onRouteSelected(route) },
                            )
                            .semantics { contentDescription = route.label },
                        contentAlignment = Alignment.Center,
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(2.dp),
                        ) {
                            Icon(
                                imageVector = if (selected) route.selectedIcon else route.unselectedIcon,
                                contentDescription = null,
                                tint = contentColor,
                                modifier = Modifier.size(21.dp),
                            )
                            Text(
                                text = route.label,
                                fontSize = 11.sp,
                                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                                color = contentColor,
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.windowInsetsBottomHeight(navigationBarInsets))
        }
    }
}
