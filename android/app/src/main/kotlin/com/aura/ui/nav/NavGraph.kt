package com.aura.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Chat
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.aura.ui.screens.ChatScreen
import com.aura.ui.screens.HomeScreen
import com.aura.ui.screens.MemoryScreen
import com.aura.ui.screens.SettingsScreen

sealed class TopLevelRoute(val route: String, val label: String, val selectedIcon: ImageVector, val unselectedIcon: ImageVector) {
    data object Home : TopLevelRoute("home", "Home", Icons.Filled.Home, Icons.Outlined.Home)
    data object Chat : TopLevelRoute("chat", "Chat", Icons.Filled.Chat, Icons.Outlined.Chat)
    data object Memory : TopLevelRoute("memory", "Memory", Icons.Filled.Memory, Icons.Outlined.Memory)
    data object Settings : TopLevelRoute("settings", "Settings", Icons.Filled.Settings, Icons.Outlined.Settings)
}

private val topLevelRoutes = listOf(TopLevelRoute.Home, TopLevelRoute.Chat, TopLevelRoute.Memory, TopLevelRoute.Settings)

@Composable
fun NavGraph() {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    Scaffold(
        bottomBar = {
            AnimatedVisibility(
                visible = topLevelRoutes.any { it.route == currentRoute },
                enter = slideInVertically { it } + fadeIn(),
                exit = slideOutVertically { it } + fadeOut(),
            ) {
                AuraBottomBar(navController, currentRoute)
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = TopLevelRoute.Home.route,
            modifier = Modifier.padding(padding),
        ) {
            composable(TopLevelRoute.Home.route) {
                HomeScreen(onOpenChat = {
                    navController.navigate(TopLevelRoute.Chat.route) {
                        popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                })
            }
            composable(TopLevelRoute.Chat.route) { ChatScreen() }
            composable(TopLevelRoute.Memory.route) { MemoryScreen() }
            composable(TopLevelRoute.Settings.route) { SettingsScreen() }
        }
    }
}

@Composable
private fun AuraBottomBar(navController: NavHostController, currentRoute: String?) {
    NavigationBar {
        topLevelRoutes.forEach { route ->
            val selected = currentRoute == route.route
            NavigationBarItem(
                selected = selected,
                onClick = {
                    if (!selected) {
                        navController.navigate(route.route) {
                            popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                },
                icon = {
                    Icon(
                        imageVector = if (selected) route.selectedIcon else route.unselectedIcon,
                        contentDescription = route.label,
                    )
                },
                label = { Text(route.label) },
            )
        }
    }
}
