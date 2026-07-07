package com.aura.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountTree
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.AccountTree
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.aura.ui.screens.ChatScreen
import com.aura.ui.screens.GraphScreen
import com.aura.ui.screens.HandsScreen
import com.aura.ui.screens.HistoryScreen
import com.aura.ui.screens.HomeScreen
import com.aura.ui.screens.MemoryScreen
import com.aura.ui.screens.ProactiveHistoryScreen
import com.aura.ui.screens.ProfileScreen
import com.aura.ui.screens.SettingsScreen
import com.aura.ui.screens.TasksScreen
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.unit.dp

sealed class TopLevelRoute(val route: String, val label: String, val selectedIcon: ImageVector, val unselectedIcon: ImageVector) {
    data object Home : TopLevelRoute("home", "Home", Icons.Filled.Home, Icons.Outlined.Home)
    data object Chat : TopLevelRoute("chat", "Chat", Icons.Filled.Chat, Icons.Outlined.Chat)
    data object Memory : TopLevelRoute("memory", "Memory", Icons.Filled.Memory, Icons.Outlined.Memory)
    data object Settings : TopLevelRoute("settings", "Settings", Icons.Filled.Settings, Icons.Outlined.Settings)
    data object Graph : TopLevelRoute("graph", "Graph", Icons.Filled.AccountTree, Icons.Outlined.AccountTree)
}

private val topLevelRoutes = listOf(TopLevelRoute.Home, TopLevelRoute.Chat, TopLevelRoute.Memory, TopLevelRoute.Settings, TopLevelRoute.Graph)

@Composable
fun NavGraph(
    openChatOnLaunch: Boolean = false,
    openMemoryOnLaunch: Boolean = false,
) {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    LaunchedEffect(openChatOnLaunch) {
        if (openChatOnLaunch) {
            navController.navigate(TopLevelRoute.Chat.route) {
                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                launchSingleTop = true
                restoreState = true
            }
        }
    }
    LaunchedEffect(openMemoryOnLaunch) {
        if (openMemoryOnLaunch) {
            navController.navigate(TopLevelRoute.Memory.route) {
                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                launchSingleTop = true
                restoreState = true
            }
        }
    }

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
                HomeScreen(
                    onOpenChat = {
                        navController.navigate(TopLevelRoute.Chat.route) {
                            popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onOpenMemory = {
                        navController.navigate(TopLevelRoute.Memory.route) {
                            popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onOpenTasks = { navController.navigate("tasks") },
                    onOpenCalendar = {
                        val intent = android.content.Intent(android.content.Intent.ACTION_VIEW).apply {
                            data = android.net.Uri.parse("content://com.android.calendar/time/${System.currentTimeMillis()}")
                            flags = android.content.Intent.FLAG_ACTIVITY_NEW_TASK
                        }
                        navController.context.startActivity(intent)
                    },
                )
            }
            composable(
                route = "chat?convId={convId}",
                arguments = listOf(navArgument("convId") { type = NavType.StringType; nullable = true; defaultValue = null }),
            ) {
                val convId = it.arguments?.getString("convId")
                val summary = it.arguments?.getString("morningBriefSummary")
                ChatScreen(
                    resumeConversationId = convId,
                    morningBriefSummary = summary,
                    onNavigateHistory = { navController.navigate("history") },
                )
            }
            composable(TopLevelRoute.Memory.route) { MemoryScreen() }
            composable(TopLevelRoute.Settings.route) {
                SettingsScreen(onNavigateProfile = { navController.navigate("profile") })
            }
            composable(TopLevelRoute.Graph.route) { GraphScreen() }
            composable("history") {
                HistoryScreen(onSelect = { convId ->
                    navController.navigate("chat?convId=$convId") {
                        popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                })
            }
            composable("hands") { HandsScreen() }
            composable("tasks") { TasksScreen() }
            composable("proactive") { ProactiveHistoryScreen() }
            composable("profile") {
                ProfileScreen(onBack = { navController.popBackStack() })
            }
        }
    }
}

@Composable
private fun AuraBottomBar(navController: NavHostController, currentRoute: String?) {
    NavigationBar(
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 3.dp,
    ) {
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
                        tint = if (selected) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                    )
                },
                label = {
                    Text(
                        route.label,
                        color = if (selected) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                    )
                },
            )
        }
    }
}
