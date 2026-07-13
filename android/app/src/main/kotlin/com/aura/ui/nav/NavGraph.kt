package com.aura.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
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
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
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
import com.aura.ui.screens.DiagnosticsScreen
import com.aura.ui.screens.HandsScreen
import com.aura.ui.screens.HistoryScreen
import com.aura.ui.screens.HomeScreen
import com.aura.ui.screens.KnowledgeGraphScreen
import com.aura.ui.screens.ToolsScreen
import com.aura.ui.screens.MemoryScreen
import com.aura.ui.screens.ProactiveHistoryScreen
import com.aura.ui.screens.RemindersScreen
import com.aura.ui.screens.ProfileScreen
import com.aura.ui.screens.SettingsScreen
import com.aura.ui.screens.TasksScreen
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.unit.dp

sealed class TopLevelRoute(val route: String, val label: String, val selectedIcon: ImageVector, val unselectedIcon: ImageVector) {
    data object Home : TopLevelRoute("home", "Home", Icons.Filled.Home, Icons.Outlined.Home)
    data object Chat : TopLevelRoute("chat", "Chat", Icons.Filled.Chat, Icons.Outlined.Chat)
    data object Memory : TopLevelRoute("memory", "Memory", Icons.Filled.Memory, Icons.Outlined.Memory)
    data object Settings : TopLevelRoute("settings", "Settings", Icons.Filled.Settings, Icons.Outlined.Settings)
}

private val topLevelRoutes = listOf(TopLevelRoute.Home, TopLevelRoute.Chat, TopLevelRoute.Memory, TopLevelRoute.Settings)

@Composable
fun NavGraph(
    openChatOnLaunch: Boolean = false,
    openMemoryOnLaunch: Boolean = false,
) {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    // Show the bottom bar on all screens — not just top-level routes.
    // The previous check (`topLevelRoutes.any { it.route == baseRoute }`)
    // hid the bar on every secondary screen (history, hands, tasks, tools,
    // proactive, reminders, profile, identity_editor), forcing the user
    // to press the system back button to return to a tab. Keeping the bar
    // always visible matches the Aura Web pattern where the left sidebar
    // persists across all views.
    val showBottomBar = true

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
        contentWindowInsets = WindowInsets.safeDrawing.only(
            WindowInsetsSides.Top + WindowInsetsSides.Horizontal,
        ),
        bottomBar = {
            AnimatedVisibility(
                visible = showBottomBar,
                enter = slideInVertically { it } + fadeIn(),
                exit = slideOutVertically { it } + fadeOut(),
            ) {
                AuraBottomBar(navController, backStackEntry?.destination?.route)
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
                    onOpenChat = { prefill ->
                        // Pre-fill the chat draft via a query param.
                        // Empty string means "just open chat" — no prefill.
                        val route = if (prefill.isNotBlank()) {
                            "chat?draft=${android.net.Uri.encode(prefill)}"
                        } else {
                            TopLevelRoute.Chat.route
                        }
                        navController.navigate(route) {
                            popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    },
                    onOpenChatWithBrief = { brief ->
                        // Morning-brief proactive card. Pass the brief text
                        // as `brief` query param so ChatScreen auto-sends
                        // it as a user message.
                        navController.navigate(
                            "chat?brief=${android.net.Uri.encode(brief)}"
                        ) {
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
                    onOpenReminders = { navController.navigate("reminders") },
                    onOpenHands = { navController.navigate("hands") },
                    onOpenTools = { navController.navigate("tools") },
                    onOpenProactive = { navController.navigate("proactive") },
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
                route = "chat?convId={convId}&draft={draft}&brief={brief}",
                arguments = listOf(
                    navArgument("convId") { type = NavType.StringType; nullable = true; defaultValue = null },
                    navArgument("draft") { type = NavType.StringType; nullable = true; defaultValue = null },
                    navArgument("brief") { type = NavType.StringType; nullable = true; defaultValue = null },
                ),
            ) {
                val convId = it.arguments?.getString("convId")
                val summary = it.arguments?.getString("brief")
                val draft = it.arguments?.getString("draft")
                ChatScreen(
                    resumeConversationId = convId,
                    morningBriefSummary = summary,
                    initialDraft = draft,
                    onNavigateHistory = { navController.navigate("history") },
                )
            }
            composable(TopLevelRoute.Memory.route) {
                MemoryScreen(onOpenKnowledgeGraph = { navController.navigate("knowledge_graph") })
            }
            composable(TopLevelRoute.Settings.route) {
                SettingsScreen(
                    onNavigateProfile = { navController.navigate("profile") },
                    onNavigateDiagnostics = { navController.navigate("diagnostics") },
                )
            }
            composable("diagnostics") {
                DiagnosticsScreen(onBack = { navController.popBackStack() })
            }
            composable("identity_editor") {
                com.aura.ui.screens.IdentityEditorScreen(
                    onBack = { navController.popBackStack() },
                )
            }
            composable("knowledge_graph") {
                KnowledgeGraphScreen(onBack = { navController.popBackStack() })
            }
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
            composable("tools") { ToolsScreen() }
            composable("proactive") { ProactiveHistoryScreen() }
            composable("reminders") {
                RemindersScreen(onBack = { navController.popBackStack() })
            }
            composable("profile") {
                ProfileScreen(onBack = { navController.popBackStack() })
            }
        }
    }
}

@Composable
private fun AuraBottomBar(navController: NavHostController, currentRoute: String?) {
    val colors = AuraThemeTokens.colors
    Surface(
        color = colors.surface0,
        shape = RectangleShape,
        shadowElevation = 0.dp,
        border = BorderStroke(1.dp, colors.borderSubtle),
        modifier = Modifier
            .navigationBarsPadding()
            .fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(AuraDimensions.bottomNavigationHeight)
                .padding(horizontal = 8.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            topLevelRoutes.forEach { route ->
                val baseRoute = currentRoute?.substringBefore('?')
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
                        .clickable {
                            if (!selected) {
                                navController.navigate(route.route) {
                                    popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Icon(
                            imageVector = if (selected) route.selectedIcon else route.unselectedIcon,
                            contentDescription = route.label,
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
    }
}
