package com.aura.ui.viewmodel

import com.aura.ui.screens.home.HomePriority
import com.aura.ui.screens.home.selectHomePriority
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

class HomeStateMappingTest {

    @Test
    fun `initial state remains loading until all first reads resolve`() {
        val state = HomeUiState()

        assertIs<HomeLoadState.Loading>(state.loadState)
        assertFalse(state.isEmptyResolved)
    }

    @Test
    fun `resolved state distinguishes empty content and data source errors`() {
        assertIs<HomeLoadState.Empty>(resolveHomeLoadState(hasData = false, dataSourceError = null))
        assertIs<HomeLoadState.Content>(resolveHomeLoadState(hasData = true, dataSourceError = null))

        val fullError = assertIs<HomeLoadState.Error>(
            resolveHomeLoadState(hasData = false, dataSourceError = "Calendar unavailable"),
        )
        assertFalse(fullError.hasPartialContent)

        val partialError = assertIs<HomeLoadState.Error>(
            resolveHomeLoadState(hasData = true, dataSourceError = "Calendar unavailable"),
        )
        assertTrue(partialError.hasPartialContent)
    }

    @Test
    fun `priority chooses proactive then calendar task reminder and memory`() {
        assertIs<HomePriority.Proactive>(
            selectHomePriority(
                HomeUiState(
                    loadState = HomeLoadState.Content,
                    proactiveEvent = com.aura.proactive.ProactiveEventBus.Event.MorningBriefReady(
                        title = "Morning brief",
                        body = "Ready",
                    ),
                    today = listOf("09:00 · Stand-up"),
                    pendingTasks = listOf("Ship the build"),
                ),
            ),
        )
        assertEquals(
            HomePriority.Calendar("09:00 · Stand-up"),
            selectHomePriority(HomeUiState(today = listOf("09:00 · Stand-up"))),
        )
        assertEquals(
            HomePriority.Task("Ship the build"),
            selectHomePriority(HomeUiState(pendingTasks = listOf("Ship the build"))),
        )
        assertEquals(
            HomePriority.Reminder("14:30 · Call Alex"),
            selectHomePriority(HomeUiState(upcomingReminders = listOf("14:30 · Call Alex"))),
        )
        assertIs<HomePriority.None>(selectHomePriority(HomeUiState()))
    }
}
