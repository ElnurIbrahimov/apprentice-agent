package com.aura.ui.viewmodel

import com.aura.creative.CreativeEngine
import com.aura.creative.CreativeMode
import com.aura.creative.CreativeProject
import com.aura.creative.CreativeProjectStore
import com.aura.creative.WorldBible
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

@OptIn(ExperimentalCoroutinesApi::class)
class CreativeStudioViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    private val store = mockk<CreativeProjectStore>(relaxed = true)
    private val engine = mockk<CreativeEngine>()
    private val project = CreativeProject(
        "p1", "Glass City", "", "fantasy", "haunting", WorldBible(overview = "Glass remembers"),
        "novel", 0, 1L, 1L,
    )

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
        every { store.observeAll() } returns flowOf(listOf(project))
    }

    @After
    fun tearDown() = Dispatchers.resetMain()

    @Test
    fun `load selects project and exposes its world`() = runTest {
        coEvery { store.get("p1") } returns project
        val vm = CreativeStudioViewModel(store, engine)
        vm.loadProject("p1")
        advanceUntilIdle()
        assertEquals("Glass City", vm.state.value.selectedProject?.name)
        assertEquals("Glass remembers", vm.state.value.selectedProject?.world?.overview)
    }

    @Test
    fun `create delegates project metadata and selects result`() = runTest {
        coEvery { store.create(any(), any(), any(), any(), any()) } returns project
        val vm = CreativeStudioViewModel(store, engine)
        vm.createProject("Glass City", "Memory city", "fantasy", "haunting", "novel")
        advanceUntilIdle()
        coVerify { store.create("Glass City", "Memory city", "fantasy", "haunting", "novel") }
        assertEquals("p1", vm.state.value.createdProjectId)
    }

    @Test
    fun `generate streams output and clears busy state`() = runTest {
        coEvery { store.get("p1") } returns project
        every { engine.generate("p1", CreativeMode.DRAFT, "Opening", "") } returns flowOf("Glass ", "sang.")
        val vm = CreativeStudioViewModel(store, engine)
        vm.loadProject("p1")
        advanceUntilIdle()
        vm.generate(CreativeMode.DRAFT, "Opening")
        advanceUntilIdle()
        assertEquals("Glass sang.", vm.state.value.output)
        assertFalse(vm.state.value.generating)
    }
}