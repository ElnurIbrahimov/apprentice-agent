package com.aura.ui.viewmodel

import com.aura.kg.EdgeType
import com.aura.kg.KgEdge
import com.aura.kg.KgNode
import com.aura.kg.KnowledgeGraphRepository
import com.aura.kg.NodeType
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class KnowledgeGraphViewModelTest {

    private val scheduler = TestCoroutineScheduler()
    private val dispatcher = StandardTestDispatcher(scheduler)
    private lateinit var repository: KnowledgeGraphRepository

    private val kotlinNode = KgNode(id = "kotlin", label = "Kotlin", type = NodeType.SKILL)
    private val auraNode = KgNode(id = "aura", label = "Aura Android", type = NodeType.PROJECT)
    private val elnurNode = KgNode(id = "elnur", label = "Elnur", type = NodeType.PERSON)

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
        repository = mockk(relaxed = true)
        coEvery { repository.recent(500) } returns listOf(kotlinNode, auraNode, elnurNode)
        coEvery { repository.stats() } returns KnowledgeGraphRepository.Stats(3, 2)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial load exposes nodes and graph stats`() = runTest(scheduler) {
        val vm = KnowledgeGraphViewModel(repository)
        runCurrent()

        assertFalse(vm.state.value.loading)
        assertEquals(listOf(kotlinNode, auraNode, elnurNode), vm.state.value.nodes)
        assertEquals(3, vm.state.value.stats.nodeCount)
        assertEquals(2, vm.state.value.stats.edgeCount)
    }

    @Test
    fun `query is debounced and combines with type filter`() = runTest(scheduler) {
        val vm = KnowledgeGraphViewModel(repository)
        runCurrent()

        vm.setTypeFilter(NodeType.PROJECT)
        vm.setQuery("elnur")
        runCurrent()
        assertEquals(listOf(auraNode), vm.state.value.nodes)

        advanceTimeBy(251)
        runCurrent()
        assertTrue(vm.state.value.nodes.isEmpty())
        assertEquals(NodeType.PROJECT, vm.state.value.typeFilter)
    }

    @Test
    fun `selectNode loads relations and resolves neighboring labels`() = runTest(scheduler) {
        coEvery { repository.getNeighbors("aura") } returns KnowledgeGraphRepository.Neighbors(
            incoming = listOf(KgEdge("in", EdgeType.CREATED_BY, "elnur", "aura")),
            outgoing = listOf(KgEdge("out", EdgeType.USES, "aura", "kotlin")),
        )
        val vm = KnowledgeGraphViewModel(repository)
        runCurrent()

        vm.selectNode(auraNode)
        runCurrent()

        val selected = assertNotNull(vm.state.value.selected)
        assertEquals(auraNode, selected.node)
        assertEquals("Elnur", selected.incoming.single().otherLabel)
        assertEquals("Kotlin", selected.outgoing.single().otherLabel)
    }

    @Test
    fun `update merge and delete refresh the list`() = runTest(scheduler) {
        val properties = buildJsonObject { put("platform", JsonPrimitive("Android")) }
        coEvery {
            repository.updateNode("aura", "Aura", NodeType.PROJECT, properties, any())
        } returns auraNode.copy(label = "Aura", properties = properties)
        coEvery { repository.mergeNodes("aura", "kotlin", any()) } returns kotlinNode
        val vm = KnowledgeGraphViewModel(repository)
        runCurrent()

        vm.updateNode("aura", "Aura", NodeType.PROJECT, properties)
        runCurrent()
        vm.mergeNode("aura", "kotlin")
        runCurrent()
        vm.deleteNode("elnur")
        runCurrent()

        coVerify(exactly = 1) { repository.updateNode("aura", "Aura", NodeType.PROJECT, properties, any()) }
        coVerify(exactly = 1) { repository.mergeNodes("aura", "kotlin", any()) }
        coVerify(exactly = 1) { repository.deleteNode("elnur") }
        coVerify(atLeast = 4) { repository.recent(500) }
    }

    @Test
    fun `mutation failure is surfaced and can be dismissed`() = runTest(scheduler) {
        coEvery { repository.deleteNode("aura") } throws IllegalStateException("database busy")
        val vm = KnowledgeGraphViewModel(repository)
        runCurrent()

        vm.deleteNode("aura")
        runCurrent()

        assertTrue(vm.state.value.error?.contains("database busy") == true)
        vm.clearError()
        assertEquals(null, vm.state.value.error)
    }
}
