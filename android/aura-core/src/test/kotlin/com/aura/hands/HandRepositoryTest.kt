package com.aura.hands

import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolResult
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class HandRepositoryTest {

    private lateinit var dao: HandDao
    private lateinit var repository: HandRepository
    private lateinit var executor: ToolExecutor

    private val ctx = ToolContext(conversationId = "test", timeout = 5000L)

    @Before
    fun setUp() {
        dao = mockk()
        executor = mockk()
        repository = HandRepository(dao)
    }

    @Test
    fun `getAll returns all hands from dao`() = runTest {
        val hands = listOf(
            Hand(id = "1", name = "Good Morning", steps = """[{"tool":"web_search","args":"{\"query\":\"weather\"}"}]"""),
            Hand(id = "2", name = "Night Routine", steps = """[]"""),
        )
        coEvery { dao.getAll() } returns hands

        val result = repository.getAll()

        assertEquals(hands, result)
    }

    @Test
    fun `getByName returns hand when exists`() = runTest {
        val hand = Hand(id = "1", name = "Good Morning")
        coEvery { dao.getByName("Good Morning") } returns hand

        val result = repository.getByName("Good Morning")

        assertEquals(hand, result)
    }

    @Test
    fun `getByName returns null when not exists`() = runTest {
        coEvery { dao.getByName("DoesNotExist") } returns null

        val result = repository.getByName("DoesNotExist")

        assertNull(result)
    }

    @Test
    fun `insert delegates to dao`() = runTest {
        val hand = Hand(id = "1", name = "Test Hand")
        coEvery { dao.insert(hand) } returns Unit

        repository.insert(hand)

        coVerify { dao.insert(hand) }
    }

    @Test
    fun `deleteByName delegates to dao`() = runTest {
        coEvery { dao.deleteByName("Test Hand") } returns Unit

        repository.deleteByName("Test Hand")

        coVerify { dao.deleteByName("Test Hand") }
    }

    @Test
    fun `run executes all steps sequentially and returns joined output`() = runTest {
        val hand = Hand(
            id = "1",
            name = "MultiStep",
            steps = """[
                {"tool": "web_search", "args": "{\"query\":\"news\"}"},
                {"tool": "get_current_time", "args": "{}"}
            ]""",
        )

        coEvery { dao.getByName("MultiStep") } returns hand
        coEvery { executor.execute("web_search", """{"query":"news"}""", ctx) } returns ToolResult.Ok("Found news")
        coEvery { executor.execute("get_current_time", "{}", ctx) } returns ToolResult.Ok("12:00 PM")

        val result = repository.run(hand, executor, ctx)

        assertNotNull(result)
        val ok = result as? ToolResult.Ok
        assertNotNull(ok)
        assertEquals("Hand 'MultiStep' completed.\nStep 1 (web_search): Found news\nStep 2 (get_current_time): 12:00 PM", ok?.output)
        coVerify { executor.execute("web_search", """{"query":"news"}""", ctx) }
        coVerify { executor.execute("get_current_time", "{}", ctx) }
    }

    @Test
    fun `run returns error on first failing step`() = runTest {
        val hand = Hand(
            id = "1",
            name = "FailingHand",
            steps = """[
                {"tool": "web_search", "args": "{\"query\":\"ok\"}"},
                {"tool": "battery_state", "args": "{}"}
            ]""",
        )

        coEvery { executor.execute("web_search", """{"query":"ok"}""", ctx) } returns ToolResult.Ok("Success")
        coEvery { executor.execute("battery_state", "{}", ctx) } returns ToolResult.Error("Battery info unavailable", "sensor_error")

        val result = repository.run(hand, executor, ctx)

        val err = result as? ToolResult.Error
        assertNotNull(err)
        assertEquals("Step 2 (battery_state) failed: Battery info unavailable", err?.message)
    }

    @Test
    fun `run returns error when hand is disabled`() = runTest {
        val hand = Hand(id = "1", name = "DisabledHand", enabled = false)

        val result = repository.run(hand, executor, ctx)

        val err = result as? ToolResult.Error
        assertNotNull(err)
        assertEquals("Hand 'DisabledHand' is disabled", err?.message)
    }

    @Test
    fun `run returns ok with no steps message when steps are empty`() = runTest {
        val hand = Hand(id = "1", name = "EmptyHand", steps = "[]")

        val result = repository.run(hand, executor, ctx)

        val ok = result as? ToolResult.Ok
        assertNotNull(ok)
        assertEquals("No steps defined for hand 'EmptyHand'", ok?.output)
    }

    @Test
    fun `run returns ok with no steps message when all steps lack a tool`() = runTest {
        val hand = Hand(
            id = "1",
            name = "BadStep",
            steps = """[{"args": "{}"}]""",
        )

        val result = repository.run(hand, executor, ctx)

        val ok = result as? ToolResult.Ok
        assertNotNull(ok)
        assertEquals("No steps defined for hand 'BadStep'", ok?.output)
    }
}
