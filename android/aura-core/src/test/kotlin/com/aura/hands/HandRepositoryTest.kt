package com.aura.hands

import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolResult
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import kotlin.test.assertFailsWith

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
    fun `variables substitute into step args and successful run is persisted`() = runTest {
        val hand = Hand(
            id = "weather",
            name = "Weather",
            variables = """{"city":"Baku","unit":"metric"}""",
            steps = """[{"tool":"web_search","args":{"query":"weather in {{city}} using {{unit}}"}}]""",
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        coEvery {
            executor.execute("web_search", """{"query":"weather in Tokyo using metric"}""", ctx)
        } returns ToolResult.Ok("Sunny")

        val result = repository.run(
            hand = hand,
            executor = executor,
            ctx = ctx,
            variables = mapOf("city" to "Tokyo"),
            trigger = HandRunTrigger.MANUAL.value,
        )

        assertEquals(true, result is ToolResult.Ok)
        coVerify {
            dao.updateRun(match {
                it.handId == "weather" && it.status == HandRunStatus.SUCCESS.value &&
                    it.trigger == HandRunTrigger.MANUAL.value && it.finishedAt != null
            })
        }
    }

    @Test
    fun `failed condition skips execution and records reason`() = runTest {
        val hand = Hand(
            id = "work",
            name = "Work only",
            variables = """{"mode":"home"}""",
            conditions = """[{"variable":"mode","operator":"equals","value":"work"}]""",
            steps = """[{"tool":"web_search","args":{"query":"status"}}]""",
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit

        val result = repository.run(hand, executor, ctx)

        val ok = result as ToolResult.Ok
        assertEquals(true, ok.output.contains("Skipped"))
        coVerify(exactly = 0) { executor.execute(any(), any(), any()) }
        coVerify {
            dao.updateRun(match {
                it.status == HandRunStatus.SKIPPED.value && it.output.contains("mode")
            })
        }
    }

    @Test
    fun `failed step records failed history with step number`() = runTest {
        val hand = Hand(
            id = "broken",
            name = "Broken",
            steps = """[{"tool":"battery_state","args":{}}]""",
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        coEvery { executor.execute("battery_state", "{}", ctx) } returns
            ToolResult.Error("unavailable", "sensor_error")

        repository.run(hand, executor, ctx, trigger = HandRunTrigger.AGENT.value)

        coVerify {
            dao.updateRun(match {
                it.status == HandRunStatus.FAILED.value && it.failedStep == 1 &&
                    it.trigger == HandRunTrigger.AGENT.value && it.output.contains("unavailable")
            })
        }
    }

    @Test
    fun `thrown tool exception becomes terminal failed history`() = runTest {
        val hand = Hand(
            id = "throws",
            name = "Throws",
            steps = """[{"tool":"unstable","args":{}}]""",
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        coEvery { executor.execute("unstable", "{}", ctx) } throws IllegalStateException("boom")

        val result = repository.run(hand, executor, ctx)

        assertEquals(true, result is ToolResult.Error)
        coVerify {
            dao.updateRun(match {
                it.status == HandRunStatus.FAILED.value && it.finishedAt != null &&
                    it.failedStep == 1 && it.output.contains("boom")
            })
        }
    }

    @Test
    fun `tool cancellation propagates without a false terminal result`() = runTest {
        val hand = Hand(
            id = "cancelled",
            name = "Cancelled",
            steps = """[{"tool":"slow","args":{}}]""",
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        coEvery { executor.execute("slow", "{}", ctx) } throws CancellationException("stop")

        assertFailsWith<CancellationException> { repository.run(hand, executor, ctx) }
        coVerify(exactly = 0) {
            dao.updateRun(match { it.finishedAt != null })
        }
    }

    @Test
    fun `history redacts secret inputs caps output and begins in running state`() = runTest {
        val secret = "do-not-store-this"
        val hand = Hand(
            id = "bounded",
            name = "Bounded",
            variables = """{"apiKey":"$secret"}""",
            steps = """[{"tool":"large_output","args":{}}]""",
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        coEvery { executor.execute("large_output", "{}", ctx) } returns ToolResult.Ok("x".repeat(9_000))

        repository.run(hand, executor, ctx)

        coVerify {
            dao.insertRun(match { it.status == HandRunStatus.RUNNING.value })
        }
        coVerify {
            dao.updateRun(match {
                it.status == HandRunStatus.SUCCESS.value &&
                    it.output.length == 8_000 &&
                    it.variablesJson.contains("[redacted]") &&
                    !it.variablesJson.contains(secret)
            })
        }
    }

    @Test
    fun `malformed automation configuration fails closed before tools execute`() = runTest {
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        val malformedHands = listOf(
            Hand(id = "vars", name = "Bad variables", variables = "{", steps = "[]"),
            Hand(id = "conditions", name = "Bad conditions", conditions = "[", steps = "[]"),
            Hand(id = "steps", name = "Bad steps", steps = "not-json"),
        )

        val results = malformedHands.map { repository.run(it, executor, ctx) }

        assertEquals(true, results.all { it is ToolResult.Error })
        coVerify(exactly = 0) { executor.execute(any(), any(), any()) }
        coVerify(exactly = 3) {
            dao.updateRun(match { it.status == HandRunStatus.FAILED.value })
        }
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

    @Test
    fun `resumed run starts at the stopped step and preserves step numbering`() = runTest {
        val hand = Hand(
            id = "resume",
            name = "Resume me",
            steps = """[
                {"tool":"first","args":{}},
                {"tool":"paid","args":{"prompt":"continue"}},
                {"tool":"last","args":{}}
            ]""",
        )
        val resumeContext = ToolContext(
            conversationId = "hand:resume:retry:r1",
            approvedRemoteCostTools = setOf("paid"),
        )
        coEvery { dao.insertRun(any()) } returns Unit
        coEvery { dao.updateRun(any()) } returns Unit
        coEvery { executor.execute("paid", """{"prompt":"continue"}""", resumeContext) } returns ToolResult.Ok("paid done")
        coEvery { executor.execute("last", "{}", resumeContext) } returns ToolResult.Ok("last done")

        val result = repository.run(
            hand = hand,
            executor = executor,
            ctx = resumeContext,
            startStepIndex = 1,
            trigger = HandRunTrigger.RESUME.value,
        )

        val ok = result as ToolResult.Ok
        assertEquals(
            "Hand 'Resume me' completed.\nStep 2 (paid): paid done\nStep 3 (last): last done",
            ok.output,
        )
        coVerify(exactly = 0) { executor.execute("first", any(), any()) }
        coVerify { executor.execute("paid", """{"prompt":"continue"}""", resumeContext) }
        coVerify { executor.execute("last", "{}", resumeContext) }
    }
}
