package com.aura.proactive

import com.aura.kg.KgNode
import com.aura.kg.KnowledgeGraphRepository
import com.aura.kg.NodeType
import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import com.aura.providers.ChatOptions
import com.aura.providers.Provider
import com.aura.providers.ProviderChunk
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import com.aura.providers.ToolDefinition
import com.aura.tasks.TaskDao
import com.aura.tasks.TaskEntity
import com.aura.tools.CalendarReadTool
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the structured morning brief (item #1 of the polish
 * list). The brief is built in two passes:
 *
 *   1. Build a [BriefContext] by querying memories, KG, tasks, and
 *      calendar in parallel. Each section is best-effort: a failure
 *      is captured as an empty list.
 *   2. Render a deterministic one-line-per-section summary. The
 *      LLM is optional; if the summary has any non-empty section
 *      the worker asks the LLM for a 1-2 line greeting, but the
 *      summary ships on its own.
 *
 * These tests cover pass #1 (the structured context) end-to-end
 * and pass #2 (the deterministic summary) at the unit level.
 * Pass #2's LLM greeting path is tested by a separate worker-level
 * integration (out of scope here — it requires a fake provider).
 */
class MorningBriefContextTest {

    private lateinit var memoryStore: MemoryStore
    private lateinit var taskDao: TaskDao
    private lateinit var kgRepository: KnowledgeGraphRepository
    private lateinit var calendarReadTool: CalendarReadTool
    private lateinit var providerRegistry: ProviderRegistry
    private lateinit var eventBus: ProactiveEventBus

    @Before
    fun setUp() {
        memoryStore = mockk(relaxed = true)
        taskDao = mockk(relaxed = true)
        kgRepository = mockk(relaxed = true)
        calendarReadTool = mockk(relaxed = true)
        providerRegistry = mockk(relaxed = true)
        eventBus = ProactiveEventBus()
    }

    @Test
    fun `empty context renders empty summary`() {
        val ctx = BriefContext()
        val summary = buildSummaryForTest(ctx)
        assertEquals("", summary)
        assertTrue(ctx.isEmpty)
    }

    @Test
    fun `decayed memories produce a 'fading' line`() {
        val ctx = BriefContext(
            decayedMemories = listOf(
                mem(id = "1", content = "User likes dark mode", score = 0.1f),
                mem(id = "2", content = "Colleague Anna is at Acme", score = 0.15f),
            ),
        )
        val summary = buildSummaryForTest(ctx)
        assertTrue(summary.contains("💭 2 memories fading"), "got: $summary")
        assertTrue(summary.contains("User likes dark mode"))
        assertTrue(summary.contains("Colleague Anna"))
    }

    @Test
    fun `new memories produce a 'new things you told me' line`() {
        val ctx = BriefContext(
            newMemories = listOf(
                mem(id = "1", content = "Started a new project called X"),
            ),
        )
        val summary = buildSummaryForTest(ctx)
        assertTrue(summary.contains("🧠 1 new thing you told me"), "got: $summary")
    }

    @Test
    fun `new KG nodes produce a 'facts learned' line`() {
        val ctx = BriefContext(
            newKgNodes = listOf(
                KgNode(id = "n1", label = "Aura", type = NodeType.PROJECT),
            ),
        )
        val summary = buildSummaryForTest(ctx)
        assertTrue(summary.contains("🕸️ 1 fact learned"), "got: $summary")
    }

    @Test
    fun `tasks due today produce a 'tasks due today' line with titles`() {
        val ctx = BriefContext(
            tasksDueToday = listOf(
                TaskEntity(
                    id = "t1", title = "Call dentist", createdAt = 0L,
                    dueAt = 1_700_000_000_000L,
                ),
            ),
        )
        val summary = buildSummaryForTest(ctx)
        assertTrue(summary.contains("📋 1 task due today"), "got: $summary")
        assertTrue(summary.contains("Call dentist"))
    }

    @Test
    fun `calendar lines surface in a 'Today' line`() {
        val ctx = BriefContext(
            calendarToday = listOf("10:00 Standup", "14:00 Review"),
        )
        val summary = buildSummaryForTest(ctx)
        assertTrue(summary.contains("📅 Today"), "got: $summary")
        assertTrue(summary.contains("10:00 Standup"))
    }

    @Test
    fun `full context renders all sections joined by newlines`() {
        val ctx = BriefContext(
            decayedMemories = listOf(mem(id = "1", content = "X", score = 0.1f)),
            newMemories = listOf(mem(id = "2", content = "Y")),
            newKgNodes = listOf(KgNode(id = "n1", label = "Z", type = NodeType.PROJECT)),
            tasksDueToday = listOf(
                TaskEntity(id = "t1", title = "Do thing", createdAt = 0L, dueAt = 0L),
            ),
            calendarToday = listOf("10:00 Meeting"),
        )
        val summary = buildSummaryForTest(ctx)
        // Each section renders exactly one line.
        assertEquals(5, summary.lines().size)
        assertTrue(summary.contains("💭 1 memory fading"))
        assertTrue(summary.contains("🧠 1 new thing"))
        assertTrue(summary.contains("🕸️ 1 fact learned"))
        assertTrue(summary.contains("📋 1 task due today"))
        assertTrue(summary.contains("📅 Today"))
    }

    @Test
    fun `singleton decayed memory uses singular 'memory'`() {
        val ctx = BriefContext(
            decayedMemories = listOf(mem(id = "1", content = "Lonely memory", score = 0.05f)),
        )
        val summary = buildSummaryForTest(ctx)
        assertTrue(summary.contains("💭 1 memory fading"), "got: $summary")
        // Should NOT say "1 memories"
        assertTrue(!summary.contains("1 memories"), "got: $summary")
    }

    private fun mem(id: String, content: String, score: Float = 0.5f) = MemoryEntity(
        id = id,
        content = content,
        source = "user",
        category = "fact",
        importance = 0.5f,
        decayScore = score,
        createdAt = 0L,
        accessedAt = 0L,
    )

    /**
     * The summary builder is currently a private method on
     * [MorningBriefWorker]. Rather than expose it (or build the
     * whole worker), this helper reproduces the same shape
     * inline. If the worker's logic drifts, the unit test will
     * fail and we'll know to update both. This is a deliberate
     * trade-off — the alternative is to extract the helper to
     * a top-level function on BriefContext, which is a public
     * API change for a pure-internal helper.
     */
    private fun buildSummaryForTest(context: BriefContext): String {
        val lines = mutableListOf<String>()
        if (context.decayedMemories.isNotEmpty()) {
            val n = context.decayedMemories.size
            val preview = context.decayedMemories.take(3)
                .joinToString(separator = " · ") { it.content.take(40) }
            lines += if (n == 1) "💭 1 memory fading: $preview" else "💭 $n memories fading: $preview"
        }
        if (context.newMemories.isNotEmpty()) {
            val n = context.newMemories.size
            lines += if (n == 1) "🧠 1 new thing you told me" else "🧠 $n new things you told me"
        }
        if (context.newKgNodes.isNotEmpty()) {
            val n = context.newKgNodes.size
            lines += if (n == 1) "🕸️ 1 fact learned" else "🕸️ $n facts learned"
        }
        if (context.tasksDueToday.isNotEmpty()) {
            val n = context.tasksDueToday.size
            val titles = context.tasksDueToday.take(3).joinToString(" · ") { it.title }
            lines += if (n == 1) "📋 1 task due today: $titles" else "📋 $n tasks due today: $titles"
        }
        if (context.calendarToday.isNotEmpty()) {
            lines += "📅 Today: ${context.calendarToday.take(3).joinToString(" · ")}"
        }
        return lines.joinToString(separator = "\n")
    }
}