package com.aura.kg

import com.aura.kg.KgNode
import com.aura.kg.NodeType
import com.aura.kg.KnowledgeGraphRepository
import com.aura.tools.KnowledgeGraphTool
import io.mockk.Runs
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.just
import io.mockk.mockk
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class ConversationKgExtractorTest {

    @Test
    fun `extract debounces multiple calls into one extraction`() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val tool = mockk<KnowledgeGraphTool> {
            coEvery { extract(any()) } returns Pair(emptyList(), emptyList())
        }
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { saveGraph(any(), any(), any()) } just Runs
        }
        val extractor = ConversationKgExtractor(tool, repo, dispatcher, Unit)

        extractor.extract("first")
        extractor.extract("second")
        extractor.extract("third")

        advanceTimeBy(6_000L)

        coVerify(exactly = 1) { tool.extract("third") }
        coVerify(exactly = 0) { tool.extract("first") }
        coVerify(exactly = 0) { tool.extract("second") }

        extractor.shutdown()
    }

    @Test
    fun `extract saves graph when nodes or edges present`() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val tool = mockk<KnowledgeGraphTool> {
            coEvery { extract(any()) } returns Pair(
                listOf(KgNode("n1", "Kotlin", NodeType.UNKNOWN)),
                emptyList(),
            )
        }
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { saveGraph(any(), any(), any()) } just Runs
        }
        val extractor = ConversationKgExtractor(tool, repo, dispatcher, Unit)
        val provenance = com.aura.provenance.ConversationProvenance("conv-7", 77L)

        extractor.extract("Kotlin is a language", provenance)
        advanceTimeBy(6_000L)

        coVerify(exactly = 1) { repo.saveGraph(any(), any(), provenance) }

        extractor.shutdown()
    }

    @Test
    fun `extract skips save when empty`() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val tool = mockk<KnowledgeGraphTool> {
            coEvery { extract(any()) } returns Pair(emptyList(), emptyList())
        }
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { saveGraph(any(), any(), any()) } just Runs
        }
        val extractor = ConversationKgExtractor(tool, repo, dispatcher, Unit)

        extractor.extract("just chatting")
        advanceTimeBy(6_000L)

        coVerify(exactly = 0) { repo.saveGraph(any(), any(), any()) }

        extractor.shutdown()
    }

    @Test
    fun `extract swallows exceptions without rethrowing`() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val tool = mockk<KnowledgeGraphTool> {
            coEvery { extract(any()) } throws RuntimeException("boom")
        }
        val repo = mockk<KnowledgeGraphRepository>()
        val extractor = ConversationKgExtractor(tool, repo, dispatcher, Unit)

        // Should not throw even though extract() throws
        extractor.extract("throw at me")
        advanceTimeBy(6_000L)

        coVerify(exactly = 0) { repo.saveGraph(any(), any(), any()) }

        extractor.shutdown()
    }

    @Test
    fun `shutdown cancels pending debounce`() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val tool = mockk<KnowledgeGraphTool> {
            coEvery { extract(any()) } returns Pair(emptyList(), emptyList())
        }
        val repo = mockk<KnowledgeGraphRepository> {
            coEvery { saveGraph(any(), any(), any()) } just Runs
        }
        val extractor = ConversationKgExtractor(tool, repo, dispatcher, Unit)

        extractor.extract("delayed")
        extractor.shutdown()

        // Allow the test scheduler to drain; no extraction should run after shutdown.
        advanceTimeBy(10_000L)
        coVerify(exactly = 0) { tool.extract("delayed") }
        coVerify(exactly = 0) { repo.saveGraph(any(), any(), any()) }
    }

    @Test
    fun `blank input is ignored`() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val tool = mockk<KnowledgeGraphTool>()
        val repo = mockk<KnowledgeGraphRepository>()
        val extractor = ConversationKgExtractor(tool, repo, dispatcher, Unit)

        extractor.extract("   ")
        extractor.extract("")
        advanceTimeBy(6_000L)

        coVerify(exactly = 0) { tool.extract(any()) }
        coVerify(exactly = 0) { repo.saveGraph(any(), any(), any()) }

        extractor.shutdown()
    }
}
