package com.aura.ui.viewmodel

import android.net.Uri
import com.aura.documents.DocumentEntity
import com.aura.documents.DocumentImportResult
import com.aura.documents.DocumentRepository
import com.aura.documents.DocumentTextExtractor
import com.aura.documents.ExtractedDocument
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull

class DocumentImportViewModelTest {
    private val extractor = mockk<DocumentTextExtractor>()
    private val repository = mockk<DocumentRepository>()

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        every { repository.observeAll() } returns flowOf(emptyList())
    }

    @After
    fun tearDown() = Dispatchers.resetMain()

    @Test
    fun `import extracts persists and reports searchable chunk count`() = runTest {
        val uri = mockk<Uri>()
        val sourceUri = "content://docs/bible"
        every { uri.toString() } returns sourceUri
        val extracted = ExtractedDocument("hash", "bible.docx", "docx", sourceUri, "world text")
        val entity = DocumentEntity("hash", "bible.docx", "docx", sourceUri, 1L, 10, 2)
        coEvery { extractor.extract(uri) } returns extracted
        coEvery {
            repository.import("hash", "bible.docx", "docx", sourceUri, "world text")
        } returns DocumentImportResult(entity, 2)
        val viewModel = DocumentImportViewModel(extractor, repository)

        viewModel.import(uri)
        advanceUntilIdle()

        coVerify { repository.import("hash", "bible.docx", "docx", sourceUri, "world text") }
        assertFalse(viewModel.state.value.importing)
        assertNull(viewModel.state.value.error)
        assertEquals("Imported bible.docx · 2 searchable chunks", viewModel.state.value.message)
    }
}