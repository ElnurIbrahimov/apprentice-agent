package com.aura.proactive

import android.content.Context
import androidx.work.WorkerParameters
import com.aura.memory.MemoryStore
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.junit.Test
import kotlin.test.assertEquals

class DecayWorkerTest {

    @Test
    fun `doWork calls runDecayPass and returns success`() = runBlocking {
        val mockContext = mockk<Context>(relaxed = true)
        val mockParams = mockk<WorkerParameters>(relaxed = true)
        val mockMemoryStore = mockk<MemoryStore>(relaxed = true)

        val worker = DecayWorker(mockContext, mockParams, mockMemoryStore)

        val result = worker.doWork()

        coVerify(exactly = 1) { mockMemoryStore.runDecayPass() }
        assertEquals(androidx.work.ListenableWorker.Result.success(), result)
    }
}
