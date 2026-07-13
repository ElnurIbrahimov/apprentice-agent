package com.aura.ui.viewmodel

import android.app.Application
import androidx.work.WorkManager
import com.aura.tasks.ReminderDao
import com.aura.tasks.ReminderEntity
import com.aura.tasks.ReminderStore
import com.aura.tasks.TaskDao
import com.aura.tasks.TaskEntity
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for [TasksViewModel]: loading tasks, adding tasks with full metadata,
 * cancelling reminders, and deleting tasks with linked reminder cleanup.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class TasksViewModelTest {

    private lateinit var app: Application
    private val taskDao = mockk<TaskDao>(relaxed = true)
    private val reminderDao = mockk<ReminderDao>(relaxed = true)
    private val reminderStore = mockk<ReminderStore>(relaxed = true)
    private val reminderFlow = MutableStateFlow(emptyList<ReminderEntity>())
    private val workManager = mockk<WorkManager>(relaxed = true)

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        app = mockk(relaxed = true)
        mockkStatic(WorkManager::class)
        every { WorkManager.getInstance(any()) } returns workManager
        every { reminderStore.observeUpcoming() } returns reminderFlow
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        unmockkStatic(WorkManager::class)
    }

    @Test
    fun `load surfaces tasks and reminders`() = runTest {
        val tasks = listOf(TaskEntity(id = "t1", title = "A", createdAt = 1L))
        coEvery { taskDao.all() } returns tasks
        every { reminderDao.observeUpcoming(any()) } returns reminderFlow
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        assertEquals(tasks, vm.state.value.tasks)
        assertFalse(vm.state.value.loading)
    }

    @Test
    fun `addTask inserts entity with description dueAt priority and tags`() = runTest {
        coEvery { taskDao.insert(any()) } returns Unit
        coEvery { taskDao.all() } returnsMany listOf(emptyList(), emptyList())
        every { reminderDao.observeUpcoming(any()) } returns flowOf(emptyList())
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        vm.addTask("Title", "Desc", 1234L, 2, "tag1, tag2")
        coVerify {
            taskDao.insert(
                match {
                    it.title == "Title" &&
                        it.description == "Desc" &&
                        it.dueAt == 1234L &&
                        it.priority == 2 &&
                        it.tags == "tag1, tag2" &&
                        it.status == "pending"
                }
            )
        }
    }

    @Test
    fun `addTask ignores blank title`() = runTest {
        coEvery { taskDao.all() } returnsMany listOf(emptyList(), emptyList())
        every { reminderDao.observeUpcoming(any()) } returns flowOf(emptyList())
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        vm.addTask("   ")
        coVerify(exactly = 0) { taskDao.insert(any()) }
    }

    @Test
    fun `deleteTask deletes task and linked reminder`() = runTest {
        coEvery { taskDao.all() } returns emptyList()
        every { reminderDao.observeUpcoming(any()) } returns flowOf(emptyList())
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        vm.deleteTask("t1")
        coVerify { taskDao.delete("t1") }
        coVerify { reminderDao.deleteByTaskId("t1") }
        coVerify { workManager.cancelUniqueWork("task-t1") }
    }

    @Test
    fun `markDone completes task and cancels reminder`() = runTest {
        coEvery { taskDao.all() } returns emptyList()
        every { reminderDao.observeUpcoming(any()) } returns flowOf(emptyList())
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        vm.markDone("t1")
        coVerify { taskDao.markComplete("t1", any()) }
        coVerify { reminderDao.deleteByTaskId("t1") }
        coVerify { workManager.cancelUniqueWork("task-t1") }
    }

    @Test
    fun `cancelReminder deletes reminder and cancels work by id`() = runTest {
        coEvery { taskDao.all() } returns emptyList()
        every { reminderDao.observeUpcoming(any()) } returns flowOf(emptyList())
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        val id = UUID.randomUUID().toString()
        vm.cancelReminder(id)
        coVerify { reminderStore.cancel(id) }
    }

    @Test
    fun `reminder flow updates state independently`() = runTest {
        coEvery { taskDao.all() } returns emptyList()
        every { reminderDao.observeUpcoming(any()) } returns reminderFlow
        val vm = TasksViewModel(app, taskDao, reminderDao, reminderStore)
        val reminder = ReminderEntity(id = "r1", workId = "w1", message = "M", triggerAt = 9_999_999_999L)
        reminderFlow.value = listOf(reminder)
        assertEquals(listOf(reminder), vm.state.value.reminders)
    }
}
