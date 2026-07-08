package com.aura.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aura.kg.KnowledgeGraphRepository
import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import com.aura.proactive.BriefContext
import com.aura.proactive.ProactiveEventBus
import com.aura.proactive.ProactiveEvents
import com.aura.tasks.TaskDao
import com.aura.tools.CalendarReadTool
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/**
 * Memory decayed below this threshold is "fading" and surfaces in the
 * morning brief. Mirrors the threshold in
 * [com.aura.proactive.MorningBriefWorker] so Home and the brief agree.
 */
private const val DECAY_FADING_THRESHOLD = 0.4f

data class HomeUiState(
    val today: List<String> = emptyList(),  // calendar events (formatted)
    val recentMemories: List<MemoryEntity> = emptyList(),
    val pendingTasks: List<String> = emptyList(),
    val userName: String? = null,
    val hour: Int = 0,
    val loading: Boolean = true,
    val proactiveEvent: ProactiveEventBus.Event? = null,
    /**
     * Count of proactive events emitted since the user last opened
     * the history screen (or dismissed the Home card). Drives the
     * "📬 N today" badge next to the proactive event card on Home.
     */
    val proactiveUnreadCount: Int = 0,
    /**
     * Built from the same data the screen shows. Rendered via
     * [com.aura.ui.util.toSummary] as a one-line-per-section
     * greeting instead of a count. Null = no data yet; Home
     * screen falls back to its count-based string.
     */
    val briefContext: BriefContext = BriefContext(),
)

@HiltViewModel
class HomeViewModel @Inject constructor(
    application: Application,
    private val memoryStore: MemoryStore,
    private val taskDao: TaskDao,
    private val proactiveEvents: ProactiveEvents,
    private val calendarReadTool: CalendarReadTool,
    private val knowledgeGraphRepository: KnowledgeGraphRepository,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(HomeUiState())
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    init {
        refresh()
        observeProactiveEvents()
        observeTasks()
        observeMemories()
    }

    private fun observeProactiveEvents() {
        viewModelScope.launch {
            proactiveEvents.latest.collect { event ->
                _state.update { it.copy(proactiveEvent = event) }
            }
        }
        viewModelScope.launch {
            proactiveEvents.unreadCount.collect { count ->
                _state.update { it.copy(proactiveUnreadCount = count) }
            }
        }
    }

    fun dismissProactiveEvent() {
        proactiveEvents.dismiss()
    }

    /**
     * Called when the user taps the proactive event card on Home
     * (the in-place "see all" link) or navigates to the proactive
     * history screen. Marks all currently-unread events as seen
     * so the badge clears.
     */
    fun onProactiveHistoryOpened() {
        proactiveEvents.markSeen()
    }

    private fun observeTasks() {
        viewModelScope.launch {
            taskDao.observeAll().collect { tasks ->
                val titles = tasks.take(5).map { t -> t.title }
                _state.update { it.copy(pendingTasks = titles) }
                rebuildBriefContext()
            }
        }
    }

    private fun observeMemories() {
        viewModelScope.launch {
            memoryStore.observeCount().collect { _ ->
                // When memories change, refresh the recent list
                val recent = memoryStore.recent(5)
                _state.update { it.copy(recentMemories = recent) }
                rebuildBriefContext()
            }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            val cal = java.util.Calendar.getInstance()
            val hour = cal.get(java.util.Calendar.HOUR_OF_DAY)
            // Recent memories
            val recent = memoryStore.recent(5)
            // Try to find the user's name in a memory (preference: "i am X" or "my name is X")
            val name = memoryStore.recent(50).mapNotNull { mem ->
                val lower = mem.content.lowercase()
                when {
                    lower.contains("my name is ") -> mem.content.substringAfter("my name is ", "").trim().take(40)
                    lower.startsWith("i am ") && mem.category == "person" -> mem.content.removePrefix("i am ").removePrefix("I am ").take(40)
                    lower.startsWith("call me ") -> mem.content.removePrefix("call me ").removePrefix("Call me ").take(40)
                    else -> null
                }
            }.firstOrNull()?.takeIf { it.isNotBlank() && it.length < 40 }
            // Tasks
            val tasks = taskDao.allPending().take(5).map { it.title }
            // Calendar — best effort, ignore exceptions.
            val events = runCatching { calendarReadTool.readTodaysEvents() }.getOrDefault(emptyList())
            _state.update {
                it.copy(
                    today = events,
                    recentMemories = recent,
                    pendingTasks = tasks,
                    userName = name,
                    hour = hour,
                    loading = false,
                )
            }
            rebuildBriefContext()
        }
    }

    /**
     * Build a [BriefContext] from the data already loaded into state
     * and publish it. The Home screen renders it via
     * [com.aura.ui.util.toSummary] for a one-line-per-section greeting
     * instead of the count-based "I remember 47 things" string.
     *
     * Decayed memories: those with [MemoryEntity.decayScore] below
     * [DECAY_FADING_THRESHOLD]. New memories: those created in the
     * last 24h. Tasks due today: pending tasks with a dueAt in
     * today's range.
     */
    private suspend fun rebuildBriefContext() {
        val current = _state.value
        val today = Calendar.getInstance()
        val startOfDay = (today.clone() as Calendar).apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
        val endOfDay = startOfDay + 24L * 60L * 60L * 1000L
        val dayAgo = System.currentTimeMillis() - 24L * 60L * 60L * 1000L

        val decayed = runCatching {
            memoryStore.recent(50).filter { it.decayScore < DECAY_FADING_THRESHOLD }.take(5)
        }.getOrDefault(emptyList())
        val newMems = runCatching {
            memoryStore.recent(50).filter { it.createdAt >= dayAgo }.take(5)
        }.getOrDefault(emptyList())
        val newKg = runCatching { knowledgeGraphRepository.recentSince(dayAgo, 5) }
            .getOrDefault(emptyList())
        val tasksToday = runCatching { taskDao.dueInRange(startOfDay, endOfDay) }
            .getOrDefault(emptyList())
            .take(5)

        val ctx = BriefContext(
            decayedMemories = decayed,
            newMemories = newMems,
            newKgNodes = newKg,
            tasksDueToday = tasksToday,
            calendarToday = current.today,
        )
        _state.update { it.copy(briefContext = ctx) }
    }
}
