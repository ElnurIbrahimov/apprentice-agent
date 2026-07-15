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
import com.aura.tasks.ReminderDao
import com.aura.tasks.TaskDao
import com.aura.tools.CalendarReadTool
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
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

sealed interface HomeLoadState {
    data object Loading : HomeLoadState
    data object Empty : HomeLoadState
    data object Content : HomeLoadState
    data class Error(
        val message: String,
        val hasPartialContent: Boolean,
    ) : HomeLoadState
}

fun resolveHomeLoadState(hasData: Boolean, dataSourceError: String?): HomeLoadState = when {
    dataSourceError != null -> HomeLoadState.Error(dataSourceError, hasPartialContent = hasData)
    hasData -> HomeLoadState.Content
    else -> HomeLoadState.Empty
}

internal fun extractUserName(memories: List<MemoryEntity>): String? = memories.firstNotNullOfOrNull { memory ->
    val content = memory.content.trim()
    val candidate = when {
        content.contains("my name is ", ignoreCase = true) ->
            Regex("my name is\\s+(.+)", RegexOption.IGNORE_CASE).find(content)?.groupValues?.get(1).orEmpty()
        content.startsWith("i am ", ignoreCase = true) && memory.category == "person" -> content.drop(5)
        content.startsWith("call me ", ignoreCase = true) -> content.drop(8)
        else -> ""
    }.trim().trimEnd('.', ',', '!', '?').take(39)
    candidate.takeIf { it.isNotBlank() }
}

data class HomeUiState(
    val today: List<String> = emptyList(),  // calendar events (formatted)
    val recentMemories: List<MemoryEntity> = emptyList(),
    val pendingTasks: List<String> = emptyList(),
    val userName: String? = null,
    val hour: Int = 0,
    val loadState: HomeLoadState = HomeLoadState.Loading,
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
    /**
     * Up to 3 upcoming reminders, soonest first. Rendered on Home
     * as a BriefCard so the user can see what's pending without
     * opening the Reminders screen.
     */
    val upcomingReminders: List<String> = emptyList(),
    /**
     * Quick counts for the second row of quick-action cards on
     * Home — Hands, Tools, Proactive. None of these drive any
     * logic, just badges.
     */
    val handsCount: Int = 0,
    val toolsCount: Int = 0,
    val proactiveCount: Int = 0,
    val skillsCount: Int = 0,
) {
    val isEmptyResolved: Boolean get() = loadState is HomeLoadState.Empty

    fun hasHomeData(): Boolean = today.isNotEmpty() ||
        recentMemories.isNotEmpty() ||
        pendingTasks.isNotEmpty() ||
        upcomingReminders.isNotEmpty() ||
        proactiveEvent != null ||
        proactiveUnreadCount > 0 ||
        handsCount > 0 ||
        proactiveCount > 0
}

@HiltViewModel
class HomeViewModel @Inject constructor(
    application: Application,
    private val memoryStore: MemoryStore,
    private val taskDao: TaskDao,
    private val proactiveEvents: ProactiveEvents,
    private val calendarReadTool: CalendarReadTool,
    private val knowledgeGraphRepository: KnowledgeGraphRepository,
    private val reminderDao: ReminderDao,
    private val handDao: com.aura.hands.HandDao,
    private val toolRegistry: com.aura.agent.ToolRegistry,
    private val skillsStore: com.aura.skills.SkillsStore,
) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(HomeUiState())
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    private fun updateObserved(transform: (HomeUiState) -> HomeUiState) {
        _state.update { current ->
            val updated = transform(current)
            when (current.loadState) {
                HomeLoadState.Loading,
                is HomeLoadState.Error,
                -> updated
                else -> updated.copy(
                    loadState = resolveHomeLoadState(updated.hasHomeData(), dataSourceError = null),
                )
            }
        }
    }

    init {
        refresh()
        observeProactiveEvents()
        observeTasks()
        observeMemories()
        observeReminders()
        observeHands()
        observeSkills()
        loadToolsCount()
    }

    private fun observeReminders() {
        viewModelScope.launch {
            reminderDao.observeUpcoming(System.currentTimeMillis()).collect { rs ->
                val fmt = java.text.DateFormat.getTimeInstance(java.text.DateFormat.SHORT)
                val lines = rs.take(3).map { r ->
                    "${fmt.format(java.util.Date(r.triggerAt))} · ${r.message}"
                }
                updateObserved { it.copy(upcomingReminders = lines) }
            }
        }
    }

    private fun observeProactiveEvents() {
        viewModelScope.launch {
            proactiveEvents.latest.collect { event ->
                updateObserved { it.copy(proactiveEvent = event) }
            }
        }
        viewModelScope.launch {
            proactiveEvents.unreadCount.collect { count ->
                updateObserved { it.copy(proactiveUnreadCount = count) }
            }
        }
        viewModelScope.launch {
            proactiveEvents.history.collect { events ->
                updateObserved { it.copy(proactiveCount = events.size) }
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
                updateObserved { it.copy(pendingTasks = titles) }
                rebuildBriefContext()
            }
        }
    }

    private fun observeHands() {
        viewModelScope.launch {
            handDao.observeAll().collect { hands ->
                updateObserved { it.copy(handsCount = hands.size) }
            }
        }
    }

    private fun observeSkills() {
        viewModelScope.launch {
            skillsStore.awaitLoaded()
            skillsStore.skills.collect { skills ->
                updateObserved { it.copy(skillsCount = skills.size) }
            }
        }
    }

    private fun loadToolsCount() {
        viewModelScope.launch {
            updateObserved { it.copy(toolsCount = toolRegistry.definitions().size) }
        }
    }

    private fun observeMemories() {
        viewModelScope.launch {
            memoryStore.observeCount().collect { _ ->
                // When memories change, refresh the recent list
                val recent = memoryStore.recent(5)
                updateObserved { it.copy(recentMemories = recent) }
                rebuildBriefContext()
            }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            _state.update { it.copy(loadState = HomeLoadState.Loading) }
            try {
                val now = System.currentTimeMillis()
                val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
                val recentForProfile = memoryStore.recent(50)
                val recent = recentForProfile.take(5)
                val tasks = taskDao.allPending().take(5).map { it.title }
                val calendarResult = runCatching { calendarReadTool.readTodaysEvents() }
                val reminders = reminderDao.observeUpcoming(now).first().take(3).map { reminder ->
                    val time = java.text.DateFormat.getTimeInstance(java.text.DateFormat.SHORT)
                        .format(Date(reminder.triggerAt))
                    "$time · ${reminder.message}"
                }
                val hands = handDao.observeAll().first()
                val latestProactive = proactiveEvents.latest.first()
                val proactiveHistory = proactiveEvents.history.first()
                val proactiveUnread = proactiveEvents.unreadCount.first()
                val toolsCount = toolRegistry.definitions().size
                val name = extractUserName(recentForProfile)

                val loaded = _state.value.copy(
                    today = calendarResult.getOrDefault(emptyList()),
                    recentMemories = recent,
                    pendingTasks = tasks,
                    userName = name,
                    hour = hour,
                    proactiveEvent = latestProactive,
                    proactiveUnreadCount = proactiveUnread,
                    upcomingReminders = reminders,
                    handsCount = hands.size,
                    toolsCount = toolsCount,
                    proactiveCount = proactiveHistory.size,
                )
                val calendarError = calendarResult.exceptionOrNull()?.let {
                    "Calendar is unavailable. Other Home information is still available."
                }
                _state.value = loaded.copy(
                    loadState = resolveHomeLoadState(loaded.hasHomeData(), calendarError),
                )
                rebuildBriefContext()
            } catch (error: Exception) {
                _state.update { current ->
                    current.copy(
                        loadState = HomeLoadState.Error(
                            message = "Home data is unavailable. Check permissions and try again.",
                            hasPartialContent = current.hasHomeData(),
                        ),
                    )
                }
            }
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
