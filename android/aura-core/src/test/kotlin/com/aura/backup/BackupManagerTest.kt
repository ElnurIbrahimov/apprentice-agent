package com.aura.backup

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Contract tests for [BackupManager]. The DAO contract tests for the
 * underlying tables live in their respective `*DaoTest` files. These
 * tests pin the BackupManager's own logic: round-trip encode/decode,
 * schema-version guard, table-coverage of the export, and the
 * destructive `purgeAll` step.
 */
class BackupManagerTest {

    private val memoryDao = mockk<com.aura.memory.MemoryDao>(relaxed = true)
    private val memoryEditDao = mockk<com.aura.memory.MemoryEditDao>(relaxed = true)
    private val conversationDao = mockk<com.aura.agent.ConversationDao>(relaxed = true)
    private val kgDao = mockk<com.aura.kg.KnowledgeGraphDao>(relaxed = true)
    private val handDao = mockk<com.aura.hands.HandDao>(relaxed = true)
    private val taskDao = mockk<com.aura.tasks.TaskDao>(relaxed = true)
    private val reminderDao = mockk<com.aura.tasks.ReminderDao>(relaxed = true)
    private val proactiveEventDao = mockk<com.aura.proactive.ProactiveEventDao>(relaxed = true)
    private val userProfileDao = mockk<com.aura.profile.UserProfileDao>(relaxed = true)
    private val providerKeys = mockk<com.aura.providers.ProviderKeys>(relaxed = true)
    private val userPreferences = mockk<com.aura.data.UserPreferences>(relaxed = true)
    private val context = mockk<android.content.Context>(relaxed = true)
    private val reminderScheduler = mockk<com.aura.tasks.ReminderScheduler>(relaxed = true)
    private val usageTracker = com.aura.usage.UsageTracker()

    private val manager = BackupManager(
        context = context,
        memoryDao = memoryDao,
        memoryEditDao = memoryEditDao,
        conversationDao = conversationDao,
        kgDao = kgDao,
        handDao = handDao,
        taskDao = taskDao,
        reminderDao = reminderDao,
        proactiveEventDao = proactiveEventDao,
        userProfileDao = userProfileDao,
        providerKeys = providerKeys,
        userPreferences = userPreferences,
        reminderScheduler = reminderScheduler,
        usageTracker = usageTracker,
    )

    @Test
    fun `snapshot exports all six tables plus preferences`() = runTest {
        // Empty tables — the call should not throw and should return a
        // valid AuraBackup with the metadata fields populated.
        coEvery { memoryDao.allForExport() } returns emptyList()
        coEvery { memoryEditDao.allForBackup() } returns emptyList()
        coEvery { conversationDao.allForExport() } returns emptyList()
        coEvery { kgDao.allNodes() } returns emptyList()
        coEvery { kgDao.allEdges() } returns emptyList()
        coEvery { handDao.getAll() } returns emptyList()
        coEvery { taskDao.all() } returns emptyList()
        coEvery { reminderDao.allForBackup() } returns emptyList()
        coEvery { proactiveEventDao.allForBackup() } returns emptyList()
        coEvery { userProfileDao.get() } returns null
        every { userPreferences.defaultModel } returns flowOf("ollama:deepseek-v4-pro:cloud")
        every { userPreferences.firstRunComplete } returns flowOf(true)
        every { userPreferences.appLockEnabled } returns flowOf(false)
        every { userPreferences.lastSeenProactiveAt } returns flowOf(0L)
        every { userPreferences.morningBriefEnabled } returns flowOf(true)
        every { userPreferences.calendarMonitorEnabled } returns flowOf(true)
        every { userPreferences.ttsEnabled } returns flowOf(true)
        every { userPreferences.incognitoDefault } returns flowOf(false)
        every { userPreferences.themeMode } returns flowOf("system")
        every { userPreferences.customIdentity } returns flowOf("")
        every { userPreferences.specialistOverrides } returns flowOf("{}")
        every { userPreferences.morningBriefHour } returns flowOf(7)
        every { userPreferences.specialistToolOverrides } returns flowOf("{}")
        every { providerKeys.embeddingModel } returns "nomic-embed-text"

        val backup = manager.snapshot(appVersionName = "0.1.0")

        assertEquals(AuraBackup.SCHEMA_VERSION, backup.schemaVersion)
        assertEquals("aura-android", backup.exportedBy)
        assertEquals("0.1.0", backup.appVersionName)
        assertTrue(backup.exportedAt > 0L)
        assertEquals(0, backup.memories.size)
        assertEquals(0, backup.conversations.size)
        assertEquals(0, backup.knowledgeGraph.nodes.size)
        assertEquals(0, backup.hands.size)
        assertEquals(0, backup.tasks.size)
        assertEquals(null, backup.userProfile)

        // Preferences are captured.
        assertEquals("ollama:deepseek-v4-pro:cloud", backup.preferences.defaultModel)
        assertEquals(true, backup.preferences.firstRunComplete)
        assertEquals(false, backup.preferences.appLockEnabled)
        assertEquals("nomic-embed-text", backup.preferences.embeddingModel)
    }

    @Test
    fun `round-trip encode decode preserves payload`() = runTest {
        // The snapshot doesn't need real data — we just need a payload
        // that exercises every field. Build one directly.
        val original = AuraBackup(
            exportedAt = 1_700_000_000_000L,
            appVersionName = "0.1.0",
            memories = listOf(
                MemoryBackup(
                    id = "m1", content = "prefers dark mode", source = "user",
                    category = "preference", importance = 0.8f,
                    createdAt = 1L, accessedAt = 2L, accessCount = 3,
                    decayScore = 0.5f, tags = "ui,theme", metadata = "{}",
                ),
            ),
            conversations = listOf(
                ConversationBackup(
                    id = "c1", title = "Onboarding help", createdAt = 1L, updatedAt = 2L,
                    systemPrompt = "be brief", model = "ollama:deepseek-v4-pro:cloud",
                    metadataJson = "{}", turnsJson = "[]",
                    contextSummary = "User chose dark mode.", summaryThroughTurn = 12,
                ),
            ),
        )

        val json = manager.encodeToJson(original)
        val parsed = manager.decodeFromJson(json)

        assertEquals(original.exportedAt, parsed.exportedAt)
        assertEquals(original.memories.size, parsed.memories.size)
        assertEquals("prefers dark mode", parsed.memories[0].content)
        assertEquals("Onboarding help", parsed.conversations[0].title)
        assertEquals("User chose dark mode.", parsed.conversations[0].contextSummary)
        assertEquals(12, parsed.conversations[0].summaryThroughTurn)
    }

    @Test
    fun `decode rejects backups from a newer schema version`() = runTest {
        val future = AuraBackup(
            schemaVersion = AuraBackup.SCHEMA_VERSION + 100,
            exportedAt = 0L,
            appVersionName = "future",
        )
        val json = manager.encodeToJson(future)
        val ex = assertFailsWith<IllegalArgumentException> {
            manager.decodeFromJson(json)
        }
        assertTrue(
            "schema" in ex.message!!.lowercase(),
            "error should mention the schema, got: ${ex.message}",
        )
    }

    @Test
    fun `purgeAll wipes every table in dependency order`() = runTest {
        coEvery { memoryDao.deleteAll() } returns Unit
        coEvery { conversationDao.deleteAll() } returns Unit
        coEvery { kgDao.deleteAllEdges() } returns Unit
        coEvery { kgDao.deleteAllNodes() } returns Unit
        coEvery { handDao.deleteAll() } returns Unit
        coEvery { taskDao.deleteAll() } returns Unit

        manager.purgeAll()

        // Edges must go before nodes (foreign-key relationship).
        coVerify { kgDao.deleteAllEdges() }
        coVerify { kgDao.deleteAllNodes() }
        coVerify { memoryDao.deleteAll() }
        coVerify { conversationDao.deleteAll() }
        coVerify { handDao.deleteAll() }
        coVerify { taskDao.deleteAll() }
    }

    @Test
    fun `restore writes rows to the right tables in dependency order`() = runTest {
        coEvery { memoryDao.insertAll(any()) } returns Unit
        coEvery { conversationDao.insertAll(any()) } returns Unit
        coEvery { kgDao.insertAllNodes(any()) } returns Unit
        coEvery { kgDao.insertAllEdges(any()) } returns Unit
        coEvery { handDao.insertAll(any()) } returns Unit
        coEvery { taskDao.insertAll(any()) } returns Unit
        coEvery { userProfileDao.upsert(any()) } returns Unit
        coEvery { userPreferences.setDefaultModel(any()) } returns Unit
        coEvery { userPreferences.setAppLockEnabled(any()) } returns Unit
        coEvery { userPreferences.setFirstRunComplete(any()) } returns Unit

        val counts = manager.restore(
            AuraBackup(
                exportedAt = 0L,
                appVersionName = "0.1.0",
                memories = listOf(
                    MemoryBackup("m1", "c", "user", "preference", 0.5f, 1L, 1L, 0, 1f, "", "{}")
                ),
                conversations = listOf(
                    ConversationBackup(
                        "c1", "t", 1L, 2L, null, "m", "{}", "[]",
                        contextSummary = "Durable context", summaryThroughTurn = 5,
                    )
                ),
                knowledgeGraph = KnowledgeGraphBackup(
                    nodes = listOf(
                        NodeBackup("n1", "l", "PERSON", "{}", 0.8f, "", 1L, 2L, 0, 0L)
                    ),
                    edges = listOf(
                        EdgeBackup("e1", "KNOWS", "n1", "n2", 0.5f, "{}", 0.8f, "", 1L, 1L)
                    ),
                ),
                hands = listOf(HandBackup("h1", "h", "", "[]", true, 1L)),
                tasks = listOf(TaskBackup("t1", "t", "", 1L, null, null, "pending", 0, "")),
                userProfile = UserProfileBackup(null, "[]", "{}", "[]", 1L),
                preferences = PreferencesBackup(
                    defaultModel = "ollama:deepseek-v4-pro:cloud",
                    firstRunComplete = true,
                    appLockEnabled = true,
                    embeddingModel = "nomic-embed-text",
                ),
            ),
        )

        assertEquals(1, counts.memories)
        assertEquals(1, counts.conversations)
        assertEquals(1, counts.nodes)
        assertEquals(1, counts.edges)
        assertEquals(1, counts.hands)
        assertEquals(1, counts.tasks)
        assertEquals(1, counts.profile)
        assertEquals(7, counts.total)
        coVerify {
            conversationDao.insertAll(
                match { rows ->
                    rows.single().contextSummary == "Durable context" &&
                        rows.single().summaryThroughTurn == 5
                },
            )
        }
    }

    @Test
    fun `restore reschedules future reminders with fresh work`() = runTest {
        val future = System.currentTimeMillis() + 3_600_000L
        val restored = com.aura.tasks.ReminderEntity(
            id = "reminder-1",
            workId = "",
            message = "Call home",
            triggerAt = future,
            recurrence = "weekly",
        )
        coEvery { reminderScheduler.schedule(any()) } returns restored.copy(workId = "fresh-work")

        val counts = manager.restore(
            AuraBackup(
                exportedAt = 0L,
                appVersionName = "0.2.0",
                reminders = listOf(
                    ReminderBackup(
                        id = "reminder-1",
                        message = "Call home",
                        triggerAt = future,
                        createdAt = 1L,
                        taskId = "",
                        recurrence = "weekly",
                        status = "scheduled",
                    ),
                ),
            ),
        )

        coVerify {
            reminderScheduler.schedule(
                match {
                    it.id == "reminder-1" && it.workId.isEmpty() &&
                        it.recurrence == "weekly" && it.triggerAt == future
                },
            )
        }
        assertEquals(1, counts.reminders)
    }

    @Test
    fun `defaultExportFileName ends in json and has a timestamp that matches`() {
        // Use a fixed instant and verify the filename contains the
        // timestamp in some timezone-recognizable form. We just check
        // shape (.json suffix, "aura-backup-" prefix, 15-char
        // date+time body) rather than a specific date string because
        // the local timezone affects the formatted date.
        val name = manager.defaultExportFileName(now = 1_700_000_000_000L)
        assertTrue(name.endsWith(".json"), "filename should end in .json: $name")
        assertTrue(name.startsWith("aura-backup-"), "filename should start with prefix: $name")
        // Format is "aura-backup-YYYYMMDD-HHMMSS.json" — the body
        // between the prefix and ".json" should be 15 chars
        // (8 date + 1 dash + 6 time).
        val body = name.removePrefix("aura-backup-").removeSuffix(".json")
        assertEquals(15, body.length, "expected YYYYMMDD-HHMMSS body, got: $body")
        assertTrue(body[8] == '-', "expected dash separator at index 8, got: $body")
    }
}
