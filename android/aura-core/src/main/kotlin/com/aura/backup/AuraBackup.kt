package com.aura.backup

import com.aura.usage.UsageSnapshot
import kotlinx.serialization.Serializable

/**
 * Top-level shape of an Aura data export. The file is JSON-encoded with
 * kotlinx.serialization. Schema is versioned: a future Aura build that
 * reads a [SCHEMA_VERSION] it doesn't recognize should refuse (or
 * warn) rather than silently dropping fields.
 *
 * The export deliberately omits API keys, OAuth tokens, and the
 * secure DataStore. Those live in [com.aura.security.SecureDataStore]
 * (encrypted with the Android Keystore) and would be useless on a
 * different device anyway — the user has to re-paste keys after a
 * fresh install. Including them would also create a security risk
 * (the JSON file would have plaintext keys sitting in a backup).
 *
 * Embeddings are also omitted. They are model-specific, take ~1.5 KB
 * per row, and are rebuilt on the next embedding pass. The
 * `memoryRebuildEmbeddings` action in Settings handles the rebuild
 * after a restore.
 */
@Serializable
data class AuraBackup(
    val schemaVersion: Int = SCHEMA_VERSION,
    val exportedAt: Long,
    val exportedBy: String = "aura-android",
    val appVersionName: String,
    val memories: List<MemoryBackup> = emptyList(),
    val memoryEdits: List<MemoryEditBackup> = emptyList(),
    val conversations: List<ConversationBackup> = emptyList(),
    val knowledgeGraph: KnowledgeGraphBackup = KnowledgeGraphBackup(),
    val hands: List<HandBackup> = emptyList(),
    val handRuns: List<HandRunBackup> = emptyList(),
    val tasks: List<TaskBackup> = emptyList(),
    val reminders: List<ReminderBackup> = emptyList(),
    val proactiveEvents: List<ProactiveEventBackup> = emptyList(),
    val userProfile: UserProfileBackup? = null,
    val preferences: PreferencesBackup = PreferencesBackup(),
    val usage: UsageSnapshot = UsageSnapshot(),
) {
    companion object {
        const val SCHEMA_VERSION = 3
    }
}

@Serializable
data class MemoryBackup(
    val id: String,
    val content: String,
    val source: String,
    val category: String,
    val importance: Float,
    val createdAt: Long,
    val accessedAt: Long,
    val accessCount: Int,
    val decayScore: Float,
    val tags: String,
    val metadata: String,
)

@Serializable
data class MemoryEditBackup(
    val id: Long,
    val memoryId: String,
    val oldContent: String,
    val newContent: String,
    val oldCategory: String,
    val newCategory: String,
    val editedAt: Long,
    val editedBy: String,
)

@Serializable
data class ConversationBackup(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val systemPrompt: String?,
    val model: String?,
    val metadataJson: String,
    val turnsJson: String,
    val contextSummary: String = "",
    val summaryThroughTurn: Int = 0,
)

@Serializable
data class KnowledgeGraphBackup(
    val nodes: List<NodeBackup> = emptyList(),
    val edges: List<EdgeBackup> = emptyList(),
)

@Serializable
data class NodeBackup(
    val id: String,
    val label: String,
    val type: String,
    val properties: String,
    val confidence: Float,
    val sourceTurnId: String,
    val createdAt: Long,
    val updatedAt: Long,
    val accessCount: Int,
    val lastAccessed: Long,
)

@Serializable
data class EdgeBackup(
    val id: String,
    val type: String,
    val sourceId: String,
    val targetId: String,
    val weight: Float,
    val properties: String,
    val confidence: Float,
    val sourceTurnId: String,
    val createdAt: Long,
    val lastReinforced: Long,
)

@Serializable
data class HandBackup(
    val id: String,
    val name: String,
    val triggerPhrase: String,
    val steps: String,
    val enabled: Boolean,
    val createdAt: Long,
    val variables: String = "{}",
    val conditions: String = "[]",
    val scheduleType: String = "none",
    val scheduleHour: Int = 9,
    val scheduleMinute: Int = 0,
    val scheduleDayOfWeek: Int = 1,
    val updatedAt: Long = createdAt,
)

@Serializable
data class HandRunBackup(
    val id: String,
    val handId: String,
    val handName: String,
    val trigger: String,
    val status: String,
    val startedAt: Long,
    val finishedAt: Long? = null,
    val output: String = "",
    val failedStep: Int? = null,
    val variablesJson: String = "{}",
)

@Serializable
data class TaskBackup(
    val id: String,
    val title: String,
    val description: String,
    val createdAt: Long,
    val dueAt: Long? = null,
    val completedAt: Long? = null,
    val status: String,
    val priority: Int,
    val tags: String,
)

@Serializable
data class ReminderBackup(
    val id: String,
    val message: String,
    val triggerAt: Long,
    val createdAt: Long,
    val taskId: String,
    val recurrence: String,
    val status: String,
    val firedAt: Long? = null,
)

@Serializable
data class ProactiveEventBackup(
    val id: Long,
    val eventType: String,
    val title: String,
    val body: String,
    val timestamp: Long,
    val payload: String,
)

@Serializable
data class UserProfileBackup(
    val name: String?,
    val traitsJson: String,
    val preferencesJson: String,
    val factsJson: String,
    val lastUpdated: Long,
)

@Serializable
data class PreferencesBackup(
    val defaultModel: String? = null,
    val firstRunComplete: Boolean = false,
    val appLockEnabled: Boolean = false,
    val embeddingModel: String? = null,
    val lastSeenProactiveAt: Long = 0L,
    val morningBriefEnabled: Boolean = true,
    val calendarMonitorEnabled: Boolean = true,
    val ttsEnabled: Boolean = true,
    val incognitoDefault: Boolean = false,
    val themeMode: String = "system",
    val customIdentity: String = "",
    val specialistOverrides: String = "{}",
    val morningBriefHour: Int = 7,
    val specialistToolOverrides: String = "{}",
)
