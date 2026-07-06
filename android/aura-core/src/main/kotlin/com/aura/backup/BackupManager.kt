package com.aura.backup

import android.content.Context
import com.aura.agent.ConversationDao
import com.aura.agent.ConversationEntity
import com.aura.data.UserPreferences
import com.aura.hands.Hand
import com.aura.hands.HandDao
import com.aura.kg.KnowledgeGraphDao
import com.aura.kg.EdgeEntity
import com.aura.kg.NodeEntity
import com.aura.memory.MemoryDao
import com.aura.memory.MemoryEntity
import com.aura.profile.UserProfile
import com.aura.profile.UserProfileDao
import com.aura.profile.UserProfileEntity
import com.aura.providers.ProviderKeys
import com.aura.tasks.TaskDao
import com.aura.tasks.TaskEntity
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Reads the full local state into an [AuraBackup] for export, and
 * writes an [AuraBackup] back into Room for import.
 *
 * Both paths are O(total_rows) over the data they touch. For a
 * personal-use install (hundreds of memories, dozens of conversations,
 * thousands of KG edges) the export completes in well under a
 * second; the import is similar.
 *
 * Embeddings are intentionally NOT exported — they are model-specific
 * (384-dim for nomic-embed-text) and would be meaningless on a
 * different device with a different model. After import, the user
 * triggers Settings → Memory → Rebuild embeddings to regenerate.
 *
 * API keys are also NOT exported — they live in [com.aura.security.SecureDataStore]
 * (encrypted with the Android Keystore) and the user has to re-paste
 * them on a fresh install. Including them in plaintext in a backup
 * file would be a security regression.
 */
@Singleton
class BackupManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val memoryDao: MemoryDao,
    private val conversationDao: ConversationDao,
    private val kgDao: KnowledgeGraphDao,
    private val handDao: HandDao,
    private val taskDao: TaskDao,
    private val userProfileDao: UserProfileDao,
    private val providerKeys: ProviderKeys,
    private val userPreferences: UserPreferences,
) {
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    /**
     * Snapshot every persisted table into an [AuraBackup]. Caller is
     * responsible for serializing (e.g. via [encodeToJson]) and
     * writing to disk / sharing via Intent.
     *
     * WARNING: The exported JSON is plaintext. It contains all
     * conversations, memories, tasks, profile facts, and preferences
     * (but NOT API keys or embeddings). Treat the file as sensitive.
     */
    suspend fun snapshot(
        appVersionName: String,
    ): AuraBackup = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        AuraBackup(
            exportedAt = now,
            appVersionName = appVersionName,
            memories = memoryDao.allForExport().map { it.toBackup() },
            conversations = conversationDao.allForExport().map { it.toBackup() },
            knowledgeGraph = KnowledgeGraphBackup(
                nodes = kgDao.allNodes().map { it.toBackup() },
                edges = kgDao.allEdges().map { it.toBackup() },
            ),
            hands = handDao.getAll().map { it.toBackup() },
            tasks = taskDao.all().map { it.toBackup() },
            userProfile = userProfileDao.get()?.toBackup(),
            preferences = PreferencesBackup(
                defaultModel = userPreferences.defaultModel.first().takeIf { it.isNotBlank() },
                firstRunComplete = userPreferences.firstRunComplete.first(),
                appLockEnabled = userPreferences.appLockEnabled.first(),
                embeddingModel = providerKeys.embeddingModel.takeIf { it.isNotBlank() },
            ),
        )
    }

    /**
     * Serialize [backup] to JSON. Public so the Settings UI can show
     * a preview or write the bytes to a content URI via
     * `ContentResolver.openOutputStream`.
     */
    fun encodeToJson(backup: AuraBackup): String = json.encodeToString(backup)

    /**
     * Parse [bytes] into an [AuraBackup]. Throws if the JSON is
     * unparseable or the schema version is newer than this build.
     */
    fun decodeFromJson(bytes: String): AuraBackup {
        val parsed = json.decodeFromString<AuraBackup>(bytes)
        require(parsed.schemaVersion <= AuraBackup.SCHEMA_VERSION) {
            "Backup schema version ${parsed.schemaVersion} is newer than " +
                "this build (${AuraBackup.SCHEMA_VERSION}). Upgrade Aura first."
        }
        return parsed
    }

    /**
     * Write [backup] into every persisted table. The order matters:
     * KG nodes must be inserted before edges (foreign-key relationship).
     * Memories and conversations are independent of each other and
     * of KG.
     *
     * This is a destructive import — it does NOT clear tables first.
     * The OnConflictStrategy.REPLACE means re-importing the same
     * file is idempotent, but importing a smaller file after a
     * larger one will leave stale rows. The caller is expected to
     * call [purgeAll] first if a clean-slate restore is intended.
     *
     * @return the count of rows written per table.
     */
    suspend fun restore(backup: AuraBackup): RestoreCounts = withContext(Dispatchers.IO) {
        val memRows = backup.memories.map { it.toEntity() }
        val convRows = backup.conversations.map { it.toEntity() }
        val nodeRows = backup.knowledgeGraph.nodes.map { it.toEntity() }
        val edgeRows = backup.knowledgeGraph.edges.map { it.toEntity() }
        val handRows = backup.hands.map { it.toEntity() }
        val taskRows = backup.tasks.map { it.toEntity() }
        val profileRow = backup.userProfile?.toEntity()

        if (memRows.isNotEmpty()) memoryDao.insertAll(memRows)
        if (convRows.isNotEmpty()) conversationDao.insertAll(convRows)
        if (nodeRows.isNotEmpty()) kgDao.insertAllNodes(nodeRows)
        if (edgeRows.isNotEmpty()) kgDao.insertAllEdges(edgeRows)
        if (handRows.isNotEmpty()) handDao.insertAll(handRows)
        if (taskRows.isNotEmpty()) taskDao.insertAll(taskRows)
        profileRow?.let { userProfileDao.upsert(it) }
        backup.preferences.defaultModel?.let { userPreferences.setDefaultModel(it) }
        if (backup.preferences.appLockEnabled) userPreferences.setAppLockEnabled(true)
        if (backup.preferences.firstRunComplete) userPreferences.setFirstRunComplete(true)
        backup.preferences.embeddingModel?.let { providerKeys.setEmbeddingModel(it) }

        RestoreCounts(
            memories = memRows.size,
            conversations = convRows.size,
            nodes = nodeRows.size,
            edges = edgeRows.size,
            hands = handRows.size,
            tasks = taskRows.size,
            profile = if (profileRow != null) 1 else 0,
        )
    }

    /**
     * Drop everything. Used by the "Restore from backup" flow when
     * the user explicitly wants a clean slate before re-importing.
     * Each table is wiped by id; cascade relationships (KG edges
     * for a node) are torn down by deleting all edges first.
     */
    suspend fun purgeAll() = withContext(Dispatchers.IO) {
        // Conversations, memories, hands, tasks are independent
        // tables — we use a single bulk DELETE via a one-off
        // Query to avoid N round-trips.
        memoryDao.deleteAll()
        conversationDao.deleteAll()
        kgDao.deleteAllEdges()
        kgDao.deleteAllNodes()
        handDao.deleteAll()
        taskDao.deleteAll()
    }

    /**
     * Default export filename: `aura-backup-YYYYMMDD-HHMMSS.json`.
     * Caller uses this when constructing the share Intent.
     */
    fun defaultExportFileName(now: Long = System.currentTimeMillis()): String {
        val date = java.text.SimpleDateFormat("yyyyMMdd-HHmmss", java.util.Locale.US).format(java.util.Date(now))
        return "aura-backup-$date.json"
    }

    /** Default cache dir for the export file before sharing. */
    fun exportFile(): File = File(context.cacheDir, defaultExportFileName())

    data class RestoreCounts(
        val memories: Int,
        val conversations: Int,
        val nodes: Int,
        val edges: Int,
        val hands: Int,
        val tasks: Int,
        val profile: Int,
    ) {
        val total: Int get() = memories + conversations + nodes + edges + hands + tasks + profile
    }
}

// ── Mappers: Entity → Backup ──

private fun MemoryEntity.toBackup() = MemoryBackup(
    id = id,
    content = content,
    source = source,
    category = category,
    importance = importance,
    createdAt = createdAt,
    accessedAt = accessedAt,
    accessCount = accessCount,
    decayScore = decayScore,
    tags = tags,
    metadata = metadata,
)

private fun MemoryBackup.toEntity() = MemoryEntity(
    id = id,
    content = content,
    source = source,
    category = category,
    importance = importance,
    // Embedding left null — caller rebuilds via Settings → Rebuild.
    embedding = null,
    createdAt = createdAt,
    accessedAt = accessedAt,
    accessCount = accessCount,
    decayScore = decayScore,
    tags = tags,
    metadata = metadata,
)

private fun ConversationEntity.toBackup() = ConversationBackup(
    id = id,
    title = title,
    createdAt = createdAt,
    updatedAt = updatedAt,
    systemPrompt = systemPrompt,
    model = model,
    metadataJson = metadataJson,
    turnsJson = turnsJson,
)

private fun ConversationBackup.toEntity() = ConversationEntity(
    id = id,
    title = title,
    createdAt = createdAt,
    updatedAt = updatedAt,
    systemPrompt = systemPrompt,
    model = model,
    metadataJson = metadataJson,
    turnsJson = turnsJson,
)

private fun NodeEntity.toBackup() = NodeBackup(
    id = id,
    label = label,
    type = type,
    properties = properties,
    confidence = confidence,
    sourceTurnId = sourceTurnId,
    createdAt = createdAt,
    updatedAt = updatedAt,
    accessCount = accessCount,
    lastAccessed = lastAccessed,
)

private fun NodeBackup.toEntity() = NodeEntity(
    id = id,
    label = label,
    type = type,
    properties = properties,
    confidence = confidence,
    sourceTurnId = sourceTurnId,
    createdAt = createdAt,
    updatedAt = updatedAt,
    accessCount = accessCount,
    lastAccessed = lastAccessed,
)

private fun EdgeEntity.toBackup() = EdgeBackup(
    id = id,
    type = type,
    sourceId = sourceId,
    targetId = targetId,
    weight = weight,
    properties = properties,
    confidence = confidence,
    sourceTurnId = sourceTurnId,
    createdAt = createdAt,
    lastReinforced = lastReinforced,
)

private fun EdgeBackup.toEntity() = EdgeEntity(
    id = id,
    type = type,
    sourceId = sourceId,
    targetId = targetId,
    weight = weight,
    properties = properties,
    confidence = confidence,
    sourceTurnId = sourceTurnId,
    createdAt = createdAt,
    lastReinforced = lastReinforced,
)

private fun Hand.toBackup() = HandBackup(
    id = id, name = name, triggerPhrase = triggerPhrase, steps = steps,
    enabled = enabled, createdAt = createdAt,
)

private fun HandBackup.toEntity() = Hand(
    id = id, name = name, triggerPhrase = triggerPhrase, steps = steps,
    enabled = enabled, createdAt = createdAt,
)

private fun TaskEntity.toBackup() = TaskBackup(
    id = id, title = title, description = description,
    createdAt = createdAt, dueAt = dueAt, completedAt = completedAt,
    status = status, priority = priority, tags = tags,
)

private fun TaskBackup.toEntity() = TaskEntity(
    id = id, title = title, description = description,
    createdAt = createdAt, dueAt = dueAt, completedAt = completedAt,
    status = status, priority = priority, tags = tags,
)

private fun UserProfileEntity.toBackup() = UserProfileBackup(
    name = name,
    traitsJson = traitsJson,
    preferencesJson = preferencesJson,
    factsJson = factsJson,
    lastUpdated = lastUpdated,
)

private fun UserProfileBackup.toEntity() = UserProfileEntity(
    id = 1,
    name = name,
    traitsJson = traitsJson,
    preferencesJson = preferencesJson,
    factsJson = factsJson,
    lastUpdated = lastUpdated,
)
