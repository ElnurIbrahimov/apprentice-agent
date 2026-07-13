package com.aura.kg

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Update

@Dao
interface KnowledgeGraphDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertNode(node: NodeEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEdge(edge: EdgeEntity)

    @Update
    suspend fun updateNode(node: NodeEntity)

    @Query("SELECT * FROM kg_nodes WHERE id = :id")
    suspend fun getNode(id: String): NodeEntity?

    @Query("SELECT * FROM kg_nodes WHERE label = :label LIMIT 1")
    suspend fun getNodeByLabel(label: String): NodeEntity?

    @Query("""
        SELECT * FROM kg_nodes
        WHERE label LIKE '%' || :queryEscaped || '%' ESCAPE '\'
           OR type LIKE '%' || :queryEscaped || '%' ESCAPE '\'
           OR properties LIKE '%' || :queryEscaped || '%' ESCAPE '\'
        ORDER BY accessCount DESC, updatedAt DESC
        LIMIT :limit
    """)
    suspend fun searchNodes(queryEscaped: String, limit: Int = 50): List<NodeEntity>

    @Query("SELECT * FROM kg_nodes ORDER BY updatedAt DESC LIMIT :limit")
    suspend fun recentNodes(limit: Int = 50): List<NodeEntity>

    /**
     * KG nodes whose updatedAt is at or after [sinceMs], newest
     * first. Bounded by [limit]. Used by the morning brief to
     * surface "facts learned in the last 24h."
     */
    @Query("SELECT * FROM kg_nodes WHERE updatedAt >= :sinceMs ORDER BY updatedAt DESC LIMIT :limit")
    suspend fun recentNodesSince(sinceMs: Long, limit: Int): List<NodeEntity>

    @Query("UPDATE kg_nodes SET accessCount = accessCount + 1, lastAccessed = :now WHERE id = :id")
    suspend fun incrementAccessCount(id: String, now: Long = System.currentTimeMillis())

    @Query("SELECT * FROM kg_edges WHERE sourceId = :id OR targetId = :id")
    suspend fun edgesForNode(id: String): List<EdgeEntity>

    /** Alias with management-domain wording. */
    suspend fun neighbors(id: String): List<EdgeEntity> = edgesForNode(id)

    @Query("SELECT * FROM kg_edges WHERE sourceId = :sourceId")
    suspend fun edgesFrom(sourceId: String): List<EdgeEntity>

    @Query("SELECT * FROM kg_edges WHERE targetId = :targetId")
    suspend fun edgesTo(targetId: String): List<EdgeEntity>

    @Query("DELETE FROM kg_edges WHERE sourceId = :id OR targetId = :id")
    suspend fun deleteEdgesForNode(id: String)

    @Query("DELETE FROM kg_nodes WHERE id = :id")
    suspend fun deleteNode(id: String)

    @Query("SELECT COUNT(*) FROM kg_nodes")
    suspend fun nodeCount(): Int

    @Query("SELECT COUNT(*) FROM kg_edges")
    suspend fun edgeCount(): Int

    @Query("SELECT * FROM kg_nodes")
    suspend fun allNodes(): List<NodeEntity>

    @Query("SELECT * FROM kg_edges")
    suspend fun allEdges(): List<EdgeEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAllNodes(rows: List<NodeEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAllEdges(rows: List<EdgeEntity>)

    /**
     * Atomically publish the merged target and its rewritten relations before
     * deleting the source. Deleting the source cascades its old incident edges.
     */
    @Transaction
    suspend fun mergeNodeRecords(
        sourceId: String,
        target: NodeEntity,
        rewrittenEdges: List<EdgeEntity>,
    ) {
        updateNode(target)
        if (rewrittenEdges.isNotEmpty()) insertAllEdges(rewrittenEdges)
        deleteNode(sourceId)
    }

    @Query("DELETE FROM kg_edges")
    suspend fun deleteAllEdges()

    @Query("DELETE FROM kg_nodes")
    suspend fun deleteAllNodes()
}
