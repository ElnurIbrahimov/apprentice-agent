package com.aura.kg

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update

@Dao
interface KnowledgeGraphDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertNode(node: NodeEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEdge(edge: EdgeEntity)

    @Query("SELECT * FROM kg_nodes WHERE id = :id")
    suspend fun getNode(id: String): NodeEntity?

    @Query("SELECT * FROM kg_nodes WHERE label = :label LIMIT 1")
    suspend fun getNodeByLabel(label: String): NodeEntity?

    @Query("""
        SELECT * FROM kg_nodes
        WHERE label LIKE '%' || :query || '%'
           OR type LIKE '%' || :query || '%'
           OR properties LIKE '%' || :query || '%'
        ORDER BY accessCount DESC, updatedAt DESC
        LIMIT :limit
    """)
    suspend fun searchNodes(query: String, limit: Int = 50): List<NodeEntity>

    @Query("SELECT * FROM kg_nodes ORDER BY updatedAt DESC LIMIT :limit")
    suspend fun recentNodes(limit: Int = 50): List<NodeEntity>

    @Query("UPDATE kg_nodes SET accessCount = accessCount + 1, lastAccessed = :now WHERE id = :id")
    suspend fun incrementAccessCount(id: String, now: Long = System.currentTimeMillis())

    @Query("SELECT * FROM kg_edges WHERE sourceId = :id OR targetId = :id")
    suspend fun edgesForNode(id: String): List<EdgeEntity>

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
}
