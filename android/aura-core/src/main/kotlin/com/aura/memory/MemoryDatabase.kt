package com.aura.memory

import androidx.room.Database
import androidx.room.RoomDatabase
import com.aura.creative.CreativeProjectDao
import com.aura.creative.CreativeProjectEntity
import com.aura.creative.CreativeArtifactDao
import com.aura.creative.CreativeArtifactEntity
import com.aura.creative.CreativeRevisionDao
import com.aura.creative.CreativeRevisionEntity
import com.aura.creative.CreativeBranchDao
import com.aura.creative.CreativeBranchEntity
import com.aura.creative.CreativeGenerationJobDao
import com.aura.creative.CreativeGenerationJobEntity
import com.aura.creative.CanonFactDao
import com.aura.creative.CanonFactEntity
import com.aura.creative.CreativeSimulationDao
import com.aura.creative.CreativeSimulationEntity
import com.aura.creative.ContinuityIssueDao
import com.aura.creative.ContinuityIssueEntity
import com.aura.creative.ArtifactDependencyDao
import com.aura.creative.ArtifactDependencyEntity
import com.aura.world.BeliefDao
import com.aura.world.BeliefEntity
import com.aura.world.EvidenceDao
import com.aura.world.EvidenceEntity
import com.aura.world.WorldEventDao
import com.aura.world.WorldEventEntity
import com.aura.world.OpportunityDao
import com.aura.world.OpportunityEntity
import com.aura.documents.DocumentDao
import com.aura.documents.DocumentEntity
import com.aura.documents.DocumentChunkDao
import com.aura.documents.DocumentChunkEntity
import com.aura.kg.EdgeEntity
import com.aura.kg.KnowledgeGraphDao
import com.aura.kg.NodeEntity

@Database(
    entities = [
        MemoryEntity::class,
        NodeEntity::class,
        EdgeEntity::class,
        MemoryEditEntity::class,
        DocumentEntity::class,
        CreativeProjectEntity::class,
        DocumentChunkEntity::class,
        CreativeArtifactEntity::class,
        CreativeRevisionEntity::class,
        CreativeBranchEntity::class,
        CreativeGenerationJobEntity::class,
        CanonFactEntity::class,
        CreativeSimulationEntity::class,
        ContinuityIssueEntity::class,
        ArtifactDependencyEntity::class,
        BeliefEntity::class,
        EvidenceEntity::class,
        WorldEventEntity::class,
        OpportunityEntity::class,
    ],
    version = 10,
    exportSchema = true,
)
abstract class MemoryDatabase : RoomDatabase() {
    abstract fun memoryDao(): MemoryDao
    abstract fun knowledgeGraphDao(): KnowledgeGraphDao
    abstract fun memoryEditDao(): MemoryEditDao
    abstract fun documentDao(): DocumentDao
    abstract fun creativeProjectDao(): CreativeProjectDao
    abstract fun documentChunkDao(): DocumentChunkDao
    abstract fun creativeArtifactDao(): CreativeArtifactDao
    abstract fun creativeRevisionDao(): CreativeRevisionDao
    abstract fun creativeBranchDao(): CreativeBranchDao
    abstract fun creativeGenerationJobDao(): CreativeGenerationJobDao
    abstract fun canonFactDao(): CanonFactDao
    abstract fun creativeSimulationDao(): CreativeSimulationDao
    abstract fun continuityIssueDao(): ContinuityIssueDao
    abstract fun artifactDependencyDao(): ArtifactDependencyDao
    abstract fun beliefDao(): BeliefDao
    abstract fun evidenceDao(): EvidenceDao
    abstract fun worldEventDao(): WorldEventDao
    abstract fun opportunityDao(): OpportunityDao
}
