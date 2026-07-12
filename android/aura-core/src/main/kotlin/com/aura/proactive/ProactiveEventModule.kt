package com.aura.proactive

import android.content.Context
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.aura.data.RoomConfig
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ProactiveEventModule {

    val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            // Migration 1→2: adds the `payload` column to `proactive_events`
            // for structured event routing data. Existing rows get the default
            // empty string, matching the entity's `payload: String = ""` default.
            db.execSQL(
                "ALTER TABLE proactive_events ADD COLUMN payload TEXT NOT NULL DEFAULT ''",
            )
        }
    }

    /**
     * Migration 2→3: adds an index on `timestamp` so the `recent(100)` and
     * `countSince()` queries on the Home screen badge are O(log n) instead
     * of a full table scan. Without this, the badge cost grows linearly
     * with the number of accumulated events.
     */
    val MIGRATION_2_3 = object : Migration(2, 3) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("CREATE INDEX IF NOT EXISTS index_proactive_events_timestamp ON proactive_events(timestamp)")
        }
    }

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): ProactiveEventDatabase =
        RoomConfig.builder(
            context,
            ProactiveEventDatabase::class.java,
            "aura-proactive.db",
            migrations = arrayOf(MIGRATION_1_2, MIGRATION_2_3),
        ).build()

    @Provides
    fun provideProactiveEventDao(db: ProactiveEventDatabase): ProactiveEventDao = db.proactiveEventDao()
}
