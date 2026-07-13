package com.aura.hands

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
object HandsModule {

    val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("ALTER TABLE hands ADD COLUMN variables TEXT NOT NULL DEFAULT '{}'")
            db.execSQL("ALTER TABLE hands ADD COLUMN conditions TEXT NOT NULL DEFAULT '[]'")
            db.execSQL("ALTER TABLE hands ADD COLUMN scheduleType TEXT NOT NULL DEFAULT 'none'")
            db.execSQL("ALTER TABLE hands ADD COLUMN scheduleHour INTEGER NOT NULL DEFAULT 9")
            db.execSQL("ALTER TABLE hands ADD COLUMN scheduleMinute INTEGER NOT NULL DEFAULT 0")
            db.execSQL("ALTER TABLE hands ADD COLUMN scheduleDayOfWeek INTEGER NOT NULL DEFAULT 1")
            db.execSQL("ALTER TABLE hands ADD COLUMN updatedAt INTEGER NOT NULL DEFAULT 0")
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS hand_runs (
                    id TEXT NOT NULL PRIMARY KEY,
                    handId TEXT NOT NULL,
                    handName TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    status TEXT NOT NULL,
                    startedAt INTEGER NOT NULL,
                    finishedAt INTEGER,
                    output TEXT NOT NULL,
                    failedStep INTEGER,
                    variablesJson TEXT NOT NULL
                )
                """.trimIndent(),
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS index_hand_runs_handId ON hand_runs(handId)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_hand_runs_startedAt ON hand_runs(startedAt)")
        }
    }

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): HandDatabase =
        RoomConfig.builder(
            context,
            HandDatabase::class.java,
            "aura-hands.db",
            migrations = arrayOf(MIGRATION_1_2),
        ).build()

    @Provides
    fun provideHandDao(db: HandDatabase): HandDao = db.handDao()
}
