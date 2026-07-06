package com.aura.data

import android.content.Context
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import com.aura.core.BuildConfig

/**
 * Shared Room database builder configuration.
 *
 * Applies:
 * - Schema export (production builds validate migrations).
 * - Explicit migrations.
 * - Debug-only destructive fallback on downgrade so developers don't get blocked
 *   during schema churn; upgrades always require explicit migrations.
 */
object RoomConfig {

    fun <T : RoomDatabase> builder(
        context: Context,
        klass: Class<T>,
        name: String,
        migrations: Array<Migration>,
    ): RoomDatabase.Builder<T> {
        val builder = Room.databaseBuilder(context, klass, name)
            .addMigrations(*migrations)
            .apply {
                if (BuildConfig.DEBUG) {
                    fallbackToDestructiveMigrationOnDowngrade()
                }
            }
        return builder
    }
}
