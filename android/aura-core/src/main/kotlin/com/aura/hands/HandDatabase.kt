package com.aura.hands

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(entities = [Hand::class, HandRun::class], version = 2, exportSchema = true)
abstract class HandDatabase : RoomDatabase() {
    abstract fun handDao(): HandDao
}
