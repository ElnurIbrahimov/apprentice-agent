package com.aura.ui.screens.chat

internal fun shouldDetachFromLiveEdge(
    isUserDragging: Boolean,
    physicallyAtLiveEdge: Boolean,
): Boolean = isUserDragging && !physicallyAtLiveEdge

internal fun shouldShowJumpToLatest(
    turnCount: Int,
    followLiveEdge: Boolean,
): Boolean = turnCount > 0 && !followLiveEdge

internal fun shouldAutoFollow(
    turnCount: Int,
    followLiveEdge: Boolean,
): Boolean = turnCount > 0 && followLiveEdge
