package com.aura.ui.util

import android.content.Context
import android.os.Build
import android.view.HapticFeedbackConstants
import android.view.View

/**
 * Tactile feedback for chat events. Haptics make the app feel responsive
 * in a way that visual feedback alone can't — sending a message, receiving
 * a response, errors, and "thinking" moments all get their own pattern.
 *
 * Uses the view's performHapticFeedback which respects the user's
 * system-level haptic setting (so if they turned it off, ours stays off too).
 */
object Haptics {
    fun tap(view: View) = perform(view, HapticFeedbackConstants.VIRTUAL_KEY)

    fun send(view: View) = perform(view, HapticFeedbackConstants.KEYBOARD_TAP)

    fun receive(view: View) = perform(view, HapticFeedbackConstants.CONTEXT_CLICK)

    fun error(view: View) = perform(view, HapticFeedbackConstants.LONG_PRESS)

    fun success(view: View) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            perform(view, HapticFeedbackConstants.CONFIRM)
        } else {
            perform(view, HapticFeedbackConstants.VIRTUAL_KEY)
        }
    }

    private fun perform(view: View, code: Int) {
        view.performHapticFeedback(code)
    }
}