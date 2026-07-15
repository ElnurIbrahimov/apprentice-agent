package com.aura.security

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.hardware.display.DisplayManager
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.Looper
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.FragmentActivity
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.lang.ref.WeakReference
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Holder that manages a screen-capture [MediaProjection] session for the
 * [CaptureScreenTool]. The activity registers the launcher and stores the
 * permission result here; the tool then requests capture and reads the
 * resulting bitmap.
 *
 * Mirrors the pattern used by [BiometricActivityHolder].
 */
@Singleton
class ScreenCaptureHolder @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private var activityRef: WeakReference<FragmentActivity>? = null
    private var permissionResult: Intent? = null
    private var permissionResultCode: Int = 0
    private var projection: MediaProjection? = null

    val activity: FragmentActivity?
        get() = activityRef?.get()

    fun attach(activity: FragmentActivity, launcher: ActivityResultLauncher<Intent>) {
        activityRef = WeakReference(activity)
        this.launcher = launcher
    }

    fun detach() {
        activityRef = null
        projection?.stop()
        projection = null
    }

    private var launcher: ActivityResultLauncher<Intent>? = null

    private val _pendingResult = MutableStateFlow<Boolean?>(null)
    val pendingResult: StateFlow<Boolean?> = _pendingResult

    fun requestPermission() {
        val activity = activityRef?.get() ?: return
        val manager = activity.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as? MediaProjectionManager
            ?: return
        launcher?.launch(manager.createScreenCaptureIntent())
    }

    fun onPermissionResult(resultCode: Int, data: Intent?) {
        permissionResultCode = resultCode
        permissionResult = data
        _pendingResult.value = resultCode == android.app.Activity.RESULT_OK && data != null
    }

    fun capture(width: Int = 1080, height: Int = 1920): Bitmap? {
        val activity = activityRef?.get() ?: return null
        val resultCode = permissionResultCode
        val data = permissionResult ?: return null
        val manager = activity.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as? MediaProjectionManager
            ?: return null

        projection?.stop()
        projection = manager.getMediaProjection(resultCode, data) ?: return null

        val density = activity.resources.displayMetrics.densityDpi
        val reader = ImageReader.newInstance(width, height, android.graphics.PixelFormat.RGBA_8888, 2)
        val virtualDisplay = projection?.createVirtualDisplay(
            "aura_capture",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            Handler(Looper.getMainLooper()),
        ) ?: return null

        val image = reader.acquireLatestImage() ?: return null
        val planes = image.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * width
        val bitmap = Bitmap.createBitmap(
            width + rowPadding / pixelStride,
            height,
            Bitmap.Config.ARGB_8888,
        )
        bitmap.copyPixelsFromBuffer(buffer)
        image.close()
        reader.close()
        virtualDisplay.release()

        return Bitmap.createBitmap(bitmap, 0, 0, width, height)
    }
}
