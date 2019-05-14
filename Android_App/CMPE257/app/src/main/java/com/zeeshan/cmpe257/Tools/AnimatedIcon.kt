package com.zeeshan.cmpe257.Tools

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import com.pawegio.kandroid.displayHeight
import com.pawegio.kandroid.displayWidth
import com.pawegio.kandroid.runAsync
import com.pawegio.kandroid.runOnUiThread
import com.zeeshan.cmpe257.Activities.MainActivity
import com.zeeshan.cmpe257.R


class AnimatedIcon : View {

    var finished: Boolean = false

    var rotatingFactor = 0f
    var iconBack: Bitmap? = null

    constructor(context: Context?) : this(context, null)
    constructor(context: Context?, attrs: AttributeSet?) : this(context, attrs, 0)
    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(context, attrs, defStyleAttr)

    override fun onFinishInflate() {
        super.onFinishInflate()

        iconBack = BitmapFactory.decodeResource(resources, R.raw.ic_splash)
        var radius = Math.min(context.displayWidth, context.displayHeight).times(MainActivity.ICON_HEIGHT_FACTOR).toInt()
        iconBack?.let { iconBack = Bitmap.createScaledBitmap(it, radius, radius, false) }
    }

    public fun startAnim (){
        runAsync {
            while (true) {
                try {
                    Thread.sleep(9)
                    rotatingFactor = (rotatingFactor + 1) % 360
                    runOnUiThread {
                        invalidate()
                    }
                } catch (e: InterruptedException) {
                    e.printStackTrace()
                }
            }
        }
    }

    @SuppressLint("DrawAllocation")
    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        iconBack?.let {
            canvas?.rotate(rotatingFactor, (context.displayWidth.div(2f)), height.div(2f))
            canvas?.drawBitmap(it, (context.displayWidth.div(2f) - it.width.div(2f)), (height.div(2f) - it.height.div(2f)), null)
        }
    }

}