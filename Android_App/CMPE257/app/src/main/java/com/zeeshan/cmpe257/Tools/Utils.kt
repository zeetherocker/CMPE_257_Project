package com.zeeshan.cmpe257.Tools

import android.content.Context
import android.content.res.Resources
import android.widget.Toast


fun getDimension(context: Context?, id: Int): Int = context?.resources?.getDimensionPixelSize(id) ?: 0

val density: Float = Resources.getSystem()?.displayMetrics?.density ?: 0f

fun showToast(message: String, context: Context, length: Int = Toast.LENGTH_SHORT) {
    try {
        Toast.makeText(context, message, length).show()
    } catch (e: Exception) {}
}