package com.zeeshan.cmpe257.Activities

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.support.v4.widget.TextViewCompat
import android.support.v7.app.AppCompatActivity
import android.util.TypedValue
import android.widget.Button
import android.widget.Toast
import com.github.dhaval2404.imagepicker.ImagePicker
import com.pawegio.kandroid.IntentFor
import com.pawegio.kandroid.displayHeight
import com.pawegio.kandroid.displayWidth
import com.soundcloud.android.crop.Crop
import com.zeeshan.cmpe257.R
import com.zeeshan.cmpe257.Tools.density
import com.zeeshan.cmpe257.Tools.getDimension
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.main_content_card0.*
import kotlinx.android.synthetic.main.main_content_card1.*
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        val ICON_HEIGHT_FACTOR = 0.6f
    }

    val REQUEST_PERMISSION = 300

    val REQUEST_GALLERY = 654
    val REQUEST_CAMERA = 450

    private var imageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setupPermissions()
        setupAppIcon()
        setupCallBacks()
        init()

    }

    private fun setupAppIcon() {
        var totalToolbarHeight = getDimension(this, R.dimen.toolbar_plus_statusbar)
        val appIconRadius = (Math.min(displayHeight, displayWidth) * ICON_HEIGHT_FACTOR)
        val appIconHeight = appIconRadius.times(1.5f)
        val serviceCardOffset = (totalToolbarHeight + appIconHeight).toInt()

        appIcon.layoutParams.height = serviceCardOffset

        /* Title View */
        appName.text = "Machine Learning"
        val textHeight = appIconRadius.times(0.275f).toInt()
        appName.layoutParams.height = textHeight
        appName.typeface = Typeface.createFromAsset(assets, "fonts/knight_brush.otf")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            appName.setAutoSizeTextTypeUniformWithConfiguration(1, textHeight, 1, TypedValue.COMPLEX_UNIT_PX)
        } else {
//            TextViewCompat.setAutoSizeTextTypeUniformWithConfiguration(appName, appIconRadius.div(density).toInt(), displayWidth.times(0.9f).div(
//                density
//            ).toInt(), 1, TypedValue.COMPLEX_UNIT_PX)
        }

        appIcon.startAnim()
    }

    private fun init() {
    }

    private fun setupCallBacks() {
        card0.setOnClickListener { openGallery() }
        card1.setOnClickListener { openCamera() }
    }

    private fun openCamera() {
        ImagePicker.with(this)
            .cameraOnly()
            .cropSquare()
            .start(REQUEST_CAMERA)
    }

    private fun openGallery() {
        ImagePicker.with(this)
            .galleryOnly()
//            .cropSquare()
            .start(REQUEST_GALLERY)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK) {
            val imgUri = data?.data
            val intent = IntentFor<ClassifierActivity>(this)
            intent.putExtra("resID_uri", imgUri)
            intent.putExtra("purpose", requestCode == REQUEST_GALLERY)
            startActivity(intent)
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Task Cancelled", Toast.LENGTH_SHORT).show()
        }
    }


    private fun setupPermissions() {

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_PERMISSION)
        }

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_PERMISSION
            )
        }

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                REQUEST_PERMISSION
            )
        }
    }

}
