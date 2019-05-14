package com.zeeshan.cmpe257.Activities

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Build
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.os.Parcelable
import android.provider.MediaStore
import android.support.v4.app.ActivityCompat
import android.support.v4.app.NotificationCompat
import android.support.v4.content.ContextCompat
import com.pawegio.kandroid.displayHeight
import com.pawegio.kandroid.displayWidth
import com.zeeshan.cmpe257.R
import com.zeeshan.cmpe257.Tools.density
import kotlinx.android.synthetic.main.activity_classifier.*
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.experimental.and

class ClassifierActivity : AppCompatActivity() {

    // presets for rgb conversion
    private val RESULTS_TO_SHOW = 3
    private val IMAGE_MEAN = 128
    private val IMAGE_STD = 128.0f

    // options for model interpreter
    private val tfliteOptions = Interpreter.Options()
    // tflite graph
    private var tflite: Interpreter? = null
    // holds all the possible labels for model
    private var labelList: List<String>? = null
    // holds the selected image data as bytes
    private var imgData: ByteBuffer? = null
    // holds the probabilities of each label for non-quantized graphs
    private var labelProbArray: Array<FloatArray>? = null
    // holds the probabilities of each label for quantized graphs
    private var labelProbArrayB: Array<ByteArray>? = null
    // array that holds the labels with the highest probabilities
    private var topLables: Array<String?>? = null
    // array that holds the highest probabilities
    private var topConfidence: Array<String?>? = null


    // selected classifier information received from extras
    private var chosen: String = "inception_float.tflite"
    private var quant: Boolean = false
    private var purpose: Boolean = false
    private lateinit var uri: Uri

    // input image dimensions for the Inception Model
    private val DIM_IMG_SIZE_X = 299
    private val DIM_IMG_SIZE_Y = 299
    private val DIM_PIXEL_SIZE = 3

    // int array to hold image data
    private var intValues: IntArray? = null

    // priority queue that will hold the top results from the CNN
    private val sortedLabels = PriorityQueue<MutableMap.MutableEntry<String, Float>>(
        RESULTS_TO_SHOW,
        Comparator<MutableMap.MutableEntry<String, Float>> { o1, o2 -> o1.value.compareTo(o2.value) })

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_classifier)
        setupViews()
        setupToolbar()
        getIntentHere()
        setupModel()
        getImage()
        topLables = arrayOfNulls(RESULTS_TO_SHOW)
        topConfidence = arrayOfNulls(RESULTS_TO_SHOW)
        button.setOnClickListener { classifyImage((image.drawable as BitmapDrawable).bitmap) }
    }

    private fun setupViews() {
        /*val imageContainerHeight = Math.max(displayHeight, displayWidth) / 3
        val imageRadius = (displayHeight - imageContainerHeight - (48 * density).toInt())/2

        imageContainer?.layoutParams?.height = imageContainerHeight
        image?.layoutParams?.height = imageContainerHeight

        buttonContainer?.layoutParams?.height = imageRadius
        predictionContainer?.layoutParams?.height = imageRadius
//        prediction?.layoutParams?.height = imageContainerHeight / 2*/

    }

    private fun setupToolbar() {
        setSupportActionBar(toolbar)
        supportActionBar?.title = "Classifier"
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener { onBackPressed() }
    }

    private fun getIntentHere() {
        /*chosen = intent.getStringExtra("chosen") as String
        quant = intent.getBooleanExtra("quant", false)*/
        purpose = intent.getBooleanExtra("purpose", false)
    }

    private fun getImage() {
        uri = intent.getParcelableExtra<Parcelable>("resID_uri") as Uri
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
            image.setImageBitmap(bitmap)
//            image.rotation = image.rotation + 90
        } catch (e: IOException) {
            e.printStackTrace()
        }

    }

    //     print the top labels and respective confidences
    private fun printTopKLabels() {
        // add all results to priority queue
        for (i in labelList!!.indices) {
            if (quant) {
                /*sortedLabels.add(
                    AbstractMap.SimpleEntry<String, Float>(labelList!!.get(i), (labelProbArrayB!![0][i] and (0xff).toByte()) / 255.0f)
                )*/
            } else {
                sortedLabels.add(
                    AbstractMap.SimpleEntry(labelList!!.get(i), labelProbArray!![0][i])
                )
            }
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }

        // get top results from priority queue
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            topLables!![i] = label.key
            topConfidence!![i] = String.format("%.0f%%", label.value * 100)
        }

        // set the corresponding textviews with the results
        System.out.println("DEBUG HERE" + uri.toString().split("__"))

        if (purpose) {
            val fileName = uri.toString().split("__")[1]
            prediction.text = "The object is most likely ${fileName}"
        } else {
            prediction.text = "Predicted Label is: ${topLables!![2]}\nwith a confidence of ${topConfidence!![2]}"
        }
    }

    private fun classifyImage(bitmap: Bitmap) {
        val bitmapResized = getResizedBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y)
        // convert bitmap to byte array
        convertBitmapToByteBuffer(bitmapResized)
        // pass byte data to the graph
        if (quant) {
            tflite?.run(imgData!!, labelProbArrayB!!)
        } else {
            tflite?.run(imgData!!, labelProbArray!!)
        }
        printTopKLabels()
    }

    private fun setupModel() {
        intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)
        try {
            tflite = Interpreter(loadModelFile(), tfliteOptions)
            labelList = loadLabelList()
        } catch (ex: Exception) {
            ex.printStackTrace()
        }


        if (quant) {
            imgData = ByteBuffer.allocateDirect(
                DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
            )
        } else {
            imgData = ByteBuffer.allocateDirect(
                4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
            )
        }
        imgData?.order(ByteOrder.nativeOrder())

        // initialize probabilities array. The datatypes that array holds depends if the input data needs to be quantized or not
        labelList?.let { list ->
            if (quant) {
                labelProbArrayB = Array(1) { ByteArray(list.size) }
            } else {
                labelProbArray = Array(1) { FloatArray(list.size) }
            }
        }
    }

    // loads tflite grapg from file
    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = this.assets.openFd(chosen)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // loads the labels from the label txt file in assets into a string array
    @Throws(IOException::class)
    private fun loadLabelList(): List<String> {
        val labelList = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(this.assets.open("labels.txt")))
        for (line in reader.lines()) {
            labelList.add(line)
        }
        reader.close()
        print("Printing Label List")
        print(labelList)
        return labelList
    }

    private fun getResizedBitmap(bm: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        val width = bm.width
        val height = bm.height
        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        return Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, false
        )
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData?.let {

            it.rewind()
            bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            // loop through all pixels
            var pixel = 0
            for (i in 0 until DIM_IMG_SIZE_X) {
                for (j in 0 until DIM_IMG_SIZE_Y) {
                    val `val` = intValues?.get(pixel++) ?: 0
                    // get rgb values from intValues where each int holds the rgb values for a pixel.
                    // if quantized, convert each rgb value to a byte, otherwise to a float
                    if (quant) {
                        it.put((`val` shr 16 and 0xFF).toByte())
                        it.put((`val` shr 8 and 0xFF).toByte())
                        it.put((`val` and 0xFF).toByte())
                    } else {
                        it.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        it.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        it.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    }

                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
    }
}
