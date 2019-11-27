package com.nutritious.tensorflow

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.min


class Classifier(assetManager: AssetManager, modelPath: String, labelPath: String, private val inputSize: Int) {
    private var interpreter: Interpreter
    private var labelList: List<String>
    private val pixelSize: Int = 3
    private val maxResult = 3
    private val threshHold = 0.4f
    private val IMAGE_MEAN = 128
    private val IMAGE_STD  = 128.0f

    data class Recognition(
            var id: String = "",
            var title: String = "",
            var confidence: Float = 0F
    ) {
        override fun toString(): String {
            return "Title = $title, Confidence = $confidence)"
        }
    }

    init {
        Log.e("CLASSIFIER", "INIT")
        val tfliteOptions = Interpreter.Options()
        tfliteOptions.setNumThreads(5)
        tfliteOptions.setUseNNAPI(true)
        interpreter = Interpreter(loadModelFile(assetManager, modelPath),tfliteOptions)
        labelList = loadLabelList(assetManager, labelPath)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        Log.e("CLASSIFIER", "loadModelFile %s".format(modelPath))
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        return assetManager.open(labelPath).bufferedReader().useLines { it.toList() }

    }

    fun recognizeImage(bitmap: Bitmap): List<Recognition> {
        Log.e("CLASSIFIER", "recognizeImage")
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) { FloatArray(labelList.size) }
        Log.e("CLASSIFIER", "recognizeImage interpreter %d".format(result[0].size))
        interpreter.run(byteBuffer, result)
        Log.e("CLASSIFIER", "recognizeImage return" )
        return getSortedResult(result)
    }


    private fun addPixelValue(byteBuffer: ByteBuffer, intValue: Int): ByteBuffer {
        Log.e("CLASSIFIER", "addPixelValue")
        byteBuffer.putFloat(((intValue.shr(16) and 0xFF)-IMAGE_MEAN)/IMAGE_STD)
        byteBuffer.putFloat(((intValue.shr(8) and 0xFF)-IMAGE_MEAN)/IMAGE_STD)
        byteBuffer.putFloat(((intValue and 0xFF)-IMAGE_MEAN)/IMAGE_STD)
        return byteBuffer
    }

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        Log.e("CLASSIFIER", "convertBitmapToByteBuffer")
        val imgData = ByteBuffer.allocateDirect(4*inputSize * inputSize * pixelSize)
        imgData.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)


        imgData.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // Convert the image to floating point.
        var pixel = 0
        SystemClock.uptimeMillis()
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                addPixelValue(imgData, value)
            }
        }
        return imgData
    }


    private fun getSortedResult(labelProbArray: Array<FloatArray>): List<Recognition> {
        Log.e("CLASSIFIER", "getSortedResult List Size:(%d, %d, %d)".format(labelProbArray.size, labelProbArray[0].size, labelList.size))

        val pq = PriorityQueue(
                maxResult,
                Comparator<Recognition> { (_, _, confidence1), (_, _, confidence2)
                    ->
                    confidence1.compareTo(confidence2) * -1
                })

        for (i in labelList.indices) {
            val confidence = labelProbArray[0][i]
            Log.e("CLASSIFIER confidence value:", "" + confidence)
            if (confidence >= threshHold) {
                Log.e(" CLASSIFIER confidence value:", "" + confidence)
                pq.add(Recognition("" + i,
                        if (labelList.size > i) labelList[i] else "Unknown",
                        confidence
                ))
            }
        }
        Log.e("CLASSIFIER", "pqsize:(%d)".format(pq.size))

        val recognitions = ArrayList<Recognition>()
        val recognitionsSize = min(pq.size, maxResult)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }
        return recognitions
    }

}