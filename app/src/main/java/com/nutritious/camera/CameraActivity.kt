package com.nutritious.camera

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.nutritious.R


class CameraActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_camera)
        savedInstanceState ?: supportFragmentManager.beginTransaction()
            .replace(R.id.container, Camera2BasicFragment.newInstance())
            .commit()
    }
}
