/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ding.objecttracker;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import com.ding.objecttracker.OverlayView.DrawCallback;
import com.ding.objecttracker.env.BorderedText;
import com.ding.objecttracker.env.ImageUtils;
import com.ding.objecttracker.env.Logger;
import com.ding.objecttracker.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 128;
  private static final int MB_IMAGE_MEAN = 64;
  private static final float MB_IMAGE_STD = 64;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd300_fix.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/label.txt";
//    private static final String TF_OD_API_MODEL_FILE =
//            "file:///android_asset/frozen_inference_graph.pb";
//    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/raccoonlabel.txt";

  private static final int INPUT_SIZE = 300;
  private static final int IMAGE_MEAN = 117;
  private static final float IMAGE_STD = 1;
  private static final String INPUT_NAME = "input";
  private static final String OUTPUT_NAME = "output";
  private static final String TF_CLASSSIFY_PB =
            "file:///android_asset/tensorflow_inception_graph.pb";
  private static final String TF_CLASSSIFY_LABELS_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO, CLASSIFY
  }
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;
  private static final float MINIMUM_CONFIDENCE_CLASSIFY = 0.2f;

  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(480, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
//      detector =
//          TensorFlowYoloDetector.create(
//              getAssets(),
//              YOLO_MODEL_FILE,
//              YOLO_INPUT_SIZE,
//              YOLO_INPUT_NAME,
//              YOLO_OUTPUT_NAMES,
//              YOLO_BLOCK_SIZE);
//      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
//      detector =
//          TensorFlowMultiBoxDetector.create(
//              getAssets(),
//              MB_MODEL_FILE,
//              MB_LOCATION_FILE,
//              MB_IMAGE_MEAN,
//              MB_IMAGE_STD,
//              MB_INPUT_NAME,
//              MB_OUTPUT_LOCATIONS_NAME,
//              MB_OUTPUT_SCORES_NAME);
//      cropSize = MB_INPUT_SIZE;
    } else if (MODE == DetectorMode.TF_OD_API){
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e("Exception initializing classifier!", e);
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    } else if (MODE == DetectorMode.CLASSIFY) {
        try {
            detector = TensorFlowImageClassifier.create(
                    getAssets(),
                    TF_CLASSSIFY_PB,
                    TF_CLASSSIFY_LABELS_FILE,
                    INPUT_SIZE,
                    IMAGE_MEAN,
                    IMAGE_STD,
                    INPUT_NAME,
                    OUTPUT_NAME);
            cropSize = INPUT_SIZE;
        } catch (IOException e) {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
              switch (MODE) {
                  case TF_OD_API:
                      minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                      break;
                  case MULTIBOX:
                      minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                      break;
                  case YOLO:
                      minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                      break;
                  case CLASSIFY:
                      minimumConfidence = MINIMUM_CONFIDENCE_CLASSIFY;
              }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              } else {

                      Log.e(DetectorActivity.class.getSimpleName(),
                              "Object found " + result.getTitle() + ":" + result.getConfidence());

              }
            }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }
}
