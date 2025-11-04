Traffic Risk Analysis using YOLO & Hotspot Clustering

A complete traffic-risk detection system implemented in a single Python
file.
It detects vehicles using YOLO, estimates collision-risk using distance
between midpoints, logs traffic data, and generates visual analytics
including hotspot clusters using DBSCAN.

All generated plots are saved inside the outputs/ directory.

------------------------------------------------------------------------

✅ Features

🚗 Vehicle Detection (YOLO11s)

-   Uses Ultralytics YOLO (yolo11s.pt)
-   Filters vehicle classes only (car, truck, bus, motorcycle)
-   Frame resizing for faster inference
-   Skipping frames to improve FPS
-   Confidence + area threshold filters to reduce noise

------------------------------------------------------------------------

⚠️ Accident Risk Detection

-   Computes midpoint for every vehicle’s bounding box
-   If distance < DIST_THRESHOLD → risky interaction
-   Risky vehicles highlighted in red
-   Safe vehicles highlighted in green
-   Risk pairs connected using white lines
-   Risky midpoints logged to hotspot_points.csv

------------------------------------------------------------------------

🌀 Bounding Box Smoothing

-   Nearest neighbor matching of detections between frames
-   Exponential smoothing (SMOOTHING_ALPHA)
-   Drastically reduces bbox jitter and flicker

------------------------------------------------------------------------

📝 Automatic Logging (CSV)

The script outputs:

  -----------------------------------------------------------------------
  File                         Purpose
  ---------------------------- ------------------------------------------
  traffic_log.csv              Frame-wise vehicle count, risk count,
                               timestamp, FPS

  hotspot_points.csv           Midpoints of risky detections for hotspot
                               clustering
  -----------------------------------------------------------------------

Both are appended — so multiple runs accumulate data.

------------------------------------------------------------------------

✅ Auto-Generated Plots (outputs/ Folder)

After processing the video, the script automatically saves:

  File                   Description
  ---------------------- -----------------------------------
  traffic_vs_time.png    Vehicle count trend
  risk_vs_time.png       Accidental-risk count over time
  hotspot_clusters.png   DBSCAN-based accident hotspot map

All PNGs appear inside outputs/.

These files are generated after the video is processed.

------------------------------------------------------------------------

✅ Project Flow

1.  Read video frame
2.  Resize frame (optional)
3.  YOLO detection
4.  Filter vehicles
5.  Smooth bounding boxes
6.  Find close midpoint pairs (risk)
7.  Save logs + hotspot points
8.  Draw visualization on frame
9.  After video ends → generate plots + hotspot clustering

------------------------------------------------------------------------

✅ Requirements

Install dependencies:

    pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn

✅ How to Run

1.  Place your video file anywhere
2.  Update the path inside the script:

    VIDEO_PATH = "/path/to/video.mp4"

3.  Run the script:

    python traffic_analysis.py

4.  After processing:
    -   Check CSV logs in the project folder
    -   Check generated PNG plots inside outputs/

------------------------------------------------------------------------

✅ Customizable Parameters

-   DIST_THRESHOLD — Risk distance threshold
-   MIN_CONF — YOLO confidence threshold
-   RESIZE_WIDTH — Lower width = higher FPS
-   PROCESS_EVERY_N — Skip frames (increase FPS)
-   DBSCAN_EPS, DBSCAN_MIN_SAMPLES — Hotspot clustering strength

------------------------------------------------------------------------

✅ Future Enhancements

-   Improved vehicle detection with stronger models (RT-DETR / YOLO11m)
-   Better risk estimation using speed + direction
-   Heatmap overlays for accident-prone areas
-   Perspective correction for more accurate distances
-   Real-time web dashboard (FastAPI + WebSockets)
-   Geographic mapping of hotspots

------------------------------------------------------------------------

✅ License

Open-source for learning, research, and experimental use.
