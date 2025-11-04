# from ultralytics import YOLO
# import cv2
# import math

# # Load model
# model = YOLO("yolo11s.pt")

# # Distance threshold (in pixels)
# DIST_THRESHOLD = 60

# cap = cv2.VideoCapture("/home/hamzah/Desktop/yolo_traffic_analysis/traffic.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)

#     centers = []
#     bboxes = []

#     # STEP 1: Collect bounding boxes and midpoints
#     for result in results:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

#             # Filter for vehicles (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
#             if cls_id not in [2, 3, 5, 7]:
#                 continue

#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             centers.append((cx, cy))
#             bboxes.append((x1, y1, x2, y2))

#     # STEP 2: Determine which bboxes should be red
#     red_flags = [False] * len(bboxes)

#     for i in range(len(centers)):
#         for j in range(i + 1, len(centers)):
#             (x1, y1) = centers[i]
#             (x2, y2) = centers[j]

#             dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#             if dist < DIST_THRESHOLD:
#                 red_flags[i] = True
#                 red_flags[j] = True

#     # STEP 3: Draw boxes, midpoints, and connecting lines
#     for i, ((x1, y1, x2, y2), (cx, cy)) in enumerate(zip(bboxes, centers)):
#         color = (0, 255, 0)  # green
#         if red_flags[i]:
#             color = (0, 0, 255)  # red

#         # bbox
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

#         # midpoint
#         cv2.circle(frame, (cx, cy), 5, color, -1)

#     # draw lines between all midpoints
#     for i in range(len(centers)):
#         for j in range(i + 1, len(centers)):
#             cv2.line(frame, centers[i], centers[j], (255, 255, 255), 1)

#     cv2.imshow("YOLOv8 Vehicle Distance Detection", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------

from ultralytics import YOLO
import cv2
import math
import time

# Load YOLO11 model
model = YOLO("yolo11s.pt")

DIST_THRESHOLD = 60              # pixel distance threshold
MIN_CONF = 0.4                   # ignore low-confidence detections
MIN_AREA = 700                   # ignore tiny detections (area threshold)
SMOOTHING_FACTOR = 0.4           # for bounding box smoothing

cap = cv2.VideoCapture("/home/hamzah/Desktop/yolo_traffic_analysis/traffic.mp4")

prev_bboxes = []                 # store smoothed bounding boxes between frames
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Run YOLO
    results = model(frame)

    centers = []
    bboxes = []
    confidences = []

    # STEP 1: Extract bounding boxes + midpoints
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # vehicle classes only (COCO: car=2, motorcycle=3, bus=5, truck=7)
            if cls_id not in [2, 3, 5, 7]:
                continue

            if conf < MIN_CONF:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # ignore tiny detections
            if (x2 - x1) * (y2 - y1) < MIN_AREA:
                continue

            # midpoint
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers.append((cx, cy))
            bboxes.append([x1, y1, x2, y2])
            confidences.append(conf)

    # STEP 1.5: Smooth bounding boxes
    if len(prev_bboxes) == len(bboxes):
        for i in range(len(bboxes)):
            for j in range(4):
                bboxes[i][j] = (
                    SMOOTHING_FACTOR * bboxes[i][j] +
                    (1 - SMOOTHING_FACTOR) * prev_bboxes[i][j]
                )

    prev_bboxes = [b[:] for b in bboxes]

    # STEP 2: Determine RED boxes (too close)
    red_flags = [False] * len(bboxes)
    close_pairs = []  # store pairs to draw lines

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            (x1, y1) = centers[i]
            (x2, y2) = centers[j]

            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            if dist < DIST_THRESHOLD:
                red_flags[i] = True
                red_flags[j] = True
                close_pairs.append((centers[i], centers[j]))

    # STEP 3: Draw bounding boxes + midpoints
    for i, ((x1, y1, x2, y2), (cx, cy)) in enumerate(zip(bboxes, centers)):
        color = (0, 255, 0)  # GREEN
        if red_flags[i]:
            color = (0, 0, 255)  # RED

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

    # STEP 4: Draw lines ONLY for close pairs
    for pt1, pt2 in close_pairs:
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

    # STEP 5: Draw FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Enhanced YOLO Vehicle Proximity Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
