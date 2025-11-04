from ultralytics import YOLO
import cv2
import math

# Load YOLOv8 model
model = YOLO("yolov8s.pt")   # use your preferred model

# Threshold distance (in pixels)
DIST_THRESHOLD = 60

# Open video
cap = cv2.VideoCapture("/home/hamzah/Desktop/yolo_traffic_analysis/traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    centers = {}   # store id : (cx, cy)
    bboxes = {}    # store id : (x1, y1, x2, y2)

    # STEP 1: Collect bounding boxes + midpoints
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()

        for i, box in enumerate(xyxy):
            obj_id = int(ids[i])
            cls_id = int(cls[i])

            # Only track cars (COCO classes: car=2, truck=7, bus=5, motorcycle=3)
            if cls_id not in [2,3,5,7]:
                continue

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers[obj_id] = (cx, cy)
            bboxes[obj_id] = (x1, y1, x2, y2)

    # STEP 2: Check distances and mark objects that should become RED
    red_ids = set()

    obj_ids = list(centers.keys())

    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            id1 = obj_ids[i]
            id2 = obj_ids[j]

            (x1, y1) = centers[id1]
            (x2, y2) = centers[id2]

            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # If close
            if dist < DIST_THRESHOLD:
                red_ids.add(id1)
                red_ids.add(id2)

    # STEP 3: Draw bounding boxes + midpoints + lines
    for obj_id, (x1, y1, x2, y2) in bboxes.items():
        cx, cy = centers[obj_id]

        # Default color = green
        color = (0, 255, 0)

        # If object too close → red
        if obj_id in red_ids:
            color = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw midpoint
        cv2.circle(frame, (cx, cy), 5, color, -1)

    # Draw lines between all midpoints
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            id1 = obj_ids[i]
            id2 = obj_ids[j]
            (x1, y1) = centers[id1]
            (x2, y2) = centers[id2]

            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.imshow("Car Tracking with Distance Alerts", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
