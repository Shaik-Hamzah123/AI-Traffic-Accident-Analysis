import cv2
import math
import supervision as sv
from inference import get_model
from PIL import Image

# Load RF-DETR model
model = get_model("rfdetr-base")

# Distance threshold in pixels
DIST_THRESHOLD = 120

# Open video
cap = cv2.VideoCapture("/home/hamzah/Desktop/yolo_traffic_analysis/traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Inference
    predictions = model.infer(pil_image, confidence=0.5)[0]
    detections = sv.Detections.from_inference(predictions)

    bboxes = []
    centers = []
    classes = []

    # STEP 1 — extract vehicle bboxes + midpoints
    for pred in predictions.predictions:
        cls_name = pred.class_name.lower()

        if cls_name not in ["car", "truck", "bus", "motorcycle"]:
            continue

        x1, y1, x2, y2 = pred.bbox

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        bboxes.append((x1, y1, x2, y2))
        centers.append((cx, cy))
        classes.append(cls_name)

    # STEP 2 — mark red if distance < threshold
    red_flags = [False] * len(bboxes)

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            (x1, y1) = centers[i]
            (x2, y2) = centers[j]

            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            if dist < DIST_THRESHOLD:
                red_flags[i] = True
                red_flags[j] = True

    # STEP 3 — draw bboxes and midpoints
    annotated = frame.copy()

    for i, ((x1, y1, x2, y2), (cx, cy)) in enumerate(zip(bboxes, centers)):

        color = (0, 255, 0)         # green
        if red_flags[i]:
            color = (0, 0, 255)     # red

        # draw bbox
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # draw midpoint
        cv2.circle(annotated, (cx, cy), 6, color, -1)

    # STEP 4 — connect all midpoints with lines
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            cv2.line(annotated, centers[i], centers[j], (255, 255, 255), 1)

    cv2.imshow("RF-DETR Proximity Detector", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
