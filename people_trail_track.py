import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("people.mp4")

trackers = {} 
next_id = 1

trail_length = 50
trails = defaultdict(lambda: deque(maxlen=trail_length)) 
appear = defaultdict(int)   # count appearances per track id

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # keep tracker IDs across frames
    results = model.track(source=frame, save=False, conf=0.4, iou=0.5, device='cpu',
                          classes=[0], verbose=False, persist=True)
    annotated_frame = frame.copy()

    res = results[0]
    if len(res.boxes) > 0 and hasattr(res.boxes, "id") and res.boxes.id is not None:
        try:
            boxes_np = res.boxes.xyxy.cpu().numpy()
            ids_np = res.boxes.id.cpu().numpy()
        except Exception:
            boxes_np = res.boxes.xyxy.numpy()
            ids_np = res.boxes.id.numpy()

        for box, tid in zip(boxes_np, ids_np):
            track_id = int(float(tid))            # convert to plain int to avoid KeyError
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[track_id] += 1

            if appear[track_id] >= 5 and track_id not in trackers:
                trackers[track_id] = next_id
                next_id += 1

            trails[track_id].append((cx, cy))

            display_id = trackers.get(track_id, track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'ID: {display_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1)

            # draw trail
            pts = list(trails[track_id])
            for i in range(1, len(pts)):
                cv2.line(annotated_frame, pts[i - 1], pts[i], (0, 0, 255), 2)

    cv2.imshow('People Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()