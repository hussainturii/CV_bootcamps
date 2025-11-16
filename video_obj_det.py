import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("people.mp4")

unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #annotating people only
    results = model.predict(source=frame, save=False, conf=0.4, iou=0.5, device='cpu', classes=[0], verbose=False) #class 0 is for person
    #counting people
    results1 = model.track(source=frame, save=False, conf=0.4, iou=0.5, device='cpu', classes=[0], persist=True, verbose = False) #class 0 is for person, persist to keep the IDs across frames
    
    annoted_frame = results1[0].plot()
    #annoted_frame = results[0].plot()

    if results1[0].boxes and results1[0].boxes.id is not None:
        ids = results1[0].boxes.id.numpy()
        for id in ids:
            unique_ids.add(int(id))
        cv2.putText(frame, f'Unique People Count: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Object Detection', frame)

    print(f'Number of unique people detected so far: {len(unique_ids)}')
    cv2.imshow('Video Object Detection', annoted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()