import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)   # 0 is usually the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, conf=0.4, iou=0.5, device='cpu')
    annoted_frame = results[0].plot()

    cv2.imshow('Live Camera Object Detection', annoted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()