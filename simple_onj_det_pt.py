import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

img = cv2.imread('people.jpg')

#results = model.predict(source=img, save=False, conf=0.4, iou=0.5, device='cpu') #conf is confidence threshold, iou is intersection over union for Non-Maximum Suppression
results = model(img)

annoted_image = results[0].plot()

cv2.imshow('Annotated Image', annoted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()