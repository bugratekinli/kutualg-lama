import cv2
import numpy as np
from ultralytics import YOLO
import math
import cvzone
import time
from sort import *

cap = cv2.VideoCapture(0)

model = YOLO("AKDGRT.pt")

tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.1)


while True:
    success, img = cap.read()

    result = model(img, stream=True)

    detections = np.empty((0,5))

    for r in result:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            if conf > 0.5:
                cvzone.putTextRect(img, f'{cls}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        # print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=2, rt=1, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=0.6, thickness=1, offset=5, colorR=(0, 255, 0))


    cv2.imshow("image", img)
    cv2.waitKey(1)
