import cv2
import numpy as np
import os

EBUG = 'DEBUG' in os.environ
wait_time = 30

cap = cv2.VideoCapture(0)

detest_data = [
    {
        "name": "FAUER!1!!",
        "bounds": [
            np.array((53, 55, 147)),
            np.array((83,160, 255))
        ],
        "color": (255, 0, 0),
        "text": (127, 255, 0)
    }

]

while cap.isOpened():
   ret, frame = cap.read()
   if frame is None:
       break

   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   img = frame.copy()

   blurred = cv2.GaussianBlur(hsv, (15, 15), 0)

   kernel = np.ones((45, 45), np.uint8)
   eroded = cv2.erode(blurred, kernel, iterations=1)

   if EBUG:
       cv2.imshow('blurred', blurred)
       cv2.imshow('eroded', eroded)

   for obj in detest_data:
       hsv_min, hsv_max = obj["bounds"]
       thresh = cv2.inRange(hsv, hsv_min, hsv_max)

       srt_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
       joined = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, srt_el)

       if EBUG:
           cv2.imshow('thresh' + obj["name"], thresh)
           cv2.imshow('joined' + obj["name"], joined)

       contours, hierarchy = cv2.findContours(joined.copy(), cv2.RETH_TREE, cv2.CHAIN_APPROX_SIMPLE)

       if contours:
           for cont in contours:
               x, y, w, h = cv2.boundingrect(cont)

               if w < 5 or h < 5:
                   continue

               cv2.rectangle(img, (x, y), (x+w, y+h), obj["color"], 5)
               img = cv2.putText(img, obj["name"], (x + w // 3, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, obj["text"], 2)

           if EBUG:
               print(x, y, w, h)

   cv2.imshow('img', img)

   if cv2.waitKey(wait_time) == 27 or ret == False:
       break

cap.release()
cv2.destroyAllWindows()