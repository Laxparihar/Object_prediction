import streamlit
from ultralytics import YOLO
import cv2
model = YOLO("yolov8s.pt")
# model("avengers.jpg", show=True)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
while True:
    flag, frame = cap.read()
    results = model(frame, stream=True, show=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
           x1,y1,x2,y2 = box.xyxy.numpy()[0]
           cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),3)
           cv2.putText(frame,result.names[int(box.cls)],(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("obj",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()