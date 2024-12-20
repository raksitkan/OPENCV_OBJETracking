import cv2
from tracker import * #เรียกใช้งานไฟล์ Tracker

# Create tracker object สร้าง ออฟเจ็คขึ้นมาเพื่อตรวจจับภาพในวีดีโอเพื่อนับจพนวนที่วัตถุผ่าน โดยเรียกใช้ class จากไฟล์ Tracker
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4") # เปิดวีดีโอ

# Object detection from Stable camera จับวัตถุที่เคลื่อนไหว ค่า history คือค่าความแม่นยำ ส่น varThresgold คือ ความแม่มยำของการตรวจจับวัตถุเคลื่อนที่ 
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest แยกพื้นทีถนน ดดยใช้ตำแหน่งประมาณบนหน้าจอ
    roi = frame[340: 720,500: 800] #ความยาว , ขอบล่าง , ขยัยซ้ายขวา ,ความแคบ #ใส่กรอบเฟรมที่เราต้องการซึ่งเป้นการจับเฉพาะที่บนหน้าจอ

    # 1. Object Detection ทำการกำหนดสิ่งที่เป้นรูปทรงทั้งหมดที่อยู่ในเฟรม โดยตีกรอบรูปทรงทั้งหมดให้เป็นสีเขียว คือ 0,255,0
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # ตรวจจับเฉพาะสีขาว ไม่เอาเงาสีเท่า
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements #ลบองคืประกอบที่ไม่จำเช่นสิ่งที่มีขนาดเล้กและไม่เคลือนไหวในเฟรม
        area = cv2.contourArea(cnt)
        if area > 100: 
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key == 0:
        break

cap.release()
cv2.destroyAllWindows()
