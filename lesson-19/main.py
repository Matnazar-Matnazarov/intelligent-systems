# ================== OBJECT DETECTION: OpenCV + MobileNet SSD ==================
import cv2
import numpy as np

# Oldindan o‘qitilgan model va konfiguratsiya (COCO sinflari uchun)
CONFIG_PATH = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS_PATH = "frozen_inference_graph.pb"

# COCO datasetidagi 80 sinf nomlari
classNames = []
with open("coco.names", "r") as f:
    classNames = f.read().strip().split("\n")

# Model yuklash
net = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Tasvir yuklash
img = cv2.imread("test.jpg")   # <-- bu yerga o‘z rasm faylingizni qo‘ying
classIds, confs, bbox = net.detect(img, confThreshold=0.5)

# Natija chizish
if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, f"{classNames[classId-1].upper()} {round(confidence*100,1)}%",
                    (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Obyekt aniqlash", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
