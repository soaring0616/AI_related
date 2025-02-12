import os
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import json
from AreaEntity import AreaEntity

### 載入模型 ###
custom_model = YOLO(r"D:\yolo\e_fence\runs\detect\train14\weights\best.pt")  # build a new model from YAML
custom_model.to(torch.device("cuda"))
detection_threshold = 0.4 #模型參數

### 影片相關設定 ###
video_path = r"demo_data\output.mp4"
cap = cv2.VideoCapture(video_path) # 創建 VideoCapture 物件

# 取得影像的屬性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) 

# 寫入影像的設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')          # 設定影片的格式為 MJPG
out = cv2.VideoWriter('detection_demo_outputa.mp4', fourcc, 15, (frame_width,  frame_height))  # 產生空的影片

# cv2.namedWindow('demo', cv2.WINDOW_NORMAL) #開啟檢視影片的視窗

### 載入禁區座標 (label studio 是以百分比匯出) ###
with open(r"D:\yolo\e_fence\demo_data\demo\danger_area_coord.json")as f:
    danger_zone = json.load(f)
    
danger1 = AreaEntity(danger_zone[2]['annotations'][0]['result'][0]['value']['points'])
danger2 = AreaEntity(danger_zone[2]['annotations'][0]['result'][1]['value']['points'])

## 依照原始影片大小轉換座標
danger1_2d = danger1.contour.squeeze()
actual_danger1 = danger1_2d.copy()
actual_danger1[:, 0] = (danger1_2d[:, 0] / 100.0) * frame_width
actual_danger1[:, 1] = (danger1_2d[:, 1] / 100.0) * frame_height

danger2_2d = danger2.contour.squeeze()
actual_danger2 = danger2_2d.copy()
actual_danger2[:, 0] = (danger2_2d[:, 0] / 100.0) * frame_width
actual_danger2[:, 1] = (danger2_2d[:, 1] / 100.0) * frame_height

### 依照影片中每幀影像來作標記 ###
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = custom_model.predict(frame, conf=detection_threshold, show_labels=True, classes=0, imgsz=1200)        
        
    situation = "Normal" # scope 為全局：預設是狀況正常

    cv2.polylines(frame, [actual_danger1], True, (0, 255, 0), 3)
    cv2.polylines(frame, [actual_danger2], True, (0, 255, 0), 3)
    
    for item in results[0].boxes.xyxy:
        
        color = (255, 0, 0) #物件狀況正常的顏色
        
        item = item.cpu().numpy()
        cx = int((item[0]+item[2])//2)
        cy = int((item[1]+item[3])//2)
        
        dangerstate_c1 = cv2.pointPolygonTest(actual_danger1, (cx, cy), False)
        dangerstate_c2 = cv2.pointPolygonTest(actual_danger2, (cx, cy), False)

        if dangerstate_c1>0 or dangerstate_c2>0 :
            color =  (0, 0, 255)
            situation = "WARNING!"
 
        # cv2.putText(frame, f'dangerstate_c1:{dangerstate_c1}, dangerstate_c2:{dangerstate_c2}', tuple((0, int(item[1]))), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
        cv2.circle(frame, (cx, cy), 2, color ,0)
        cv2.rectangle(frame, tuple(np.int16(item[:2]).tolist()) , tuple(np.int16(item[-2:]).tolist()), color, 2)

    
    cv2.rectangle(frame, (0,0), (1000,150), (0,0,0), -1) # 標示出異常情況
    info_color =  (0, 0, 255) if situation == 'WARNING!' else  (255, 0, 0)
    cv2.putText(frame, f'Current situation:{situation}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, info_color, 4)


    out.write(frame)
    cv2.imshow('demo', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'): # 按下 'q' 鍵退出
        break

# 釋放所有資源
cap.release()
out.release()
cv2.destroyAllWindows()

