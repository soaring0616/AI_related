from ultralytics import YOLO 

model = YOLO("yolov8s.pt")
results = model.train(data=r"my-yolo.yaml", epochs=20, imgsz=640) # Train the model