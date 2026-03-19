from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model
model = YOLO("/data/hhl_suda/ybzh/lzh/Facial-Expression-Recognition/runs/classify/archive/weights/best.pt")  # load a custom model

# Predict with the model
results = model("/data/hhl_suda/ybzh/lzh/Facial-Expression-Recognition/datasets/archive/test/fear/PrivateTest_166793.jpg")  # predict on an image