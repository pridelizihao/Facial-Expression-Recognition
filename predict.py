from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model
model = YOLO("best_yolo26x_cls.pt")  # load a custom model

# Predict with the model
results = model("pic/self_catch.jpg")  # predict on an image