from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model

# Export the model
model.export(format="onnx", device="cpu")