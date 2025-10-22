from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model

# or load from a yaml file e.g. "yolov8n-seg.yaml"
# model = YOLO("yolov8m-seg.yaml")

# Export the model
model.export(format="onnx", device="cpu")