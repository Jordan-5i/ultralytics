import torch
from PIL import Image
import cv2
import onnxruntime

from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms
from ultralytics.utils import YAML

### --------ultralytics推理api------  ####
# Load the exported ONNX model
onnx_model = YOLO("yolo11n-cls.onnx")

# Run inference
results = onnx_model("cat.jpg")
### --------------------------------  ####


def preprocess(img):
    """Convert input images to model-compatible tensor format with appropriate normalization."""
    
    transforms = classify_transforms()
    
    if not isinstance(img, torch.Tensor):
        img = torch.stack(
            [transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
        )
    img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img))
    return img.float() 


if __name__ == "__main__":

    names = YAML.load("/data/wangjian/project/ultralytics/ultralytics/cfg/datasets/ImageNet.yaml").get("names")

    session = onnxruntime.InferenceSession("yolo11n-cls.onnx")

    im = cv2.imread("cat.jpg")
    im = preprocess([im]).numpy()
    y = session.run(None, {session.get_inputs()[0].name: im})[0][0]

    top1 = int(y.argmax())
    top5 = (-y).argsort(0)[:5].tolist()

    top1conf =y[top1]
    top5conf = y[top5]

    for t1, t2 in zip(top5, top5conf):
        print(f"{names[t1]:<20} {t2:.4f}")