import torch
import cv2
import re
import axengine as onnxruntime
from pathlib import Path
from PIL import Image
from typing import Union, Tuple 

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class YAML:
    """
    YAML utility class for efficient file operations with automatic C-implementation detection.

    This class provides optimized YAML loading and saving operations using PyYAML's fastest available implementation
    (C-based when possible). It implements a singleton pattern with lazy initialization, allowing direct class method
    usage without explicit instantiation. The class handles file path creation, validation, and character encoding
    issues automatically.

    The implementation prioritizes performance through:
        - Automatic C-based loader/dumper selection when available
        - Singleton pattern to reuse the same instance
        - Lazy initialization to defer import costs until needed
        - Fallback mechanisms for handling problematic YAML content

    Attributes:
        _instance: Internal singleton instance storage.
        yaml: Reference to the PyYAML module.
        SafeLoader: Best available YAML loader (CSafeLoader if available).
        SafeDumper: Best available YAML dumper (CSafeDumper if available).

    Examples:
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """Initialize singleton instance on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize with optimal YAML implementation (C-based when available)."""
        import yaml

        self.yaml = yaml
        # Use C-based implementation if available for better performance
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        """
        Save Python object as YAML file.

        Args:
            file (str | Path): Path to save YAML file.
            data (dict | None): Dict or compatible object to save.
            header (str): Optional string to add at file beginning.
        """
        instance = cls._get_instance()
        if data is None:
            data = {}

        # Create parent directories if needed
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects to strings
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)

        # Write YAML file
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        """
        Load YAML file to Python object with robust error handling.

        Args:
            file (str | Path): Path to YAML file.
            append_filename (bool): Whether to add filename to returned dict.

        Returns:
            (dict): Loaded YAML content.
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"

        # Read file content
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()

        # Try loading YAML with fallback for problematic characters
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception:
            # Remove problematic characters and retry
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}

        # Check for accidental user-error None strings (should be 'null' in YAML)
        if "None" in data.values():
            data = {k: None if v == "None" else v for k, v in data.items()}

        if append_filename:
            data["yaml_file"] = str(file)
        return data


# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(
    size: Union[Tuple[int, int], int] = 224,
    mean: Tuple[float, float, float] = DEFAULT_MEAN,
    std: Tuple[float, float, float] = DEFAULT_STD,
    interpolation: str = "BILINEAR",
    crop_fraction: float = None,
):
    """
    Create a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (Tuple[float, float, float]): Mean values for each RGB channel used in normalization.
        std (Tuple[float, float, float]): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Deprecated, will be removed in a future version.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    scale_size = size if isinstance(size, (tuple, list)) and len(size) == 2 else (size, size)

    if crop_fraction:
        raise DeprecationWarning(
            "'crop_fraction' arg of classify_transforms is deprecated, will be removed in a future version."
        )

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl += [T.CenterCrop(size), T.PILToTensor()]
    return T.Compose(tfl)


def preprocess(img):
    """Convert input images to model-compatible tensor format with appropriate normalization."""
    
    transforms = classify_transforms()
    
    if not isinstance(img, torch.Tensor):
        img = torch.stack(
            [transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
        )
    img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)) # NCHW
    return img.permute(0, 2, 3, 1) # NHWC


if __name__ == "__main__":

    names = YAML.load("ImageNet.yaml").get("names")

    session = onnxruntime.InferenceSession("compiled.axmodel")

    im = cv2.imread("cat.jpg")
    im = preprocess([im]).numpy()
    y = session.run(None, {session.get_inputs()[0].name: im})[0][0]

    top1 = int(y.argmax())
    top5 = (-y).argsort(0)[:5].tolist()

    top1conf =y[top1]
    top5conf = y[top5]

    for t1, t2 in zip(top5, top5conf):
        print(f"{names[t1]:<20} {t2:.4f}")