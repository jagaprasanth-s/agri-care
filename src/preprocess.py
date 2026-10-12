"""Image validation and preprocessing with OpenCV."""
import cv2
import numpy as np

IMG_SIZE = 224
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 235
MIN_SHARPNESS = 15  # variance of the Laplacian on the 224x224 grayscale image


class ImageProblem(Exception):
    """Raised when an uploaded file cannot be used."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR image."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageProblem("The file is not a readable image. Use a JPG or PNG leaf photo.")
    return img


def quality_stats(img: np.ndarray) -> dict:
    """Brightness and sharpness of the image after resizing."""
    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "bright": int(round(float(gray.mean()))),
        "sharp": int(round(float(cv2.Laplacian(gray, cv2.CV_64F).var()))),
    }


def check_quality(stats: dict):
    """Return a reason string if the image is unusable, otherwise None."""
    if stats["bright"] < MIN_BRIGHTNESS:
        return "The photo is too dark."
    if stats["bright"] > MAX_BRIGHTNESS:
        return "The photo is too bright."
    if stats["sharp"] < MIN_SHARPNESS:
        return "The photo looks blurry."
    return None


def preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to 224x224, convert BGR to RGB and normalize to 0-1.

    Returns a batch of one image with shape (1, 224, 224, 3).
    """
    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    arr = rgb.astype("float32") / 255.0
    return np.expand_dims(arr, 0)
