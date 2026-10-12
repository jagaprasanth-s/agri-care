import numpy as np

from src.predict import parse_class
from src.preprocess import check_quality, preprocess, quality_stats


def test_parse_class():
    assert parse_class("Tomato___Early_blight") == ("Tomato", "Early blight")
    assert parse_class("Corn_(maize)___Common_rust_") == ("Maize", "Common rust")
    assert parse_class("Pepper,_bell___Bacterial_spot") == ("Pepper", "Bacterial spot")
    assert parse_class("Grape___healthy") == ("Grape", "Healthy")


def test_preprocess_shape_and_range():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    batch = preprocess(img)
    assert batch.shape == (1, 224, 224, 3)
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_quality_rejects_dark_and_flat_images():
    dark = np.zeros((224, 224, 3), dtype=np.uint8)
    assert check_quality(quality_stats(dark)) == "The photo is too dark."
    flat = np.full((224, 224, 3), 120, dtype=np.uint8)
    assert check_quality(quality_stats(flat)) == "The photo looks blurry."
    noisy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    assert check_quality(quality_stats(noisy)) is None
