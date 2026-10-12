"""Leaf classifier wrapper. Uses the trained Keras model when present, otherwise a clearly flagged simulation."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "plant_model.keras"
CLASSES_PATH = ROOT / "model" / "class_names.json"

CROP_ALIASES = {"Pepper,_bell": "Pepper", "Corn_(maize)": "Maize"}


def parse_class(name: str):
    """Turn a PlantVillage-style folder name into (crop, disease).

    'Tomato___Early_blight' -> ('Tomato', 'Early blight')
    'Corn_(maize)___Common_rust_' -> ('Maize', 'Common rust')
    """
    crop_raw, _, disease_raw = name.partition("___")
    crop = CROP_ALIASES.get(crop_raw, crop_raw.replace("_", " ").strip())
    disease = disease_raw.replace("_", " ").strip()
    disease = "Healthy" if disease.lower() == "healthy" else disease.capitalize()
    return crop, disease


class Classifier:
    def __init__(self, kb):
        self.kb = kb
        self.model = None
        self.classes = []
        try:
            if MODEL_PATH.exists() and CLASSES_PATH.exists():
                import tensorflow as tf  # imported lazily so the app also runs without TensorFlow

                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # TensorFlow missing or model file unreadable
            print(f"[AgriCare] Model not loaded, using demo mode: {exc}")
            self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, batch):
        """Return (crop, disease, confidence 0-1, top3 list, simulated flag)."""
        if self.loaded:
            probs = self.model.predict(batch, verbose=0)[0]
            order = probs.argsort()[::-1][:3]
            top3 = [(*parse_class(self.classes[i]), float(probs[i])) for i in order]
            crop, disease, conf = top3[0]
            return crop, disease, conf, top3, False
        return self._simulate()

    def _simulate(self):
        """Demo only: random result. Never present this as a real prediction."""
        pool = [(e["crop"], e["disease"]) for e in self.kb.entries]
        pool += [("Tomato", "Healthy")] * 2
        crop, disease = random.choice(pool)
        conf = random.uniform(0.6, 0.97)
        return crop, disease, conf, [(crop, disease, conf)], True
