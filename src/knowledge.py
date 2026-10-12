"""Verified recommendation knowledge base (crop-specific and disease-specific)."""
import json
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

HEALTHY_INFO = {
    "risk": "Low",
    "symptoms": "none visible",
    "why": "an even green colour with no spots, patches or curling.",
}


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        with open(path, encoding="utf-8") as f:
            self.entries = json.load(f)["entries"]

    def crops(self):
        return {e["crop"] for e in self.entries}

    def get(self, crop: str, disease: str):
        for e in self.entries:
            if e["crop"] == crop and e["disease"] == disease:
                return e
        return None

    def is_supported(self, crop: str, disease: str) -> bool:
        if disease == "Healthy":
            return crop in self.crops()
        return self.get(crop, disease) is not None

    def info(self, crop: str, disease: str):
        if disease == "Healthy":
            return HEALTHY_INFO
        e = self.get(crop, disease)
        return {"risk": e["risk"], "symptoms": e["symptoms"], "why": e["why"]}
