# AgriCare AI edited

AI-powered plant disease detection with biological and non-chemical treatment recommendations.

**HackZen 2.0 · Computer Vision Track · Team 18**
Captain: Jagaprasanth S · Members: Nithin S, Varunesh T

## What it does
A farmer uploads or captures a leaf photo. The backend checks it with OpenCV, resizes it to 224 × 224 and normalizes pixels to 0–1. A MobileNetV2 transfer-learning classifier predicts the crop condition and a confidence score. The app then shows crop-specific, prevention-first guidance. Low-confidence results ask for a clearer image or expert verification.

## Status
| Part | Status |
| --- | --- |
| Web app (dashboard, scan, history, guide, admin) | Prototype |
| OpenCV image checks and preprocessing | Implemented |
| Training script (MobileNetV2 transfer learning) | Written. Run it to create the model |
| Disease prediction | **Simulated until `model/plant_model.keras` exists** |
| Verified biological options | None added yet |
| Measured accuracy | [Insert measured accuracy] |

The model performs image classification only. It does not detect or outline objects.

## Project structure
```
agricare-ai/
├── app.py                  # Flask server and API
├── train_model.py          # Transfer-learning training and evaluation
├── requirements.txt
├── src/
│   ├── preprocess.py       # OpenCV validation, resize, normalize
│   ├── predict.py          # Model loading and prediction (demo fallback)
│   └── knowledge.py        # Knowledge base access
├── data/knowledge_base.json  # Crop- and disease-specific guidance
├── templates/index.html    # Page
├── static/css/style.css    # Styles
├── static/js/app.js        # Front end logic
├── tests/test_basic.py     # Unit tests
├── model/                  # Trained model goes here (not committed)
└── docs/                   # Report PDF, screenshots, metrics
```

## Run the app
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```
Without a trained model, the banner shows "Demo mode" and predictions are simulated. The OpenCV image checks are always real.

## Train the model
1. Download a plant disease dataset such as PlantVillage and arrange it as one folder per class, for example `data/PlantVillage/Tomato___Early_blight/`.
2. Train:
```bash
python train_model.py --data data/PlantVillage --epochs 5 --finetune 3
```
3. The script saves `model/plant_model.keras` and `model/class_names.json`, plus `docs/metrics.json` and `docs/confusion_matrix.png`. Restart `python app.py` and the banner changes to "Trained model connected".

Report only the numbers the script prints after you have actually run it. Class folders must follow the `Crop___Disease` naming so the app can show the crop and disease.

## API
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | `{status, model_loaded}` |
| GET | `/api/knowledge` | Knowledge base entries |
| POST | `/api/predict` | Multipart field `image`. Returns `status` (`ok`, `poor`, `unsupported`), crop, disease, confidence, top 3, image stats, `simulated` |

## Tests
```bash
python -m pytest
```

## Knowledge base rules
- Entries are crop-specific and disease-specific.
- Every entry needs a trusted source (agricultural university, extension service or official guidance).
- Add biological options only when a source supports that crop and disease pair.
- Never add pesticide doses or unverified claims.

## Limitations
Image quality, similar symptoms, limited data and unsupported crops affect results. Confidence is not proof of a correct diagnosis. This tool is decision support, not a replacement for an agricultural expert.

## Links
- Report: [Insert documentation PDF link]
- Demo video: [Insert Demo Video URL]
