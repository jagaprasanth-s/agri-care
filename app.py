"""AgriCare AI: Flask backend and web app host.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os

from flask import Flask, jsonify, render_template, request

from src.knowledge import KnowledgeBase
from src.predict import Classifier
from src.preprocess import ImageProblem, check_quality, decode_image, preprocess, quality_stats

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

kb = KnowledgeBase()
classifier = Classifier(kb)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_loaded=classifier.loaded)


@app.get("/api/knowledge")
def knowledge():
    return jsonify(entries=kb.entries)


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(status="error", message="Send an image in the 'image' field."), 400
    try:
        img = decode_image(file.read())
    except ImageProblem as exc:
        return jsonify(status="poor", why=str(exc), stats=None)

    stats = quality_stats(img)
    why = check_quality(stats)
    if why:
        return jsonify(status="poor", why=why, stats=stats)

    batch = preprocess(img)
    stats.update(min=f"{float(batch.min()):.2f}", max=f"{float(batch.max()):.2f}")

    crop, disease, conf, top3, simulated = classifier.predict(batch)
    top3_out = [{"crop": c, "disease": d, "conf": round(p * 100)} for c, d, p in top3]

    if not kb.is_supported(crop, disease):
        return jsonify(status="unsupported", crop=crop, stats=stats, top3=top3_out, simulated=simulated)

    return jsonify(
        status="ok",
        crop=crop,
        dis=disease,
        k="H" if disease == "Healthy" else "D",
        conf=round(conf * 100),
        info=kb.info(crop, disease),
        top3=top3_out,
        stats=stats,
        simulated=simulated,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(status="poor", why="The image is larger than 8 MB.", stats=None), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
