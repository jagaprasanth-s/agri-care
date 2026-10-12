"""Train the leaf-disease classifier with transfer learning (MobileNetV2).

Expected dataset layout (for example the PlantVillage dataset), one folder per class:
    data/PlantVillage/Tomato___Early_blight/*.jpg
    data/PlantVillage/Tomato___healthy/*.jpg
    ...

Usage:
    python train_model.py --data data/PlantVillage --epochs 5
    python train_model.py --data data/PlantVillage --epochs 5 --finetune 3

Only report the numbers this script prints after you have actually run it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG = 224
ROOT = Path(__file__).resolve().parent


def load_data(data_dir: str, batch: int):
    common = dict(validation_split=0.2, seed=42, image_size=(IMG, IMG), batch_size=batch)
    train = keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val = keras.utils.image_dataset_from_directory(data_dir, subset="validation", shuffle=False, **common)
    names = train.class_names
    norm = lambda x, y: (x / 255.0, y)  # same 0-1 normalization as src/preprocess.py
    autotune = tf.data.AUTOTUNE
    return train.map(norm).prefetch(autotune), val.map(norm).prefetch(autotune), names


def build_model(n_classes: int):
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1, value_range=(0, 1)),
        ],
        name="augmentation",
    )
    inputs = keras.Input((IMG, IMG, 3))  # values 0-1
    x = augment(inputs)
    x = layers.Rescaling(2.0, offset=-1.0)(x)  # 0-1 -> -1..1, as MobileNetV2 expects
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate(model, val, names):
    y_true, y_pred = [], []
    for x, y in val:
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1).tolist())
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
        print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        cm = confusion_matrix(y_true, y_pred)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Greens")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion matrix (validation set)")
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / "confusion_matrix.png", dpi=150)
        print("Saved docs/metrics.json and docs/confusion_matrix.png")
    except ImportError:
        print("Install scikit-learn and matplotlib to get the full evaluation report.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with one sub-folder per class")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--finetune", type=int, default=0, help="Extra epochs with the top of the base model unfrozen")
    args = ap.parse_args()

    train, val, names = load_data(args.data, args.batch)
    print(f"{len(names)} classes found")
    model, base = build_model(len(names))
    compile_model(model, 1e-3)
    model.fit(train, validation_data=val, epochs=args.epochs)

    if args.finetune > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train, validation_data=val, epochs=args.finetune)

    loss, acc = model.evaluate(val, verbose=0)
    print(f"Validation accuracy: {acc:.4f}")
    evaluate(model, val, names)

    (ROOT / "model").mkdir(exist_ok=True)
    model.save(ROOT / "model" / "plant_model.keras")
    (ROOT / "model" / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print("Saved model/plant_model.keras and model/class_names.json")


if __name__ == "__main__":
    main()
