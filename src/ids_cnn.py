"""
1D CNN intrusion detection on CIC-IDS2017 flow features.

Reproduces the deep-learning core of Sudyana et al. (2024) on two public
CIC-IDS2017 capture days: Tuesday (benign + FTP-Patator + SSH-Patator) and
Friday afternoon (benign + DDoS). The task is binary, BENIGN vs ATTACK.

Two evaluation protocols are supported (see common.py):
  random  combined days, random stratified 80/20 split
  day     train on Tuesday, test on Friday (unseen attack family)

Usage:
    python src/ids_cnn.py --split random
    python src/ids_cnn.py --split day
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before tensorflow is imported

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

import common

EPOCHS = 15
BATCH_SIZE = 256


def build_model(n_features):
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def plot_training_curves(history, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(history.history[key], label="Train")
        ax.plot(history.history[f"val_{key}"], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out / "fig2_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=common.SPLITS, default="random",
                        help="evaluation protocol (default: random)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: results/<split>/cnn)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out_dir or common.ROOT / "results" / args.split / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    results = {"model": "1d_cnn", "split": args.split, "seed": args.seed}

    X_train_df, X_test_df, y_train, y_test = common.build_splits(args.split, args.seed, results)
    X_train_df, X_test_df = common.impute(X_train_df, X_test_df)
    common.record_split(results, X_train_df, X_test_df, y_train, y_test)
    common.plot_class_distribution(args.split, results, y_train, y_test, out)

    # Scaler is fit on the training split only. Conv1D expects (samples, steps,
    # channels), so the features are treated as a 1D sequence with one channel.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)[..., np.newaxis]
    X_test = scaler.transform(X_test_df)[..., np.newaxis]

    # Hold out 20% of the training split for early stopping. The test split is
    # never used for fitting, validation, or model selection.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )
    model = build_model(X_tr.shape[1])
    model.summary()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ],
        verbose=2,
    )
    results["epochs_trained"] = len(history.history["loss"])
    results["final_train_loss"] = float(history.history["loss"][-1])
    results["final_val_loss"] = float(history.history["val_loss"][-1])
    results["final_train_acc"] = float(history.history["accuracy"][-1])
    results["final_val_acc"] = float(history.history["val_accuracy"][-1])
    plot_training_curves(history, out)

    proba = model.predict(X_test, batch_size=1024, verbose=0).ravel()
    cm = common.evaluate(y_test, proba, results)
    common.plot_confusion_matrix(cm, args.split, out)

    model.save(out / "cicids2017_1dcnn.keras")
    joblib.dump(scaler, out / "cicids2017_scaler.joblib")
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Artifacts written to:", out)


if __name__ == "__main__":
    main()
