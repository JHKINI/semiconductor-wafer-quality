# -*- coding: utf-8 -*-
"""
[확장버전 1단계] Keras CNN 베이스라인 학습
==========================================
- 입력: 64x64 웨이퍼 맵 → 3채널 one-hot (배경/정상다이/불량다이)
- 증강: 좌우/상하 반전 + 90도 회전 (웨이퍼는 회전 대칭이라 라벨 불변)
- 불균형 보정: class_weight
- 출력: models/cnn_best.keras, outputs/cnn_history.json, outputs/cnn_test_preds.npz

Colab(T4) 기준 약 10~15분 소요.
사용법: python src/train_cnn.py --epochs 30
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import load_dataset, to_onehot_channels, compute_class_weights, CLASS_NAMES


def build_cnn(input_shape=(64, 64, 3), n_classes=9):
    """VGG 스타일의 가벼운 CNN. 파라미터 약 1.2M."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          name=f"conv_{filters}_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="wafer_cnn")


def augment(x, y):
    """회전/반전 증강. 웨이퍼 불량 패턴은 방향 대칭이므로 라벨이 유지된다.
    (단, Edge-Ring/Edge-Loc 등도 회전해도 같은 클래스)"""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wm811k_64.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X, y, split = load_dataset(args.data)
    Xc = to_onehot_channels(X)
    tr, va, te = split == 0, split == 1, split == 2

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xc[tr], y[tr]))
             .shuffle(20000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(args.batch).prefetch(tf.data.AUTOTUNE))
    ds_va = tf.data.Dataset.from_tensor_slices((Xc[va], y[va])).batch(args.batch)

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("models/cnn_best.keras",
                                        monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                      restore_best_weights=True),
    ]
    hist = model.fit(
        ds_tr, validation_data=ds_va, epochs=args.epochs,
        class_weight=compute_class_weights(y[tr]),
        callbacks=callbacks,
    )
    with open("outputs/cnn_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f)

    # 테스트셋 예측 저장 (evaluate.py에서 공통 분석)
    probs = model.predict(Xc[te], batch_size=256)
    np.savez("outputs/cnn_test_preds.npz",
             probs=probs, y_true=y[te], class_names=np.array(CLASS_NAMES))
    acc = float((probs.argmax(1) == y[te]).mean())
    print(f"[CNN] test accuracy = {acc:.4f}  → outputs/cnn_test_preds.npz 저장")


if __name__ == "__main__":
    main()
