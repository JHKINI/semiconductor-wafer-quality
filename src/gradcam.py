# -*- coding: utf-8 -*-
"""
[확장버전 5단계] Grad-CAM: 모델이 웨이퍼의 어느 영역을 보고 판단했는지 시각화
==============================================================================
- 대상: models/cnn_best.keras 의 마지막 conv 블록 (conv_128_2)
- 출력: outputs/figures/gradcam_grid.png (클래스별 대표 샘플 + 히트맵 오버레이)
- Streamlit 앱에서도 이 모듈의 make_gradcam_heatmap()을 그대로 사용

사용법: python src/gradcam.py
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# 한글 폰트 설정 - Colab / Linux 환경
# ============================================================
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
else:
    # Colab이 아닌 환경에서 실행할 경우를 위한 fallback
    plt.rcParams["font.family"] = [
        "Malgun Gothic",
        "NanumGothic",
        "DejaVu Sans"
    ]

plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import load_dataset, to_onehot_channels, CLASS_NAMES

LAST_CONV = "conv_128_2"


def make_gradcam_heatmap(img_batch, model, last_conv_name=LAST_CONV,
                         pred_index=None):
    """img_batch: (1,64,64,3) float32 → (64,64) 0~1 히트맵"""

    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_out
    )

    pooled = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_out[0] * pooled,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    ) / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = tf.image.resize(
        heatmap[..., None],
        (64, 64)
    ).numpy()[..., 0]

    return heatmap


def main():

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    model = keras.models.load_model(
        "models/cnn_best.keras"
    )

    X, y, split = load_dataset()

    Xc = to_onehot_channels(X)

    te = np.where(
        split == 2
    )[0]

    from matplotlib.colors import ListedColormap

    wafer_cmap = ListedColormap(
        [
            "#0d1b2a",
            "#3fb68b",
            "#f4d35e"
        ]
    )

    rng = np.random.default_rng(1)

    fig, axes = plt.subplots(
        2,
        len(CLASS_NAMES),
        figsize=(2 * len(CLASS_NAMES), 4.6)
    )

    for c, cname in enumerate(CLASS_NAMES):

        cand = te[y[te] == c]

        idx = int(
            rng.choice(cand)
        )

        img = Xc[
            idx:idx + 1
        ]

        heat = make_gradcam_heatmap(
            img,
            model,
            pred_index=c
        )

        axes[0, c].imshow(
            X[idx],
            cmap=wafer_cmap,
            vmin=0,
            vmax=2
        )

        axes[0, c].set_title(
            cname,
            fontsize=9
        )

        axes[1, c].imshow(
            X[idx],
            cmap="gray",
            vmin=0,
            vmax=2
        )

        axes[1, c].imshow(
            heat,
            cmap="jet",
            alpha=0.55
        )

        for r in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    axes[0, 0].set_ylabel(
        "웨이퍼 맵",
        fontsize=10
    )

    axes[1, 0].set_ylabel(
        "Grad-CAM",
        fontsize=10
    )

    fig.suptitle(
        "Grad-CAM: CNN이 각 불량 패턴에서 주목한 영역"
    )

    fig.tight_layout()

    fig.savefig(
        "outputs/figures/gradcam_grid.png",
        dpi=150
    )

    plt.close(fig)

    print(
        "outputs/figures/gradcam_grid.png 저장"
    )


if __name__ == "__main__":
    main()