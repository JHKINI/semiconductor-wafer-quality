# -*- coding: utf-8 -*-
"""
[확장버전 3~4단계] 두 모델 성능 비교 + 불량 유형별 오분류 분석
==============================================================
입력: outputs/cnn_test_preds.npz, outputs/vit_test_preds.npz
출력:
  outputs/figures/confusion_cnn.png / confusion_vit.png
  outputs/figures/model_comparison.png
  outputs/figures/misclassified_examples.png
  outputs/model_comparison.md  (README에 붙여 넣을 수 있는 결과 표)

사용법: python src/evaluate.py
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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from utils import load_dataset, CLASS_NAMES


FIG_DIR = "outputs/figures"


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.imshow(
        cm,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES
    )

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if cm[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black"
                )

    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(title)

    fig.colorbar(
        im,
        fraction=0.046
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150
    )

    plt.close(fig)


def per_class_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average=None,
        labels=range(len(CLASS_NAMES))
    )


def main():

    os.makedirs(
        FIG_DIR,
        exist_ok=True
    )

    results = {}

    for name, path in [
        ("CNN", "outputs/cnn_test_preds.npz"),
        ("ViT", "outputs/vit_test_preds.npz")
    ]:

        if not os.path.exists(path):
            print(
                f"[skip] {path} 없음 — "
                "해당 모델 학습을 먼저 실행하세요."
            )
            continue

        d = np.load(
            path,
            allow_pickle=True
        )

        probs = d["probs"]
        y_true = d["y_true"]

        y_pred = probs.argmax(1)

        results[name] = dict(
            probs=probs,
            y_true=y_true,
            y_pred=y_pred
        )

        acc = accuracy_score(
            y_true,
            y_pred
        )

        mf1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )

        print(
            f"=== {name} ===  "
            f"accuracy {acc:.4f} / "
            f"macro-F1 {mf1:.4f}"
        )

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=CLASS_NAMES,
                digits=3
            )
        )

        plot_confusion(
            y_true,
            y_pred,
            f"{name} Confusion Matrix (row-normalized)",
            f"{FIG_DIR}/confusion_{name.lower()}.png"
        )

    # ========================================================
    # 모델 비교 막대그래프
    # ========================================================
    if len(results) == 2:

        f1_cnn = per_class_f1(
            results["CNN"]["y_true"],
            results["CNN"]["y_pred"]
        )

        f1_vit = per_class_f1(
            results["ViT"]["y_true"],
            results["ViT"]["y_pred"]
        )

        xpos = np.arange(
            len(CLASS_NAMES)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            xpos - 0.2,
            f1_cnn,
            width=0.4,
            label="CNN",
            color="#5c7cfa"
        )

        ax.bar(
            xpos + 0.2,
            f1_vit,
            width=0.4,
            label="ViT",
            color="#f4a261"
        )

        ax.set_xticks(
            xpos,
            CLASS_NAMES,
            rotation=30
        )

        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1.05)

        ax.set_title(
            "클래스별 F1: CNN vs ViT"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            f"{FIG_DIR}/model_comparison.png",
            dpi=150
        )

        plt.close(fig)

        # ====================================================
        # 결과 표 마크다운
        # ====================================================
        with open(
            "outputs/model_comparison.md",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "| 클래스 | CNN F1 | ViT F1 |\n"
                "|---|---|---|\n"
            )

            for i, c in enumerate(CLASS_NAMES):

                f.write(
                    f"| {c} | "
                    f"{f1_cnn[i]:.3f} | "
                    f"{f1_vit[i]:.3f} |\n"
                )

            for name in ("CNN", "ViT"):

                r = results[name]

                f.write(
                    f"| **{name} 전체** | "
                    f"acc "
                    f"{accuracy_score(r['y_true'], r['y_pred']):.4f} | "
                    f"macro-F1 "
                    f"{f1_score(r['y_true'], r['y_pred'], average='macro'):.4f} |\n"
                )

        print(
            "outputs/model_comparison.md 저장"
        )

    # ========================================================
    # 오분류 사례 시각화
    # ========================================================
    if results:

        best = max(
            results,
            key=lambda k: accuracy_score(
                results[k]["y_true"],
                results[k]["y_pred"]
            )
        )

        r = results[best]

        X, y, split = load_dataset()

        X_te = X[split == 2]

        wrong = np.where(
            r["y_pred"] != r["y_true"]
        )[0]

        # 신뢰도 높은데 틀린 순서로 정렬
        conf = r["probs"][wrong].max(1)

        wrong = wrong[
            np.argsort(-conf)
        ][:12]

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(
            [
                "#0d1b2a",
                "#3fb68b",
                "#f4d35e"
            ]
        )

        fig, axes = plt.subplots(
            3,
            4,
            figsize=(12, 9)
        )

        for ax, wi in zip(
            axes.ravel(),
            wrong
        ):

            ax.imshow(
                X_te[wi],
                cmap=cmap,
                vmin=0,
                vmax=2
            )

            ax.set_title(
                f"실제 {CLASS_NAMES[r['y_true'][wi]]}\n"
                f"예측 {CLASS_NAMES[r['y_pred'][wi]]} "
                f"({r['probs'][wi].max():.0%})",
                fontsize=9
            )

            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"{best} 고신뢰 오분류 사례 — "
            "Loc↔Edge-Loc, Scratch↔Loc 혼동이 주 패턴"
        )

        fig.tight_layout()

        fig.savefig(
            f"{FIG_DIR}/misclassified_examples.png",
            dpi=150
        )

        plt.close(fig)

        print(
            f"{FIG_DIR}/misclassified_examples.png 저장"
        )


if __name__ == "__main__":
    main()