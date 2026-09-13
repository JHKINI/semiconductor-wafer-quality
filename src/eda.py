# -*- coding: utf-8 -*-
"""
WM-811K EDA: 클래스 분포 + 불량 유형별 웨이퍼 맵 샘플 시각화
사용법: python src/eda.py --data data/wm811k_64.npz --out outputs/figures
"""

import argparse
import json
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
    # 혹시 Colab이 아닌 환경에서 실행할 경우를 위한 fallback
    plt.rcParams["font.family"] = [
        "Malgun Gothic",
        "NanumGothic",
        "DejaVu Sans"
    ]

plt.rcParams["axes.unicode_minus"] = False

from matplotlib.colors import ListedColormap
import numpy as np


WAFER_CMAP = ListedColormap(
    ["#0d1b2a", "#3fb68b", "#f4d35e"]
)  # 0배경/1정상/2불량


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wm811k_64.npz")
    ap.add_argument("--meta", default="data/class_names.json")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data)
    X, y = d["X"], d["y"]

    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)

    names = meta["class_names"]

    # ========================================================
    # 1) 클래스 분포 (원본 vs 사용) - 로그 스케일
    # ========================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    orig = [meta["counts_original"][c] for c in names]
    used = [meta["counts_used"][c] for c in names]
    xpos = np.arange(len(names))

    ax.bar(
        xpos - 0.2,
        orig,
        width=0.4,
        label="원본",
        color="#5c7cfa"
    )

    ax.bar(
        xpos + 0.2,
        used,
        width=0.4,
        label="학습 사용",
        color="#f4a261"
    )

    ax.set_yscale("log")
    ax.set_xticks(xpos, names, rotation=30)
    ax.set_ylabel("샘플 수 (log)")
    ax.set_title(
        "WM-811K 클래스 분포: 'none' 다운샘플링으로 불균형 완화"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(args.out, "class_distribution.png"),
        dpi=150
    )

    plt.close(fig)

    # ========================================================
    # 2) 클래스별 샘플 웨이퍼 맵 그리드
    #    (9클래스 × 5샘플)
    # ========================================================
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(
        len(names),
        5,
        figsize=(9, 15)
    )

    for r, cname in enumerate(names):

        idx = np.where(y == r)[0]

        pick = rng.choice(
            idx,
            size=5,
            replace=False
        )

        for c in range(5):

            ax = axes[r, c]

            ax.imshow(
                X[pick[c]],
                cmap=WAFER_CMAP,
                vmin=0,
                vmax=2
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if c == 0:
                ax.set_ylabel(
                    cname,
                    fontsize=10,
                    rotation=0,
                    ha="right",
                    va="center"
                )

    fig.suptitle(
        "불량 패턴별 웨이퍼 맵 샘플 "
        "(초록=정상 다이, 노랑=불량 다이)",
        y=0.995
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(args.out, "samples_per_class.png"),
        dpi=150
    )

    plt.close(fig)

    print("EDA 그림 저장 완료:", args.out)


if __name__ == "__main__":
    main()