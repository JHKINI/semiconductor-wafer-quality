# -*- coding: utf-8 -*-
"""
[확장버전 6단계] 웨이퍼 불량 패턴 품질 진단 대시보드 (Streamlit)
================================================================
기능
  1. 웨이퍼 맵 업로드(.npy 64x64 / 이미지 파일) 또는 테스트셋 샘플 선택
  2. CNN / ViT 예측 결과 + 클래스별 신뢰도 막대그래프
  3. Grad-CAM 히트맵으로 모델이 주목한 불량 영역 표시
  4. 테스트셋 전체 불량 유형 분포 대시보드 (품질관리 관점)

실행: streamlit run app/streamlit_app.py
(models/cnn_best.keras 가 있어야 예측 탭 동작. ViT는 models/vit_best 있으면 자동 활성화)
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import load_dataset, to_onehot_channels, CLASS_NAMES

WAFER_CMAP = ListedColormap(["#0d1b2a", "#3fb68b", "#f4d35e"])
CLASS_DESC = {
    "Center": "웨이퍼 중앙부 집중 불량 — CMP/증착 공정 중심부 이상 의심",
    "Donut": "도넛 형태 링 불량 — 스핀 코팅 회전 불균일 의심",
    "Edge-Loc": "가장자리 국부 불량 — 엣지 핸들링/척 접촉 이상 의심",
    "Edge-Ring": "가장자리 링 전체 불량 — 베벨 식각/열처리 경계 이상 의심",
    "Loc": "국부(클러스터) 불량 — 파티클 오염 의심",
    "Near-full": "웨이퍼 대부분 불량 — 공정 전면 이상, 즉시 라인 점검 필요",
    "Random": "무작위 산포 불량 — 랜덤 파티클, 클린룸 환경 점검",
    "Scratch": "스크래치 선형 불량 — 이송/핸들링 장비 접촉 의심",
    "none": "특정 패턴 없음 — 정상 범주",
}


@st.cache_resource
def load_cnn():
    from tensorflow import keras
    path = "models/cnn_best.keras"
    return keras.models.load_model(path) if os.path.exists(path) else None


@st.cache_resource
def load_vit():
    if not os.path.isdir("models/vit_best"):
        return None
    import torch
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained("models/vit_best")
    model.eval()
    return model


@st.cache_data
def load_test_split():
    X, y, split = load_dataset("data/wm811k_64.npz")
    te = split == 2
    return X[te], y[te]


def predict_cnn(model, wafer64):
    x = to_onehot_channels(wafer64[None])
    return model.predict(x, verbose=0)[0]


def predict_vit(model, wafer64):
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(to_onehot_channels(wafer64[None])).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=224, mode="nearest") * 2.0 - 1.0
    with torch.no_grad():
        logits = model(pixel_values=x).logits
    return torch.softmax(logits, dim=1).numpy()[0]


def show_wafer(wafer64, title=""):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(wafer64, cmap=WAFER_CMAP, vmin=0, vmax=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    st.pyplot(fig)
    plt.close(fig)


def parse_upload(file):
    """npy(64x64, 값 0/1/2) 또는 이미지 파일 → 64x64 uint8 웨이퍼 맵"""
    import cv2
    if file.name.endswith(".npy"):
        arr = np.load(file)
    else:
        data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        # 밝기 3단계 양자화 (배경/정상/불량)
        arr = np.digitize(img, bins=[img.max() * 0.33, img.max() * 0.66])
    arr = cv2.resize(arr.astype(np.uint8), (64, 64),
                     interpolation=cv2.INTER_NEAREST)
    return np.clip(arr, 0, 2).astype(np.uint8)


# ============================ UI ============================
st.set_page_config(page_title="웨이퍼 불량 패턴 품질 진단", layout="wide")
st.title("반도체 웨이퍼 맵 불량 패턴 분류 · 품질 진단 대시보드")
st.caption("WM-811K | Keras CNN + HuggingFace ViT | Grad-CAM 설명 가능 AI")

cnn = load_cnn()
vit = load_vit()

tab_pred, tab_dash = st.tabs(["불량 패턴 진단", "테스트셋 품질 현황"])

# ---------------- 탭 1: 진단 ----------------
with tab_pred:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("웨이퍼 맵 입력")
        src = st.radio("입력 방식", ["테스트셋 샘플", "파일 업로드"])
        wafer, true_label = None, None
        if src == "테스트셋 샘플":
            X_te, y_te = load_test_split()
            cls = st.selectbox("불량 유형 선택", CLASS_NAMES)
            cand = np.where(y_te == CLASS_NAMES.index(cls))[0]
            i = st.slider("샘플 번호", 0, len(cand) - 1, 0)
            wafer, true_label = X_te[cand[i]], cls
        else:
            up = st.file_uploader("64x64 .npy 또는 웨이퍼 맵 이미지", type=["npy", "png", "jpg"])
            if up is not None:
                wafer = parse_upload(up)
        if wafer is not None:
            show_wafer(wafer, f"실제: {true_label}" if true_label else "업로드 웨이퍼")

    with right:
        if wafer is None:
            st.info("왼쪽에서 웨이퍼 맵을 선택하거나 업로드하세요.")
        elif cnn is None:
            st.warning("models/cnn_best.keras 가 없습니다. train_cnn.py를 먼저 실행하세요.")
        else:
            st.subheader("모델 예측 결과")
            probs_cnn = predict_cnn(cnn, wafer)
            cols = st.columns(2 if vit is not None else 1)

            def render(col, name, probs):
                pred = CLASS_NAMES[int(probs.argmax())]
                with col:
                    st.metric(f"{name} 예측", pred, f"신뢰도 {probs.max():.1%}")
                    fig, ax = plt.subplots(figsize=(4, 2.6))
                    order = np.argsort(probs)
                    ax.barh([CLASS_NAMES[i] for i in order], probs[order],
                            color=["#f4a261" if i == probs.argmax() else "#5c7cfa"
                                   for i in order])
                    ax.set_xlim(0, 1)
                    st.pyplot(fig); plt.close(fig)
                return pred

            pred_cnn = render(cols[0], "CNN", probs_cnn)
            if vit is not None:
                render(cols[1], "ViT", predict_vit(vit, wafer))

            st.markdown(f"**품질관리 소견:** {CLASS_DESC[pred_cnn]}")

            # Grad-CAM
            st.subheader("Grad-CAM — 모델이 주목한 영역")
            from gradcam import make_gradcam_heatmap
            heat = make_gradcam_heatmap(to_onehot_channels(wafer[None]), cnn)
            fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
            axes[0].imshow(wafer, cmap=WAFER_CMAP, vmin=0, vmax=2)
            axes[0].set_title("웨이퍼 맵", fontsize=10)
            axes[1].imshow(wafer, cmap="gray", vmin=0, vmax=2)
            axes[1].imshow(heat, cmap="jet", alpha=0.55)
            axes[1].set_title("Grad-CAM", fontsize=10)
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            st.pyplot(fig); plt.close(fig)

# ---------------- 탭 2: 품질 현황 대시보드 ----------------
with tab_dash:
    X_te, y_te = load_test_split()
    st.subheader("테스트셋 불량 유형 분포 (라인 품질 모니터링 가정)")
    counts = np.bincount(y_te, minlength=len(CLASS_NAMES))
    defect_total = counts[:-1].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 웨이퍼", f"{len(y_te):,}장")
    c2.metric("불량 패턴 검출", f"{defect_total:,}장")
    c3.metric("불량 패턴 비율", f"{defect_total / len(y_te):.1%}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(CLASS_NAMES, counts, color="#5c7cfa")
    ax.set_ylabel("웨이퍼 수")
    plt.xticks(rotation=30)
    st.pyplot(fig); plt.close(fig)

    st.markdown(
        "불량 유형별 발생 비율을 추적하면 **어느 공정 단계에서 이상이 시작됐는지** "
        "역추적할 수 있습니다. 예: Edge-Ring 급증 → 베벨 식각 장비 점검, "
        "Scratch 급증 → 웨이퍼 이송 로봇 점검."
    )
