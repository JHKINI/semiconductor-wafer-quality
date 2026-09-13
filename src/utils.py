# -*- coding: utf-8 -*-
"""공통 유틸: 데이터 로딩 / 전처리 / 클래스 가중치"""
import json
import numpy as np

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]


def load_dataset(npz_path="data/wm811k_64.npz"):
    """npz에서 (X, y, split) 로드. X: (N,64,64) uint8, 값 0/1/2"""
    d = np.load(npz_path)
    return d["X"], d["y"], d["split"]


def to_onehot_channels(X):
    """웨이퍼 맵(0=배경,1=정상,2=불량)을 3채널 one-hot 이미지로 변환.
    단순 정규화(x/2)보다 다이 상태를 채널로 분리하는 편이 CNN에 유리하다."""
    X = X.astype(np.uint8)
    out = np.zeros((*X.shape, 3), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = (X == ch).astype(np.float32)
    return out


def compute_class_weights(y, n_classes=9):
    """불균형 보정용 클래스 가중치 (sklearn 'balanced' 방식)"""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = len(y) / (n_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def load_class_names(meta_path="data/class_names.json"):
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)["class_names"]
    except FileNotFoundError:
        return CLASS_NAMES
