# -*- coding: utf-8 -*-
"""
WM-811K (LSWMD.pkl) 전처리 스크립트
====================================
- 원본: 811,457개 웨이퍼 맵 (크기 제각각, uint8, 값: 0=배경, 1=정상 다이, 2=불량 다이)
- 라벨이 있는 172,950개 중 8개 불량 패턴 + none(정상) → 9클래스 분류용 데이터셋 생성
- 모든 웨이퍼 맵을 64x64로 리사이즈 (INTER_NEAREST: 다이 값 0/1/2 보존)
- 'none' 클래스가 압도적으로 많아(약 14.7만 장) 학습 효율을 위해 최대 13,000장으로 다운샘플링
- 결과: data/wm811k_64.npz (X, y, split) + data/class_names.json

주의: 2019년(pandas 0.19 / Python2) 시절 피클이라 최신 pandas에서 바로 열리지 않음
      → 아래 legacy module shim + encoding='latin1' 으로 해결 (트러블슈팅 포인트)

사용법:
    python src/prepare_data.py --pkl /path/to/LSWMD.pkl --out data/
"""
import argparse
import json
import os
import pickle
import sys
import types

import numpy as np


# ---------------------------------------------------------------------------
# 1. 레거시 피클 호환 셤(shim)
#    LSWMD.pkl은 pandas 0.19 시절 모듈 경로(pandas.indexes.*)를 참조한다.
#    최신 pandas(2.x/3.x)에는 해당 모듈이 없으므로 가짜 모듈을 등록해 준다.
# ---------------------------------------------------------------------------
def install_legacy_pandas_shims():
    import pandas as pd
    import pandas.core.indexes.base as _cib
    import pandas.core.indexes.range as _cir

    legacy = types.ModuleType("pandas.indexes")
    sys.modules["pandas.indexes"] = legacy
    sys.modules["pandas.indexes.base"] = _cib
    sys.modules["pandas.indexes.range"] = _cir

    num_mod = types.ModuleType("pandas.indexes.numeric")

    class Int64Index(pd.Index):
        # pandas 2.0에서 Int64Index가 제거됨 → 일반 Index로 대체
        def __new__(cls, data=None, dtype=None, copy=False, name=None, **kw):
            return pd.Index(data, dtype=dtype, copy=copy, name=name)

    num_mod.Int64Index = Int64Index
    num_mod.Float64Index = Int64Index
    sys.modules["pandas.indexes.numeric"] = num_mod


def load_lswmd(pkl_path):
    install_legacy_pandas_shims()
    with open(pkl_path, "rb") as f:
        # Python 2 시절 문자열 → encoding='latin1' 필수
        df = pickle.load(f, encoding="latin1")
    return df


# ---------------------------------------------------------------------------
# 2. 리사이즈 (0/1/2 라벨 맵이므로 최근접 보간)
# ---------------------------------------------------------------------------
def resize_map(wafer_map, size):
    import cv2
    return cv2.resize(wafer_map, (size, size), interpolation=cv2.INTER_NEAREST)


CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="wm811k/LSWMD.pkl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--none-cap", type=int, default=13000,
                    help="'none' 클래스 최대 샘플 수 (클래스 불균형 완화)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[1/4] LSWMD.pkl 로딩 중... (약 2GB, 1~2분 소요)")
    df = load_lswmd(args.pkl)
    print(f"      전체 웨이퍼 수: {len(df):,}")

    # failureType은 [['Center']] 형태의 중첩 배열, 라벨 없는 행은 빈 배열
    print("[2/4] 라벨 추출 및 필터링...")
    labels = []
    for ft in df["failureType"].values:
        try:
            labels.append(str(np.asarray(ft).ravel()[0]))
        except Exception:
            labels.append("")
    labels = np.array(labels)

    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    keep_mask = np.isin(labels, CLASS_NAMES)
    idx_all = np.where(keep_mask)[0]
    print(f"      라벨 보유 웨이퍼: {len(idx_all):,}")

    # 클래스별 인덱스 수집 + none 다운샘플링
    selected = []
    counts_before, counts_after = {}, {}
    for cls in CLASS_NAMES:
        cls_idx = idx_all[labels[idx_all] == cls]
        counts_before[cls] = int(len(cls_idx))
        if cls == "none" and len(cls_idx) > args.none_cap:
            cls_idx = rng.choice(cls_idx, size=args.none_cap, replace=False)
        counts_after[cls] = int(len(cls_idx))
        selected.append(cls_idx)
    sel = np.sort(np.concatenate(selected))
    print("      클래스 분포(원본 → 사용):")
    for c in CLASS_NAMES:
        print(f"        {c:<10} {counts_before[c]:>7,} → {counts_after[c]:>6,}")

    print(f"[3/4] {args.size}x{args.size} 리사이즈 중... ({len(sel):,}장)")
    X = np.zeros((len(sel), args.size, args.size), dtype=np.uint8)
    y = np.zeros(len(sel), dtype=np.int64)
    wm_col = df.columns.get_loc("waferMap")
    for i, ridx in enumerate(sel):
        X[i] = resize_map(df.iat[ridx, wm_col], args.size)
        y[i] = label_to_idx[labels[ridx]]
        if (i + 1) % 10000 == 0:
            print(f"      {i + 1:,} / {len(sel):,}")

    # 층화(stratified) train/val/test = 70/15/15 분할
    print("[4/4] 층화 분할 및 저장...")
    split = np.zeros(len(sel), dtype=np.int8)  # 0=train 1=val 2=test
    for c in range(len(CLASS_NAMES)):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        n = len(ci)
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        split[ci[n_tr:n_tr + n_va]] = 1
        split[ci[n_tr + n_va:]] = 2

    out_npz = os.path.join(args.out, f"wm811k_{args.size}.npz")
    np.savez_compressed(out_npz, X=X, y=y, split=split)
    with open(os.path.join(args.out, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES,
                   "counts_original": counts_before,
                   "counts_used": counts_after,
                   "split_ratio": [0.70, 0.15, 0.15]}, f, ensure_ascii=False, indent=2)

    sz_mb = os.path.getsize(out_npz) / 1e6
    print(f"완료! {out_npz} ({sz_mb:.1f} MB)")
    print(f"  train {int((split==0).sum()):,} / val {int((split==1).sum()):,} / test {int((split==2).sum()):,}")


if __name__ == "__main__":
    main()
