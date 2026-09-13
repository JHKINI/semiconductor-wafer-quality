# -*- coding: utf-8 -*-
"""
[확장버전 2단계] Hugging Face ViT fine-tuning
=============================================
- 모델: google/vit-base-patch16-224-in21k (ImageNet-21k 사전학습)
- 웨이퍼 맵 64x64(0/1/2) → 3채널 one-hot → 224x224 업샘플 후 파인튜닝
- 불균형 보정: 클래스 가중치 CrossEntropyLoss
- 출력: models/vit_best/ (HF 포맷), outputs/vit_test_preds.npz

Colab(T4) 기준 3 epoch에 약 40~50분 소요. (freeze 옵션 사용 시 단축 가능)
사용법: python src/train_vit.py --epochs 3
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from utils import load_dataset, to_onehot_channels, CLASS_NAMES

MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WaferDataset(Dataset):
    """one-hot 3채널 웨이퍼 맵을 224x224로 업샘플해 ViT 입력으로 변환"""

    def __init__(self, X, y, train=False):
        self.X = X          # (N,64,64,3) float32
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(2, 0, 1)     # (3,64,64)
        if self.train:                                        # 회전/반전 증강
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=[1, 2])
        x = F.interpolate(x.unsqueeze(0), size=224,
                          mode="nearest").squeeze(0)          # (3,224,224)
        # ViT 사전학습 통계 대신 [-1,1] 스케일 (one-hot 입력이라 단순 스케일이 안정적)
        x = x * 2.0 - 1.0
        return x, int(self.y[i])


def evaluate(model, loader):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, yb in loader:
            logits = model(pixel_values=x.to(DEVICE)).logits
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_y.append(yb.numpy())
    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y)
    return probs, float((probs.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wm811k_64.npz")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="분류 헤드만 학습 (빠르지만 성능은 다소 낮음)")
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X, y, split = load_dataset(args.data)
    Xc = to_onehot_channels(X)
    tr, va, te = split == 0, split == 1, split == 2

    dl_tr = DataLoader(WaferDataset(Xc[tr], y[tr], train=True),
                       batch_size=args.batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(WaferDataset(Xc[va], y[va]), batch_size=64, num_workers=2)
    dl_te = DataLoader(WaferDataset(Xc[te], y[te]), batch_size=64, num_workers=2)

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        id2label={i: c for i, c in enumerate(CLASS_NAMES)},
        label2id={c: i for i, c in enumerate(CLASS_NAMES)},
    ).to(DEVICE)

    if args.freeze_backbone:
        for p in model.vit.parameters():
            p.requires_grad = False

    counts = np.bincount(y[tr], minlength=len(CLASS_NAMES)).astype(np.float32)
    w = torch.tensor(len(y[tr]) / (len(CLASS_NAMES) * counts)).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        for step, (xb, yb) in enumerate(dl_tr):
            logits = model(pixel_values=xb.to(DEVICE)).logits
            loss = criterion(logits, yb.to(DEVICE))
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 100 == 0:
                print(f"epoch {ep+1} step {step}/{len(dl_tr)} loss {loss.item():.4f}")
        _, va_acc = evaluate(model, dl_va)
        print(f"[epoch {ep+1}] val accuracy = {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            model.save_pretrained("models/vit_best")
            print("  → models/vit_best 저장")

    # best 모델로 테스트 예측 저장
    model = ViTForImageClassification.from_pretrained("models/vit_best").to(DEVICE)
    probs, te_acc = evaluate(model, dl_te)
    np.savez("outputs/vit_test_preds.npz",
             probs=probs, y_true=y[te], class_names=np.array(CLASS_NAMES))
    print(f"[ViT] test accuracy = {te_acc:.4f}  → outputs/vit_test_preds.npz 저장")


if __name__ == "__main__":
    main()
