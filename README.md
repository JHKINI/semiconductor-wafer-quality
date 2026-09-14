# 반도체 웨이퍼 맵 불량 패턴 분류 및 품질 진단 대시보드

> 웨이퍼 맵 이미지를 기반으로 반도체 불량 패턴을 자동 분류하고, 불량 유형별 분포·모델 성능·오분류 사례를 분석하여 품질 진단과 불량 원인 후보의 점검 우선순위를 지원하는 개인 프로젝트입니다.

## 프로젝트 배경

기존에는 번호판 인식(NextFrame ALPR) 프로젝트로 YOLO 기반 이미지 탐지·OCR·성능 평가를 경험했고, 이후 데이터 품질 스코어카드 프로젝트를 통해 Streamlit 기반 진단 결과 시각화 구조를 경험했습니다. 이를 확장해 **반도체 품질 분야에서 실제로 활용되는 웨이퍼 맵 불량 패턴 분류 모델과 품질 진단 대시보드**를 개인 프로젝트로 구현했습니다.

## 데이터셋: WM-811K (LSWMD)

- 실제 반도체 팹에서 수집된 **811,457장**의 웨이퍼 맵 (46,393 로트)
- 각 픽셀 값: `0=배경`, `1=정상 다이`, `2=불량 다이`
- 이 중 라벨이 부여된 **172,950장**을 사용, 9개 클래스(8개 불량 패턴 + none) 분류

| 클래스 | 원본 수량 | 학습 사용 | 대표적인 원인 후보 |
|---|---:|---:|---|
| Center | 4,294 | 4,294 | CMP/증착 중심부 이상 |
| Donut | 555 | 555 | 스핀 코팅 회전 불균일 |
| Edge-Loc | 5,189 | 5,189 | 엣지 핸들링/척 접촉 |
| Edge-Ring | 9,680 | 9,680 | 베벨 식각·열처리 경계 |
| Loc | 3,593 | 3,593 | 국부 파티클 오염 |
| Near-full | 149 | 149 | 공정 전면 이상 |
| Random | 866 | 866 | 랜덤 파티클(클린룸) |
| Scratch | 1,193 | 1,193 | 이송 장비 접촉 |
| none | 147,431 | 13,000 | 정상 (다운샘플링) |

> ※ 원인 후보는 웨이퍼 맵 패턴에 대한 도메인 지식을 바탕으로 정리한 것으로, 본 데이터셋에는 개별 공정 조건·장비 정보가 포함되어 있지 않아 실제 불량 원인으로 검증한 결과는 아닙니다.

> ⚠️ 본 프로젝트의 데이터셋은 WM-811K 원본 데이터셋을 기반으로 하였으며,
> 프로젝트에서는 필요한 데이터를 선별·전처리하여 모델 학습에 사용하였다.
> 원본 데이터셋 자체는 본 저장소에 포함하지 않는다.

### Dataset Reference

M.-J. Wu, J.-S. R. Jang, and J.-L. Chen,
"Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets,"
*IEEE Transactions on Semiconductor Manufacturing*, vol. 28, no. 1, pp. 1–12, 2015.

Dataset:
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
`none`이 전체의 85%를 차지하는 극단적 불균형 → **다운샘플링(13,000장) + 클래스 가중치**로 이중 보정했습니다. 최종 학습 데이터는 38,519장(train 26,961 / val 5,774 / test 5,784, 층화 분할)입니다.

![클래스 분포](outputs/figures/class_distribution.png)
![클래스별 샘플](outputs/figures/samples_per_class.png)

## 파이프라인

```
LSWMD.pkl (2GB, 2019년 pandas 0.19 피클)
  └─ prepare_data.py   레거시 피클 호환 셤 + latin1 디코딩 → 64x64 리사이즈
                       → wm811k_64.npz (9.4MB)
  └─ eda.py            클래스 분포 / 샘플 시각화
  └─ train_cnn.py      Keras CNN baseline (증강 + class_weight)
  └─ train_vit.py      HuggingFace ViT(google/vit-base-patch16-224-in21k) fine-tuning
  └─ evaluate.py       accuracy / F1 / confusion matrix / 모델 비교 / 오분류 분석
  └─ gradcam.py        Grad-CAM으로 모델 판단 근거 시각화
  └─ app/streamlit_app.py  진단 대시보드 (업로드 → 예측 + 신뢰도 + Grad-CAM)
```

### 기술 포인트

1. **레거시 데이터 호환 처리** — LSWMD.pkl은 pandas 0.19/Python 2 시절 피클이라 최신 pandas에서 `ModuleNotFoundError: pandas.indexes`, `UnicodeDecodeError`가 발생합니다. 가짜 모듈 등록(shim)과 `encoding='latin1'`로 해결했습니다.
2. **도메인 반영 입력 인코딩** — 픽셀값 0/1/2를 단순 정규화하지 않고 **3채널 one-hot(배경/정상/불량)** 으로 분리해 모델이 다이 상태를 명확히 구분하도록 설계했습니다.
3. **대칭성 기반 증강** — 웨이퍼 불량 패턴은 회전·반전해도 클래스가 유지되므로 90° 회전 + 상하/좌우 반전 증강을 적용해 소수 클래스(Donut 555장, Near-full 149장)를 보완했습니다.
4. **파이프라인 사전 검증** — GPU 학습 전에 RandomForest(26x26)로 라벨 학습 가능성을 확인했습니다: accuracy 78.3% / macro-F1 0.647. 이 수치가 딥러닝 모델의 최소 기준선(baseline floor)이 됩니다.

### 실행 환경

- Python 3.13
- TensorFlow / Keras
- PyTorch / Hugging Face Transformers
- Google Colab (T4 GPU 권장)

> 한글 시각화는 Colab 환경의 NanumGothic 폰트를 사용합니다.

## 실행 방법

```bash
pip install -r requirements.txt

# 1. 전처리 (LSWMD.pkl 필요 — Kaggle: qingyi/wm811k-wafer-map)
python src/prepare_data.py --pkl /path/to/LSWMD.pkl --out data

# 2. EDA
python src/eda.py

# 3. 학습 (Colab T4 권장 — notebooks/colab_train.ipynb 참고)
cd src
python train_cnn.py --epochs 30 --data ../data/wm811k_64.npz     # 약 10~15분
python train_vit.py --epochs 3 --data ../data/wm811k_64.npz      # 약 40~50분

# 4. 평가 + Grad-CAM
ln -sf ../data data
python evaluate.py
python gradcam.py
cd ..

# 5. 대시보드
streamlit run app/streamlit_app.py
```

## 결과 요약

| 모델 | Accuracy | Macro-F1 | 비고 |
|---|---|---|---|
| RandomForest (사전 검증) | 0.783 | 0.647 | 26x26, 파이프라인 검증용 |
| **Keras CNN** | **0.918** | **0.879** | 1.2M 파라미터, 증강+가중치, 30 epoch |
| HuggingFace ViT | 0.737 | 0.668 | ImageNet-21k 사전학습 파인튜닝, 3 epoch |

### 클래스별 F1

| 클래스 | CNN F1 | ViT F1 |
|---|---|---|
| Center | 0.927 | 0.802 |
| Donut | 0.806 | 0.451 |
| Edge-Loc | 0.868 | 0.687 |
| Edge-Ring | 0.982 | 0.959 |
| Loc | 0.777 | 0.473 |
| Near-full | 0.939 | 0.857 |
| Random | 0.885 | 0.780 |
| Scratch | 0.781 | 0.224 |
| none | 0.946 | 0.780 |

![모델 비교](outputs/figures/model_comparison.png)
![CNN Confusion Matrix](outputs/figures/confusion_cnn.png)
![ViT Confusion Matrix](outputs/figures/confusion_vit.png)
![오분류 사례](outputs/figures/misclassified_examples.png)
![Grad-CAM](outputs/figures/gradcam_grid.png)

### 분석

- **CNN이 ViT를 모든 지표에서 상회**(accuracy +18.1%p, macro-F1 +21.1%p). 웨이퍼 맵은 0/1/2 세 값만 갖는 희소한 그리드 패턴으로, 지역적 합성곱 연산(local convolution) 기반 CNN이 전역 어텐션(global attention) 기반 ViT보다 이런 저수준 공간 패턴을 포착하는 데 더 적합했습니다.
- **소수 클래스에서 격차가 극대화**: Scratch(1,193장, ΔF1 0.56), Donut(555장, ΔF1 0.36)처럼 원본 샘플이 적은 클래스일수록 ViT의 성능 저하가 뚜렷합니다. 8600만 파라미터급 사전학습 모델도 도메인이 크게 다른 소량 데이터(26,961장)로 3 epoch만 파인튜닝하면 충분히 수렴하지 못한다는 것을 보여줍니다.
- **Edge-Ring은 두 모델 모두 최고 성능**(CNN 0.982 / ViT 0.959) — 링 형태가 이미지 전체에 걸쳐 뚜렷하게 나타나는 패턴이라 두 아키텍처 모두 잘 학습했습니다.
- **혼동 패턴**: Loc↔Edge-Loc, Scratch↔Loc은 공간적 정의가 겹쳐 사람도 판단이 갈리는 구간 → `misclassified_examples.png`에서 고신뢰 오분류 사례 확인
- **Grad-CAM**: 모델이 실제 불량 다이 군집 위치를 근거로 판단하는지 검증 → 품질 담당자에게 "왜 이 판정인지" 설명 가능
- **결론**: 이미지 도메인이 자연 이미지와 크게 다르고 데이터가 제한적인 상황에서는, 사전학습된 대형 트랜스포머보다 태스크에 맞게 설계한 경량 CNN이 더 실용적인 선택이 될 수 있습니다. 실제 배포 모델은 **CNN을 채택**했습니다.

### 품질 진단 관점의 분석

불량 유형별 발생 분포를 분석하여 빈도가 높은 패턴을 우선적으로 확인할 수 있도록 구성했습니다. 또한 클래스별 F1과 오분류 사례를 함께 확인하여 단순 정확도뿐 아니라 특정 불량 유형에 대한 분류 성능을 평가했습니다.

Grad-CAM을 활용하여 모델이 불량 패턴의 어느 영역을 근거로 판단했는지 확인하고, 분류 결과를 품질 담당자가 검토할 수 있도록 설명 가능성을 보완했습니다.

다만 WM-811K 데이터에는 장비·공정조건 등의 정보가 포함되어 있지 않으므로, 본 프로젝트에서 제시하는 원인은 실제 원인 규명이 아닌 **공정 관점의 원인 후보**로 정의했습니다. 실제 원인 분석을 위해서는 장비 ID, 공정 조건, 시간별 발생 이력 등의 추가 데이터가 필요합니다.

## 반도체 품질관리 활용 가능성

### 1. 불량 패턴 기반 점검 우선순위 지원

불량 패턴을 자동 분류하고 유형별 발생 현황을 집계하여 빈도가 높거나 특정 시점에 증가하는 패턴을 우선적으로 확인할 수 있습니다. 패턴별 대표적인 원인 후보를 함께 제공하여 후속 공정·장비 점검의 참고자료로 활용할 수 있습니다.

### 2. 검사 자동화

웨이퍼 맵의 불량 패턴을 CNN 모델로 자동 분류하여 반복적인 판독 작업을 지원하고, 불량 유형별 분류 결과를 일관된 기준으로 관리할 수 있습니다.

### 3. 품질 현황 모니터링

불량 유형별 발생 분포와 모델의 분류 결과를 대시보드에서 시각화하여 품질 상태를 한눈에 확인할 수 있도록 구성했습니다.

### 4. 설명 가능 AI(XAI)

Grad-CAM을 활용하여 모델이 예측에 활용한 영역을 시각화하고, 단순한 분류 결과뿐 아니라 모델의 판단 근거를 함께 확인할 수 있도록 구현했습니다.

### 5. 모델 성능 기반 의사결정

CNN과 ViT의 전체 성능뿐 아니라 클래스별 F1과 오분류 사례를 비교하여 모델을 선택했습니다. 자연 이미지와 다른 웨이퍼 맵의 데이터 특성과 학습 데이터 규모를 고려했을 때, 본 프로젝트에서는 CNN을 실제 배포 모델로 선택했습니다.

### 6. 실제 품질 분석으로의 확장

현재 WM-811K는 웨이퍼 맵과 불량 패턴 라벨 중심의 데이터이므로 실제 공정 원인 분석에는 한계가 있습니다. 실제 제조 환경에서는 다음과 같은 데이터를 추가로 결합하여 분석 범위를 확장할 수 있습니다.

- 장비 ID / Chamber ID
- 공정 단계
- 온도·압력·시간 등 공정 조건
- Lot / Wafer ID
- 검사 시점
- 불량 발생 이력
- 수율 및 재작업 여부

이러한 데이터를 웨이퍼 맵 패턴과 연결하면 다음과 같은 품질 분석 파이프라인으로 확장할 수 있습니다.

**불량 패턴 분류 → 발생 추이 분석 → 원인 후보 도출 → 공정·장비 조건 검증 → 개선 효과 확인**

현재 프로젝트에서는 공개 데이터의 한계를 고려하여 실제 원인 규명보다는 **불량 패턴 분류 및 품질 진단을 위한 분석 기반 구축**에 초점을 맞췄습니다.

## 프로젝트 구조

```
wafer-defect-classification/
├── README.md
├── requirements.txt
├── data/
│   ├── wm811k_64.npz          # 전처리 완료 데이터 (9.4MB)
│   └── class_names.json
├── src/
│   ├── prepare_data.py        # LSWMD.pkl → npz (레거시 피클 호환 처리 포함)
│   ├── eda.py
│   ├── utils.py
│   ├── train_cnn.py           # Keras CNN baseline
│   ├── train_vit.py           # HuggingFace ViT fine-tuning
│   ├── evaluate.py            # 비교 평가 + 오분류 분석
│   └── gradcam.py             # 설명 가능 AI
├── app/
│   └── streamlit_app.py       # 품질 진단 대시보드
├── notebooks/
│   └── colab_train.ipynb      # T4 GPU 학습 노트북
├── models/
│   ├── cnn_best.keras
│   └── vit_best/
└── outputs/
    ├── model_comparison.md
    └── figures/
```
## 프로젝트 한계 및 향후 개선

### 현재 한계

- WM-811K 데이터는 웨이퍼 맵과 불량 패턴 라벨 중심이므로 실제 공정 조건 및 장비 정보와 직접 연결할 수 없습니다.
- 따라서 본 프로젝트의 원인 분석 결과는 실제 원인 규명이 아닌 공정 관점의 원인 후보 제시에 해당합니다.
- 공개 데이터의 특성상 실제 제조 라인의 시간 순서에 따른 불량 발생 추이나 수율 변화까지 분석하는 데 한계가 있습니다.

### 향후 개선

실제 제조 환경의 공정 데이터를 확보할 경우 웨이퍼 맵 불량 패턴과 장비·공정 조건을 결합하여 다음과 같이 확장할 수 있습니다.

1. 불량 패턴별 발생 빈도 및 추이 분석
2. 장비·공정 조건별 불량 발생률 비교
3. 주요 원인 후보 도출
4. 공정 조건과 불량 발생 관계 검증
5. 개선 전·후 수율 및 불량률 비교

이를 통해 현재의 **불량 패턴 분류 시스템을 실제 품질 원인 분석 및 공정 개선 지원 시스템으로 확장**하는 것을 목표로 합니다.
