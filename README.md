# 🧩 6x6 Sudoku Solver with Transformer (Refactored)

Transformer 기반의 **6x6 스도쿠 풀이 AI 프로젝트**입니다.
단순히 숫자를 나열하는 방식이 아닌, 스도쿠의 **구조적 정보(행, 열, 박스)**를 이해하는 독자적인 임베딩 방식을 적용하여 **99.9% 이상의 정답률**을 달성했습니다.

최근 **리팩토링(Refactoring)** 을 통해 `Config` 기반의 중앙 집중식 설정 관리 시스템을 도입하여, 추후 9x9 등 더 큰 스도쿠로 쉽게 확장할 수 있도록 구조를 개선했습니다.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Refactored-blueviolet)

## 📌 주요 특징 (Key Features)

1. **구조적 임베딩 (Structural Embeddings)**
   - 기존의 단순 위치 인코딩(Positional Encoding)을 대체하여, 각 셀의 **행(Row), 열(Col), 박스(Box)** 정보를 결합한 임베딩을 사용합니다.
   - 이를 통해 모델이 스도쿠의 규칙을 명확하게 학습합니다.

2. **검증된 데이터 생성 (Validated Data Generation)**
   - 무작위 생성이 아닌, **'유일한 해(Unique Solution)'**를 가진 문제만 엄선하는 검증 로직(Solver)이 포함되어 있습니다.
   - 정답이 여러 개인 불량 데이터를 원천 차단하여 AI의 학습 혼란을 방지했습니다.

3. **고성능 학습 최적화 & 시각화**
   - `AdamW` 옵티마이저와 `CosineAnnealingLR` 스케줄러를 적용하여 학습 후반부의 정체 구간 없이 완벽한 수렴을 유도합니다.
   - `tqdm` 라이브러리를 도입하여 학습 및 데이터 생성 진행 상황을 실시간 진행 바(Progress Bar)로 제공합니다.

4. **유연한 설정 관리 (Centralized Config)**
   - `src/config.py` 파일 하나로 모델 크기, 학습 파라미터, 경로 등을 통합 관리합니다.
   - 하드코딩을 제거하여 9x9 스도쿠 등 다양한 규격으로 쉽게 확장할 수 있는 유연한 아키텍처를 갖췄습니다.

## 📂 디렉토리 구조 (Structure)

```text
sudoku-transformer-6x6/
├── data/                        # 데이터 저장소
│   └── processed/               # 생성된 학습용 데이터 (.pt 파일)
├── saved_models/                # 학습된 모델 가중치 저장소
├── src/
│   ├── data/
│   │   ├── generator.py         # 스도쿠 생성, 검증(Solver) 로직
│   │   └── dataset.py           # PyTorch Dataset 클래스
│   ├── model/
│   │   └── transformer.py       # Config 기반 일반화된 Transformer 모델
│   ├── config.py                # [NEW] 하이퍼파라미터 통합 설정 파일
│   └── utils.py                 # 유틸리티 함수 (시드 고정, 정확도 계산 등)
├── generate_data.py             # [실행 1] 데이터 생성 스크립트
├── train.py                     # [실행 2] 모델 학습 스크립트
├── inference.py                 # [실행 3] 추론 및 테스트 스크립트
├── requirements.txt             # 의존성 목록
└── README.md                    # 프로젝트 설명서
````

## 🚀 설치 및 실행 순서 (Installation & Usage)

이 프로젝트는 Python 3.10 및 PyTorch 환경에서 테스트되었습니다. (RTX 3060 기준)

### 1\. 환경 설정 (Installation)

먼저 레포지토리를 클론하고 필수 라이브러리를 설치합니다.

```bash
# 레포지토리 다운로드
git clone [본인의_깃허브_레포지토리_주소]
cd sudoku-transformer-6x6

# 라이브러리 설치
pip install -r requirements.txt
```

*(권장: GPU 가속을 위해 본인의 그래픽 카드에 맞는 CUDA 버전의 PyTorch를 설치하세요.)*

### 2\. 설정 확인 (Optional)

`src/config.py` 파일에서 학습 파라미터나 난이도 설정을 변경할 수 있습니다.

  - 기본값: 6x6 그리드, 배치 사이즈 256, 에폭 30

### 3\. 데이터 생성 (Data Generation)

가장 먼저 데이터를 생성해야 합니다. 검증 로직을 거친 고품질 데이터 50만 개를 생성합니다.

```bash
python generate_data.py
```

  - **결과:** `data/processed/` 폴더에 `train.pt`, `val.pt` 파일이 생성됩니다.
  - *소요 시간: 약 5\~10분 (검증 로직 포함)*

### 4\. 모델 학습 (Training)

생성된 데이터를 바탕으로 AI를 학습시킵니다.

```bash
python train.py
```

  - **결과:** `saved_models/` 폴더에 `best_model.pth`가 저장됩니다.
  - **성능:** Epoch 1\~3 내에 정확도 99%에 도달하며, 최종적으로 \*\*99.9%\*\*를 달성합니다.

### 5\. 테스트 및 추론 (Inference)

학습된 모델이 실제로 문제를 어떻게 푸는지 확인합니다.

```bash
python inference.py
```

  - 메뉴에서 \*\*1번(랜덤 문제)\*\*을 선택하여 AI의 풀이 실력을 검증할 수 있습니다.
  - 사용자가 직접 문제를 입력하여 풀게 할 수도 있습니다.

## 🧠 모델 아키텍처 정보 (Architecture)

| 항목 | 설정값 | 설명 |
| :--- | :--- | :--- |
| **Grid Size** | 6x6 | Config에서 변경 가능 (`GRID_SIZE`) |
| **Embedding** | 256 dim | Token + (Row + Col + Box) Embedding |
| **Layers** | 8 | Transformer Encoder Layers |
| **Heads** | 8 | Multi-head Attention |
| **Params** | \~6.5M | Lightweight yet powerful |

## 📝 License

This project is licensed under the MIT License.

```
```