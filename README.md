## Marabou 기반 MLP 모델 검증 실험
이 프로젝트는 MIT-BIH Arrhythmia Dataset 기반으로 학습된 간단한 MLP 모델에 대해 Marabou 도구를 활용하여 **formal verification**을 수행합니다. 
ECG 시계열 입력에 대해 부정맥 예측 모델이 특정 입력 조건 하에서 얼마나 안정적이고 신뢰할 수 있는 출력을 내는지를 수학적으로 확인합니다.

`EndToEnd.ipynb` 파일을 실행하면 아래와 같은 폴더 구조를 갖게 됩니다.
## 프로젝트 구성

- 📁 `subject/`
  - 📄 `EndToEnd.ipynb` — 데이터 로딩, MLP 학습, ONNX 모델 저장, `mlp_input.txt` 제약 파일 생성
  - 📄 `pulse.onnx` — 학습된 ONNX 모델
  - 📄 `mlp_input.txt` — Marabou용 입력 제약 조건
  - 📄 `result.txt` — Marabou 실행 결과 로그

- 📁 `Marabou/`  
  - 모델 검증 엔진 (GitHub에서 직접 설치)
## 환경 준비
```bash
git clone https://github.com/Hmiru/ReliableAI.git
git checkout pulse
pip install requirements.txt
```
## Getting started
### 1. Marabou 설치
```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou
mkdir build 
cd build
cmake ..
cmake --build . -j$(nproc)
```
### 2. 모델 학습 및 ONNX 변환
- `EndToEnd.ipynb`의 Section 1을 실행하여 주어진 데이터셋에 대한 모델을 학습하고 이를 ONNX로 저장합니다. 
- Marabou용 입력 제약 조건.txt도 생성하여 저장합니다.
### 3. Marabou로 검증 실행
```bash
./Marabou ../../subject/pulse.onnx --property ../../subject/mlp_input.txt| tee ../../subject/result.txt
``` 
- `EndToEnd.ipynb`의 Section 2를 실행하여 Marabou 실행 결과 로그를 확인하고 부정맥과 생성된 입력의 그래프를 비교합니다.


## Environments information
```bash
python==3.9.21
torch==2.6.0
torchvision==0.21.0
```