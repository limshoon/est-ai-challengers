# Day 13. Keras 딥러닝과 PyTorch 데이터 처리

> 작성일: 2026-07-30

## 📝 오늘 배운 내용 요약
1. **문제 유형에 맞는 손실함수와 Optimizer를 학습했다**
  - 손실함수는 예측값과 실제값의 차이를 하나의 수치로 나타내며, 모델은 이 값을 줄이는 방향으로 가중치를 조정한다.
  - 회귀에는 MSE, 이진 분류에는 Binary Cross-Entropy, 다중 분류에는 Categorical Cross-Entropy를 사용한다.
  - SGD는 고정된 학습률로 단순하게 가중치를 갱신하고, Adam은 과거 기울기와 적응적 학습률을 활용해 비교적 빠르고 안정적으로 수렴한다.
  - 학습률이 지나치게 크면 손실이 발산하고 너무 작으면 학습이 느려지므로 적절한 값의 설정이 중요하다.
2. **Keras Sequential API로 신경망을 구성하고 학습했다**
  - `keras.Input(shape=)`으로 입력 형태를 지정하고 `Dense` 층을 순서대로 쌓아 완전연결 신경망을 구성했다.
  - `compile()`에서 Optimizer, 손실함수와 평가지표를 설정하고 `fit()`에서 epoch, batch size와 검증 데이터 비율을 지정했다.
  - 이진 분류는 출력 뉴런 1개와 Sigmoid, 다중 분류는 클래스 수만큼의 출력 뉴런과 Softmax를 사용해야 한다.
  - `summary()`에서 층별 출력 형태와 학습 가능한 파라미터 수를 확인했다.
3. **학습 이력을 평가하고 Callback으로 과대적합을 관리했다**
  - `evaluate()`로 테스트 데이터의 최종 손실과 정확도를 확인하고, `history.history`의 훈련·검증 손실과 정확도를 그래프로 비교했다.
  - 훈련 손실은 감소하지만 검증 손실이 다시 증가하면 과대적합이 시작된 것으로 판단할 수 있다.
  - `EarlyStopping`은 검증 성능이 일정 기간 개선되지 않을 때 학습을 중단하고, `ModelCheckpoint`는 가장 좋은 모델을 저장한다.
  - `restore_best_weights=True`를 사용해 마지막 epoch가 아니라 검증 성능이 가장 좋았던 시점의 가중치를 복원했다.
4. **Dropout과 BatchNormalization의 역할을 비교했다**
  - Dropout은 학습 중 일부 뉴런을 무작위로 비활성화해 특정 뉴런 조합에 대한 의존과 과대적합을 줄인다.
  - BatchNormalization은 층 사이의 값 분포를 정규화해 깊은 신경망의 학습을 안정시키고 수렴을 빠르게 만든다.
  - 일반적으로 `Dense → BatchNormalization → Activation → Dropout` 순서로 배치한다.
  - Dropout 비율이 지나치게 높으면 과소적합이 발생할 수 있으므로 훈련·검증 성능을 함께 비교해야 한다.
5. **PyTorch Tensor와 자동 미분의 기초를 학습했다**
  - Tensor는 NumPy 배열과 문법이 유사하지만 GPU 연산과 자동 미분을 지원하는 PyTorch의 기본 데이터 단위이다.
  - `.shape`, `.dtype`, `.device`로 속성을 확인하고 `torch.from_numpy()`와 `.numpy()`로 상호 변환했다.
  - `requires_grad=True`로 연산 기록을 추적한 뒤 `.backward()`를 호출해 기울기를 자동으로 계산했다.
  - NumPy에서 변환한 Tensor는 원본 배열과 메모리를 공유할 수 있다는 점에 주의해야 한다.
6. **Dataset과 DataLoader로 미니배치 데이터 공급 구조를 만들었다**
  - 커스텀 `Dataset`에서 `__len__()`과 `__getitem__()`을 구현해 데이터의 개수와 개별 샘플 반환 방식을 정의했다.
  - `DataLoader`는 데이터를 batch 단위로 나누어 학습 루프에 공급하며, 대용량 데이터를 한 번에 메모리에 올리지 않아도 되게 한다.
  - 일반적으로 훈련 데이터는 `shuffle=True`, 검증·테스트 데이터는 `shuffle=False`로 설정한다.
  - batch size에 따라 한 epoch의 배치 수, 메모리 사용량과 학습 안정성이 달라진다는 점을 확인했다.
## 💭 오늘의 회고
1. **배운 점**
  - 딥러닝은 모델 구조만 설계하는 것이 아니라 손실함수, Optimizer, 데이터 공급, 평가와 과대적합 방지까지 하나의 흐름으로 구성해야 한다.
  - Keras는 `compile()`과 `fit()`으로 학습 과정을 간결하게 제공하고, PyTorch는 Tensor와 DataLoader를 통해 내부 과정을 더 명시적으로 다룬다는 차이를 이해했다.
  - 학습 곡선과 Callback을 함께 활용하면 무작정 epoch 수를 늘리는 것보다 효율적으로 최적 모델을 선택할 수 있다.
2. **어려운 점/개선할 점**
  - 이진·다중 분류에 따라 출력층 활성화 함수와 손실함수 조합이 달라지는 부분을 정확히 구분할 필요가 있다.
  - Dropout, BatchNormalization과 EarlyStopping이 모두 과대적합 관리에 사용되지만 작동 방식이 서로 달라 혼동하기 쉬웠다.
  - Keras의 자동화된 학습 흐름과 PyTorch의 명시적인 Tensor·DataLoader 구조를 연결해 이해하는 연습이 필요하다.
3. **액션 플랜**
  - Breast Cancer 데이터로 Keras 이진 분류 모델을 구성하고 EarlyStopping과 ModelCheckpoint를 함께 적용한다.
  - Dropout 적용 전후의 훈련·검증 손실 곡선을 비교해 과대적합 완화 효과를 확인한다.
  - PyTorch로 커스텀 Dataset과 DataLoader를 만들고 미니배치의 shape과 데이터 순서를 직접 출력해 본다.
4. **함께 나누고 싶은 점**
  - 딥러닝 성능은 층을 많이 쌓는 것만으로 좋아지지 않으며, 문제에 맞는 손실함수와 안정적인 데이터·평가 파이프라인이 함께 갖춰져야 한다.
## 📚 참고자료
- AI Challengers_교안 백업 - CH 14 딥러닝 기초 Keras
- Keras Sequential Model, Callback, BatchNormalization과 Dropout 실습
- PyTorch Tensor, Dataset과 DataLoader 실습
## 🔍 내일 학습 예정
- `nn.Module`과 PyTorch 학습 루프
- FashionMNIST 이미지 분류 미니 프로젝트
- CNN, RNN과 Transformer 구조
- GPU 활용과 NLP 기초
