# Day 14. PyTorch 모델링과 딥러닝 구조

> 작성일: 2026-07-31

## 📝 오늘 배운 내용 요약
1. **nn.Module을 활용해 PyTorch 신경망과 학습 루프를 구현했다**
  - `nn.Module`을 상속한 클래스의 `__init__()`에서 층을 정의하고 `forward()`에서 순전파 순서를 작성했다.
  - 손실함수는 `nn.CrossEntropyLoss()`, Optimizer는 `optim.Adam()`처럼 모델과 별도로 정의했다.
  - 학습 루프는 `zero_grad() → forward → loss 계산 → backward() → step()` 순서로 진행했다.
  - 학습 시 `model.train()`, 평가 시 `model.eval()`과 `torch.no_grad()`를 사용해야 Dropout과 기울기 추적이 올바르게 처리된다.
2. **모델 저장·복원과 학습 곡선 시각화의 표준 흐름을 확인했다**
  - 학습한 모델을 저장하고 다시 불러와 테스트 데이터로 평가하는 과정을 학습했다.
  - 훈련 손실과 검증 손실을 epoch별로 시각화해 과대적합이 시작되는 구간을 확인했다.
  - 저장 파일만 남기는 것이 아니라 모델 구조, 전처리 방식과 평가 결과를 함께 기록해야 재현성을 높일 수 있다.
3. **FashionMNIST로 이미지 분류 파이프라인을 구성했다**
  - `torchvision.datasets`에서 10종류의 의류 이미지로 구성된 FashionMNIST를 불러왔다.
  - Dataset을 DataLoader에 연결하고 이미지 Tensor를 `nn.Flatten()`으로 펼친 뒤 `nn.Linear` 층에 입력했다.
  - 데이터 준비, DataLoader, 모델 정의, Loss·Optimizer, 학습 루프와 평가로 이어지는 전체 흐름을 구현했다.
  - 이 표준 파이프라인은 이후 CNN과 같은 더 복잡한 모델에서도 동일한 틀로 확장할 수 있다.
4. **CNN이 이미지의 공간적 특징을 학습하는 원리를 이해했다**
  - CNN은 작은 필터가 이미지의 국소 영역을 이동하며 모서리와 질감 같은 특징을 추출한다.
  - `Conv2d`로 특징 지도를 만들고 `MaxPool2d`로 중요한 특징을 유지하면서 공간 크기를 줄인다.
  - Conv와 Pool을 반복한 뒤 Flatten과 Dense 층으로 최종 클래스를 분류한다.
  - 이미지의 공간 구조를 보존하므로 픽셀을 처음부터 펼치는 완전연결 모델보다 일반적으로 이미지 분류에 적합하다.
5. **RNN과 LSTM의 순차 데이터 처리 구조를 학습했다**
  - RNN은 은닉 상태를 통해 이전 시점의 정보를 다음 시점으로 전달하므로 문장과 시계열처럼 순서가 중요한 데이터에 적합하다.
  - `nn.RNN(input_size=, hidden_size=, batch_first=True)`의 입력 형태와 은닉 상태 출력을 확인했다.
  - 기본 RNN은 긴 문맥의 정보를 유지하기 어렵고, LSTM은 게이트 구조로 장기 의존성 문제를 완화한다.
  - 이미지의 공간 구조에는 CNN, 시간과 순서 구조에는 RNN·LSTM이 적합하다는 차이를 정리했다.
6. **Self-Attention과 Transformer의 핵심 원리를 학습했다**
  - Self-Attention은 각 토큰을 Query, Key와 Value로 변환하고 토큰 사이의 관련도를 계산해 문맥을 반영한다.
  - Query와 Key의 내적에 Softmax를 적용해 Attention 가중치를 구하고, 이를 Value에 반영한다.
  - Transformer는 Self-Attention과 Feed Forward 층을 쌓고 Positional Encoding으로 순서 정보를 보완한다.
  - RNN보다 병렬 처리가 쉽고 장거리 관계를 잘 포착해 현재 LLM의 기반 구조로 사용된다.
7. **CUDA를 이용한 GPU 학습 방법을 확인했다**
  - `torch.cuda.is_available()`로 GPU 사용 가능 여부를 확인하고 CPU 또는 CUDA 장치를 선택했다.
  - 모델과 입력·정답 Tensor를 모두 `.to(device)`로 같은 장치에 이동해야 연산할 수 있다.
  - GPU는 대량의 단순 연산을 병렬 처리해 큰 신경망의 학습 시간을 단축하지만 작은 모델에서는 전송 비용 때문에 이점이 작을 수 있다.
8. **NLP의 주요 과제와 텍스트 수치화 필요성을 학습했다**
  - NLP는 텍스트 분류, 개체명 인식, 번역, 요약, 생성과 질의응답처럼 사람의 언어를 처리하는 기술이다.
  - 모델은 문자열을 직접 이해할 수 없으므로 텍스트를 토큰으로 나누고 숫자 벡터로 변환해야 한다.
  - 규칙을 사람이 직접 작성하던 방식에서 데이터로 언어 패턴을 학습하는 방식으로 발전해 왔다.
9. **토큰화와 불용어 처리 방법을 비교했다**
  - 토큰화는 문장을 모델이 처리할 수 있는 단위로 분리하는 과정이며 단어, 형태소 또는 부분단어 단위로 수행할 수 있다.
  - 불용어는 문서에서 자주 등장하지만 분석 목적에 따라 정보량이 낮은 단어로, 무조건 제거하기보다 과제 특성을 고려해야 한다.
  - 한국어는 조사와 어미가 결합하는 교착어이므로 형태소 분석이 중요하다.
  - BPE와 WordPiece 같은 부분단어 방식은 처음 보는 단어도 이미 학습한 조각으로 처리할 수 있어 GPT와 BERT 계열 모델에 사용된다.
## 💭 오늘의 회고
1. **배운 점**
  - PyTorch에서는 자동화된 `fit()` 대신 순전파, 손실 계산, 역전파와 가중치 갱신을 직접 작성해 신경망 학습 원리를 더 구체적으로 확인할 수 있었다.
  - Dense, CNN, RNN과 Transformer는 데이터의 구조를 어떤 방식으로 반영하는지에 따라 구분된다는 점을 이해했다.
  - 이미지와 텍스트 데이터도 결국 Tensor와 DataLoader를 통해 동일한 학습 파이프라인으로 처리할 수 있다.
2. **어려운 점/개선할 점**
  - `zero_grad()`, `backward()`와 `step()`의 순서를 바꾸거나 생략하면 학습이 잘못될 수 있어 반복 연습이 필요하다.
  - CNN의 채널·가로·세로 차원 변화와 RNN의 batch·sequence·feature 차원을 추적하는 것이 어려웠다.
  - Self-Attention의 Query, Key와 Value가 실제 Tensor 연산에서 어떻게 연결되는지 더 자세히 확인할 필요가 있다.
3. **액션 플랜**
  - FashionMNIST에 완전연결 모델과 간단한 CNN을 각각 적용해 테스트 정확도와 학습 시간을 비교한다.
  - PyTorch 학습 루프를 함수로 분리하고 `model.train()`, `model.eval()`과 장치 이동 코드를 포함한 기본 템플릿을 만든다.
  - 짧은 문장을 토큰화한 뒤 RNN 입력 Tensor와 Attention 행렬의 shape을 단계별로 출력한다.
4. **함께 나누고 싶은 점**
  - 모델 구조를 선택할 때 최신 구조라는 이유만으로 Transformer를 사용하는 것보다 이미지의 공간성, 시계열의 순서성과 데이터 규모를 먼저 고려해야 한다.
## 📚 참고자료
- AI Challengers_교안 백업 - CH 15 딥러닝 PyTorch
- PyTorch `nn.Module`, 학습 루프, Dataset과 DataLoader 실습
- FashionMNIST, CNN, RNN, Transformer와 CUDA 학습 자료
- NLP 기초, 토큰화와 불용어 처리 자료
## 🔍 내일 학습 예정
- 임베딩과 텍스트 벡터화
- Transformer와 GPT 구조
- Hugging Face Pipeline
- Fine-tuning과 LoRA
