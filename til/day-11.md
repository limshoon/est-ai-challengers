# Day 11. 분류 모델과 평가

> 작성일: 2026-07-28

## 📝 오늘 배운 내용 요약
1. **대표적인 분류 모델의 원리와 특징을 학습했다**
  - Logistic Regression은 시그모이드 함수를 통해 0\~1 사이의 확률을 계산하고 임계값에 따라 클래스를 예측한다.
  - Decision Tree Classifier는 조건 분기로 분류하지만 깊이가 커지면 과대적합될 수 있다.
  - Random Forest Classifier는 여러 트리의 다수결 투표로 단일 트리의 불안정성을 줄이고, XGBoost Classifier는 이전 모델의 오차를 순차적으로 보완한다.
  - `predict()`는 최종 클래스를, `predict_proba()`는 클래스별 예측 확률을 반환한다는 차이를 이해했다.
2. **혼동행렬과 분류 평가 지표를 학습했다**
  - 혼동행렬을 TP, TN, FP, FN으로 나누어 모델이 어떤 종류의 실수를 했는지 확인했다.
  - Precision은 양성이라고 예측한 것 중 실제 양성의 비율로 오탐을 줄이는 데 중요하다.
  - Recall은 실제 양성 중 모델이 찾아낸 비율로 누락을 줄이는 데 중요하며, F1-score는 Precision과 Recall의 균형을 조화평균으로 표현한다.
  - `classification_report()`로 클래스별 Precision, Recall, F1-score와 support를 한 번에 확인했다.
3. **ROC-AUC와 Precision-Recall Curve를 비교했다**
  - 분류 임계값을 바꾸면 Precision과 Recall, TPR과 FPR이 함께 달라진다는 점을 배웠다.
  - ROC Curve는 여러 임계값에서 FPR과 TPR의 관계를 나타내고 AUC로 전체 분류 성능을 요약한다.
  - Precision-Recall Curve는 Recall과 Precision의 관계를 보여 주며, 클래스 불균형이 큰 문제에서는 ROC-AUC보다 더 신뢰할 수 있는 경우가 많다.
  - 업무에서 오탐과 누락 중 어느 쪽의 비용이 큰지에 따라 임계값과 핵심 지표를 선택해야 한다.
4. **비지도학습과 군집화 방법을 학습했다**
  - K-Means는 중심점을 반복적으로 이동하며 데이터를 K개의 군집으로 나누는 알고리즘이다.
  - Elbow Method의 inertia와 Silhouette Score를 활용해 적절한 군집 개수를 찾는 방법을 학습했다.
  - PCA는 고차원 데이터를 2차원으로 축소해 군집을 시각화하는 데 활용하고, DBSCAN은 밀도 기반으로 군집과 노이즈를 구분한다.
  - 거리 기반 알고리즘인 K-Means를 적용하기 전에는 특성 스케일을 맞추는 것이 중요하다.
5. **Feature Importance와 다중 클래스 평가를 학습했다**
  - 트리 기반 모델의 내장 Feature Importance는 불순도 감소량을 기준으로 각 특성의 중요도를 계산한다.
  - Permutation Importance는 특정 특성 값을 섞었을 때 성능이 얼마나 감소하는지 측정해 다양한 모델에 적용할 수 있다.
  - 다중 클래스 `classification_report()`에서 macro avg는 클래스별 단순 평균, weighted avg는 클래스 크기를 반영한 가중 평균이라는 점을 배웠다.
  - 클래스 불균형이 있으면 전체 정확도뿐 아니라 클래스별 지표와 macro·weighted 평균을 함께 확인해야 한다.
## 💭 오늘의 회고
1. **배운 점**
  - 분류 모델의 성능은 정확도 하나로 판단할 수 없으며, 오류의 비용과 클래스 불균형을 고려해 Precision, Recall, F1-score와 곡선 지표를 함께 봐야 한다.
  - 모델의 예측 결과뿐 아니라 어떤 특성이 영향을 주었는지 설명하는 과정도 실무에서 중요하다는 점을 배웠다.
2. **어려운 점/개선할 점**
  - Precision과 Recall의 기준이 헷갈릴 수 있어 혼동행렬의 행과 열을 직접 그려가며 계산할 필요가 있다.
  - ROC Curve와 Precision-Recall Curve를 어떤 상황에서 선택해야 하는지 실제 불균형 데이터로 비교해봐야 한다.
3. **액션 플랜**
  - 동일한 분류 데이터셋에 Logistic Regression, Random Forest와 XGBoost를 적용하고 평가 지표를 비교한다.
  - 임계값을 여러 값으로 변경하면서 Precision과 Recall이 어떻게 변하는지 표와 그래프로 확인한다.
4. **함께 나누고 싶은 점**
  - 질병 진단처럼 놓치면 위험한 문제는 Recall, 스팸 차단처럼 정상 데이터를 잘못 막는 비용이 큰 문제는 Precision을 더 중요하게 볼 수 있다.
## 📚 참고자료
- AI Challengers_교안 백업 - CH 12 회귀와 분류
- scikit-learn classification metrics, clustering 및 inspection 문서
- 분류 모델 평가 및 ROC/PR Curve 실습 예제
## 🔍 내일 학습 예정
- 교차검증과 GridSearchCV
- 과소적합·과대적합 진단과 하이퍼파라미터 조정
- Pipeline, SHAP과 인공신경망 기초
