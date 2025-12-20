
# Credit Card Fraud Detection (2025-1 Machine Learning Team Project)

## Project Overview

**2025-1학기 기계학습 팀 프로젝트 (1조)** 결과물입니다.
유럽 카드 소지자들의 신용카드 거래 내역 데이터를 활용하여, 정상 거래와 사기 거래(Fraud)를 구분하는 **이진 분류(Binary Classification)** 모델을 구축했습니다.

사기 거래가 전체의 **0.172%** 에 불과한 극심한 **클래스 불균형(Imbalanced Data)** 문제를 해결하고, 실제 금융 시스템에서 중요한 '높은 감지율(Recall)'과 '정확성(Precision)'의 균형을 맞추는 데 집중했습니다.

* **Dataset:** Kaggle Credit Card Fraud Detection Dataset (September 2013)
* **Key Challenge:** 0.172%의 희소한 사기 데이터를 효과적으로 학습시키는 것.
* **Goal:** 오탐(False Positive)을 최소화하며 **AUPRC(Area Under Precision-Recall Curve)** 를 최대화.

## Team Members

* **강나언, 서동주, 이정연, 이현석**

## Tech Stack & Methodology

### 1. Preprocessing Pipeline

데이터 불균형 해결과 피처 엔지니어링을 위해 다음과 같은 전처리 과정을 거쳤습니다.

* **Scaling:**
* `Amount`: 데이터 분포의 치우침을 줄이기 위해 **Log Transformation** 적용.
* `Time`: 주기성을 반영하기 위해 **Sine Wave** 변환 적용.


* **Dimensionality Reduction:** 데이터셋에서 제공된 PCA Features (V1 ~ V28) 활용.
* **Oversampling (Key Strategy):**
* 단순 복제가 아닌 데이터를 합성하는 **SMOTE (Synthetic Minority Over-sampling Technique)** 사용.
* **Optimization:** 원본 비율(1:462)을 **1:10** 비율로 조정하여 과적합(Overfitting) 방지 및 성능 최적화.



### 2. Libraries

* `Python`, `Pandas`, `NumPy` (Data Manipulation)
* `Scikit-learn` (Modeling)
* `Imbalanced-learn` (SMOTE)
* `Matplotlib`, `Seaborn` (Visualization)

## Modeling & Evaluation

### Tested Models

다양한 알고리즘을 적용하여 성능을 비교 분석했습니다.

* **Linear:** Logistic Regression (L1/L2 Regularization)
* **Tree-based:** Decision Tree, RandomForest
* **Distance-based:** K-Nearest Neighbors (KNN)
* **Neural Network:** MLP Classifier
* **Ensemble:** Voting, Bagging, AdaBoost

### Best Model: RandomForest Classifier

여러 모델 중 **RandomForest**가 가장 우수한 성능을 보였습니다. 특히 `Class Weight`를 조절하여 소수 클래스(Fraud)에 가중치를 부여한 것이 유효했습니다.

* **Hyperparameters:**
* `n_estimators`: 100
* `criterion`: 'entropy'
* `class_weight`: 'balanced'
* `random_state`: 42



### Final Results

AUPRC(Area Under Precision-Recall Curve)를 주 지표로 설정하여 평가했습니다.

| Metric | Base Model (Original) | **Final Model (Optimized)** | Improvement |
| --- | --- | --- | --- |
| **AUPRC** | 0.8582 | **0.8807** | **+2.6%** 🔺 |
| Precision | - | **0.85** | Balanced |
| Recall | - | **0.85** | Balanced |
| F1-Score | - | **0.85** | High Stability |

> **Conclusion:** 초기 원본 데이터 모델 대비, **SMOTE 비율(1:10)** 조정 및 **Balanced Weight** 파라미터 튜닝을 통해 최종 AUPRC 점수를 **0.8807**로 향상시켰습니다.

## File Structure

```bash
2025-ML-Fraud-Detection/
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb  # 탐색적 데이터 분석 및 전처리
│   ├── 02_Modeling_Comparison.ipynb    # 모델 별 성능 비교
│   └── 03_Final_Model_Evaluation.ipynb # 최종 모델 튜닝 및 결과
├── presentations/
│   └── Team1_Project_Presentation.pdf  # 발표 자료
├── data/
│   └── (Data files not included due to size)
├── README.md
└── requirements.txt

```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Jupyter Notebook
jupyter notebook

```

---
