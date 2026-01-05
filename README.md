# **Customer Churn Prediction using Artificial Neural Networks (ANN)**

## **Project Overview**
Customer churn is a critical business problem in the banking industry, where retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds an **end-to-end Artificial Neural Network (ANN)** to predict customer churn using demographic, financial, and behavioral data.

Rather than optimizing solely for accuracy, the project focuses on **business-aligned churn detection**, prioritizing the identification of customers at risk of leaving—even at the cost of increased false positives.

---

## **Dataset Description**
- **Source:** Bank Churn Modelling Dataset  
- **Records:** 10,000 customers  
- **Target Variable:** `Exited`  
  - `1` → Customer churned  
  - `0` → Customer retained  

### **Key Features**
- Demographic: Geography, Gender, Age  
- Financial: CreditScore, Balance, EstimatedSalary  
- Behavioral: Tenure, NumOfProducts, IsActiveMember, HasCrCard  

### **Dropped Columns**
- `RowNumber`, `CustomerId`, `Surname`  
(Identifiers with no predictive value)

---

## **Data Preprocessing**
- Removed non-informative identifiers
- One-Hot Encoded categorical variables (`Geography`, `Gender`)
- Standardized numerical features using `StandardScaler`
- Ensured all preprocessing was fit **only on training data** to prevent leakage
- Final feature matrix: **11 input features**

---

## **Modeling Approach**
- **Model:** Feedforward Artificial Neural Network
- **Architecture:**  
  - Input Layer → Dense(32) → Dropout  
  - Dense(16) → Dropout  
  - Output Layer (Sigmoid)
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Loss Function:** Binary Cross-Entropy
- **Regularization:** Dropout to reduce overfitting

SGD was intentionally chosen to maintain transparent learning dynamics and to better understand convergence behavior.

---

## **Handling Class Imbalance**
The dataset is imbalanced (~80% non-churn vs ~20% churn).  
To address this:

- Applied **class weighting** to penalize churn misclassification
- Tuned the **decision threshold (0.3 instead of 0.5)** to improve recall
- Shifted evaluation focus from accuracy to **recall and business impact**

---

## **Model Performance (Final)**
After threshold tuning and class-weighting:

| Class | Precision | Recall |
|-----|----------|--------|
| No Churn (0) | 0.94 | 0.65 |
| Churn (1) | 0.37 | 0.84 |

- **Churn Recall:** ~84%  
- **Accuracy:** ~68%  

> This trade-off is intentional and appropriate for churn prediction, where missing a churner is more costly than flagging a false positive.

---

## **Evaluation Highlights**
- Confusion Matrix used to visualize false positives and false negatives
- Training vs validation curves show stable convergence
- Demonstrates how **threshold tuning dramatically changes business outcomes**

---

## **Key Insights**
- Optimizing for accuracy alone leads to poor churn detection
- Decision thresholds are as important as model architecture
- ANN performance must be evaluated in the context of **business costs**
- Metric trade-offs are unavoidable and must be explicitly managed

---

## **Limitations**
- Lower precision for churn predictions increases false positives
- Model performance is sensitive to threshold selection
- Limited interpretability compared to tree-based models
- No extensive hyperparameter tuning performed
- No benchmark comparison with gradient-boosting models

---

## **Future Improvements**
- Cost-based threshold optimization
- ROC-AUC–driven decision rules
- Model explainability using SHAP
- Comparison with XGBoost / LightGBM
- Production-ready preprocessing pipelines

---

## **Tools & Libraries**
- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- Keras / TensorFlow  

---

## **How to Run**

1. Clone the repository:
```bash
git clone https://github.com/PranayDomal/Customer-Churn-ANN.git
```

2. Navigate to the folder:
```bash
cd Customer-Churn-ANN
```

3. Run the notebook:
```bash
jupyter notebook sCustomer_Churn_ANN.ipynb
```

---

## **Author**

https://www.linkedin.com/in/pranay-domal-a641bb368/
