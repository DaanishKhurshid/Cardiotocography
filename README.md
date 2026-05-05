# 👶 Fetal Health Classification from Cardiotocographic Data

![Streamlit App](https://streamlit.io) 
*Deployment status: Live on Streamlit Cloud*

## 📌 Project Overview
This project uses **Machine Learning** to classify fetal health into three states: **Normal**, **Suspect**, and **Pathological**. By analyzing 2,126 records of Cardiotocogram (CTG) exams, I have built a diagnostic tool to help healthcare professionals detect fetal distress early and reduce maternal/child mortality.

## 🚀 Key Features
- **High Reliability**: Achieved **94% accuracy** using a Random Forest Classifier.
- **Clinically Focused**: Optimized for **Pathological Recall (93%)** to ensure high-risk cases are rarely missed.
- **Interactive Web App**: A live Streamlit interface where users can input clinical measurements for instant diagnostics.
- **Robust Validation**: Proven stability via **10-Fold Stratified Cross-Validation**.

## 📊 Dataset Summary
- **Source**: UCI Machine Learning Repository / Kaggle.
- **Instances**: 2,126 patient records.
- **Features**: 21 clinical attributes (Baseline HR, Accelerations, Uterine Contractions, etc.).
- **Target**: `NSP` (1: Normal, 2: Suspect, 3: Pathological).

## 🛠️ Tech Stack & Workflow
- **Languages**: Python (Pandas, NumPy, Matplotlib, Seaborn).
- **Machine Learning**: Scikit-Learn (Random Forest, SVM, KNN, Logistic Regression).
- **Tuning**: RandomizedSearchCV for hyperparameter optimization.
- **Deployment**: Streamlit Cloud & Joblib for model serialization.

## 📈 Model Performance (80/20 Split)

| Model | Accuracy | Pathologic Recall | Pathologic Precision |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **94%** | **93%** | **96%** |
| KNN | 92% | 79% | 82% |
| SVM | 88% | 90% | 74% |
| Logistic Regression | 87% | 76% | 65% |

## 💻 Installation & Usage
1. **Clone the repo**:
   `git clone https://github.com`
2. **Install dependencies**:
   `pip install -r requirements.txt`
3. **Run the app**:
   `streamlit run app.py`

## 📁 File Structure
- `app.py`: Streamlit web application code.
- `rf_best_model.pkl`: Serialized winning Random Forest model.
- `ctg_scaler.pkl`: Fitted StandardScaler for input transformation.
- `classification_report.json`: Detailed performance metrics for auditing.
- `requirements.txt`: List of required libraries for deployment.
