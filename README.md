"""# Exam Score Prediction & Deployment

This project aims to predict students' final exam scores based on their academic habits and lifestyle factors. It includes a comprehensive Exploratory Data Analysis (EDA) pipeline, multiple machine learning models, hyperparameter optimization using Grid Search, and a Streamlit web application for real-time predictions.

## 📊 Dataset Overview
The dataset (`Exam_Score_Prediction.csv`) contains **20,000 student records** with 13 features:
* **Numerical Features**: `study_hours`, `class_attendance`, `sleep_hours`, `age`.
* **Categorical Features**: `gender`, `course`, `internet_access`, `sleep_quality`, `study_method`, `facility_rating`, `exam_difficulty`.
* **Target Feature**: `exam_score` (continuous variable from 0 to 100).

## 🛠️ Project Pipeline

### 1. Exploratory Data Analysis (EDA)
* Analyzed target variable distribution (approximately normal).
* Found key correlations: **Study Hours** has the highest positive correlation (`0.72`) with Exam Scores, followed by **Class Attendance** (`0.31`).
* Conducted a **One-Way ANOVA** test which confirmed that features like `sleep_quality`, `study_method`, and `facility_rating` significantly impact final scores, while `age` and `gender` showed no statistical significance.

### 2. Feature Selection & Preprocessing
* Irrelevant and statistically non-significant features (`student_id`, `age`, `gender`, `internet_access`, `exam_difficulty`, `course`) were dropped.
* Selected categorical features (`sleep_quality`, `study_method`, `facility_rating`) were encoded using `LabelEncoder`.

### 3. Model Training & Tuning
We evaluated three regression models on an 80-20 train-test split:

| Model | MAE | MSE | RMSE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Regressor** | 11.7722 | 218.0949 | 14.7680 | 0.3903 |
| **AdaBoost Regressor** | 8.8908 | 118.1829 | 10.8712 | 0.6696 |
| **Tuned XGBoost Regressor** | **7.9583** | **97.1228** | **9.8551** | **0.7285** |

* **Hyperparameter Optimization**: Used `GridSearchCV` on the `XGBRegressor` to lock down optimal parameters: `{'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.7}`.

### 4. Serialization
* Saved the fitted `LabelEncoder` objects in a dictionary as `label_encoders.pkl`.
* Saved the final tuned XGBoost Model as `final_xgb_model.pkl`.

---

## 💻 Running the Streamlit App

We have built a simple interactive user interface using **Streamlit** to predict test scores on the fly.

### Requirements
Make sure you have the required packages installed:
```bash
pip install streamlit pandas numpy scikit-learn xgboost
```

### Project Directory Structure
Place your files as follows:
```text
├── main.py                 # Streamlit application file
├── final_xgb_model.pkl     # Exported model
├── label_encoders.pkl      # Exported LabelEncoders dictionary
└── README.md               # Project documentation
```

### Execution
To launch your application locally, run:
```bash
streamlit run main.py
```
"""