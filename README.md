# 📚 Exam Score Prediction

A machine learning project that predicts a student's exam score based on their study habits, attendance, sleep, and other academic factors.

The project includes data analysis, preprocessing, model comparison, hyperparameter tuning, and a Streamlit web application for making predictions.

## 🚀 Live Demo

👉 **[Try the Exam Score Prediction App](https://sujalnegi12-marks-prediction-main-k3wotz.streamlit.app/)**

Enter the required student details and get the predicted exam score instantly.

---

## 📊 Dataset

The dataset contains **20,000 student records** with 13 features.

### Numerical Features

- `study_hours`
- `class_attendance`
- `sleep_hours`
- `age`

### Categorical Features

- `gender`
- `course`
- `internet_access`
- `sleep_quality`
- `study_method`
- `facility_rating`
- `exam_difficulty`

### Target Variable

- `exam_score`

The target represents the student's final exam score between **0 and 100**.

---

## 🔍 Exploratory Data Analysis

I performed Exploratory Data Analysis to understand the relationship between different factors and exam scores.

Some important observations were:

- `study_hours` had the strongest positive correlation with `exam_score` (**0.72**).
- `class_attendance` also showed a positive correlation (**0.31**).
- One-Way ANOVA was used to check the effect of categorical features.
- `sleep_quality`, `study_method`, and `facility_rating` showed a significant relationship with exam scores.
- `age` and `gender` were not statistically significant in the analysis.

---

## ⚙️ Data Preprocessing

After analyzing the dataset, the following features were removed:

- `student_id`
- `age`
- `gender`
- `internet_access`
- `exam_difficulty`
- `course`

The following categorical features were selected:

- `sleep_quality`
- `study_method`
- `facility_rating`

These categorical features were converted into numerical values using `LabelEncoder`.

The trained encoders were saved using Pickle so the same encoding can be used when making predictions in the Streamlit application.

---

## 🤖 Model Comparison

I trained and compared three regression models:

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Decision Tree Regressor | 11.7722 | 218.0949 | 14.7680 | 0.3903 |
| AdaBoost Regressor | 8.8908 | 118.1829 | 10.8712 | 0.6696 |
| **Tuned XGBoost Regressor** | **7.9583** | **97.1228** | **9.8551** | **0.7285** |

The **Tuned XGBoost Regressor** performed the best among the tested models.

---

## 🔧 Hyperparameter Tuning

`GridSearchCV` was used to find suitable hyperparameters for the XGBoost model.

The selected parameters were:

```python
{
    'colsample_bytree': 0.7,
    'gamma': 0,
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 200,
    'subsample': 0.7
}