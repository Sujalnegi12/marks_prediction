# Exam Score Prediction

A Machine Learning project that predicts students' final exam scores based on their academic habits and lifestyle factors.

The project includes Exploratory Data Analysis (EDA), data preprocessing, model comparison, hyperparameter tuning, model saving, and a Streamlit web application for making predictions.

## 🚀 Live Demo

👉 [Open the Streamlit App](https://sujalnegi12-marks-prediction-main-k3wotz.streamlit.app/)

## 📂 Dataset

The dataset used in this project is `Exam_Score_Prediction.csv`.

It contains **20,000 student records** and **13 features** related to students' academic performance, study habits, and lifestyle.

### Features

**Numerical Features**
- `study_hours`
- `class_attendance`
- `sleep_hours`
- `age`

**Categorical Features**
- `gender`
- `course`
- `internet_access`
- `sleep_quality`
- `study_method`
- `facility_rating`
- `exam_difficulty`

**Target Variable**
- `exam_score` — final exam score ranging from 0 to 100

## 📊 Exploratory Data Analysis

During EDA, different features were analyzed to understand their relationship with exam scores.

Some important findings:

- `study_hours` showed a strong positive relationship with exam score with a correlation of approximately **0.72**.
- `class_attendance` showed a positive relationship with exam score with a correlation of approximately **0.31**.
- One-Way ANOVA showed that `sleep_quality`, `study_method`, and `facility_rating` were statistically significant.
- `age` and `gender` did not show a significant relationship with the target variable.

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed `student_id` because it does not provide useful information for prediction.
- Removed `age`, `gender`, `internet_access`, `exam_difficulty`, and `course` based on the analysis.
- Applied `LabelEncoder` to:
  - `sleep_quality`
  - `study_method`
  - `facility_rating`
- Split the dataset into training and testing sets.
- Trained multiple regression models and compared their performance.

## 🤖 Models Used

The following models were compared:

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Decision Tree | 11.7722 | 218.0949 | 14.7680 | 0.3903 |
| AdaBoost | 8.8908 | 118.1829 | 10.8712 | 0.6696 |
| Tuned XGBoost | 7.9583 | 97.1228 | 9.8551 | **0.7285** |

### 🏆 Best Model

The **Tuned XGBoost Regressor** performed the best among the tested models.

Final R² Score:

**0.7285**

This means the model explains approximately **72.85% of the variation** in the exam scores in the test data.

## ⚙️ Hyperparameter Tuning

Grid Search was used to find better XGBoost hyperparameters.

The final model used:

```text
colsample_bytree = 0.7
gamma = 0
learning_rate = 0.05
max_depth = 3
n_estimators = 200
subsample = 0.7
```

## 🖥️ Streamlit Application

The trained model was integrated into a Streamlit web application.

The application allows users to enter student-related information and get a predicted exam score.

The prediction workflow is:

```text
User Input
    ↓
Data Preprocessing
    ↓
Label Encoding
    ↓
Trained XGBoost Model
    ↓
Predicted Exam Score
```

## 📁 Project Structure

```text
Marks_Prediction/
│
├── Exam_Score_Prediction.csv
├── Exam_Score_Prediction.ipynb
├── final_xgb_model.pkl
├── label_encoders.pkl
├── main.py
├── requirements.txt
└── README.md
```

## 🔗 Project Files

- 📊 [Dataset](Exam_Score_Prediction.csv)
- 📓 [Machine Learning Notebook](Exam_Score_Prediction.ipynb)
- 🤖 [Trained XGBoost Model](final_xgb_model.pkl)
- 🔤 [Label Encoders](label_encoders.pkl)
- 🌐 [Streamlit Application](main.py)
- 📦 [Requirements](requirements.txt)

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook
- Pickle

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/SujalNegi12/marks_prediction.git
cd marks_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run main.py
```

The application will open in your browser.

## 📌 Project Highlights

- Performed complete EDA on 20,000 student records.
- Used correlation analysis and One-Way ANOVA for feature analysis.
- Compared multiple machine learning models.
- Used Grid Search for XGBoost hyperparameter tuning.
- Saved the trained model and label encoders using Pickle.
- Built a Streamlit application for real-time prediction.
- Deployed the application using Streamlit.

## 👨‍💻 Author

**Sujal Negi**

B.Tech Computer Science Student

---

⭐ If you find this project useful, feel free to star the repository.
