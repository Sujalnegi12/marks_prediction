import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("final_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(f"Error loading model: {e}")

    st.info(
        "Make sure final_xgb_model.pkl is present "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Exam Score Predictor")

st.write(
    "Enter the student's details below to predict "
    "the expected exam score."
)

st.divider()


# ============================================================
# STUDENT DETAILS
# ============================================================

st.subheader("📚 Student Details")


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )


with col2:

    class_attendance = st.number_input(
        "Class Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col3, col4 = st.columns(2)


with col3:

    sleep_hours = st.number_input(
        "Sleep Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )


with col4:

    sleep_quality = st.selectbox(
        "Sleep Quality",
        [
            "average",
            "good",
            "poor"
        ]
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col5, col6 = st.columns(2)


with col5:

    study_method = st.selectbox(
        "Study Method",
        [
            "coaching",
            "group study",
            "mixed",
            "online videos",
            "self-study"
        ]
    )


with col6:

    facility_rating = st.selectbox(
        "Facility Rating",
        [
            "high",
            "low",
            "medium"
        ]
    )


st.write("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🎯 Predict Exam Score",
    use_container_width=True
):

    try:

        # ====================================================
        # MANUAL LABEL ENCODING
        # ====================================================

        sleep_mapping = {
            "average": 0,
            "good": 1,
            "poor": 2
        }


        study_mapping = {
            "coaching": 0,
            "group study": 1,
            "mixed": 2,
            "online videos": 3,
            "self-study": 4
        }


        facility_mapping = {
            "high": 0,
            "low": 1,
            "medium": 2
        }


        # Convert text into numbers

        encoded_sleep = sleep_mapping[sleep_quality]

        encoded_method = study_mapping[study_method]

        encoded_facility = facility_mapping[facility_rating]


        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame({

            "study_hours": [study_hours],

            "class_attendance": [class_attendance],

            "sleep_hours": [sleep_hours],

            "sleep_quality": [encoded_sleep],

            "study_method": [encoded_method],

            "facility_rating": [encoded_facility]

        })


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(input_data)[0]


        # Keep score between 0 and 100

        prediction = np.clip(
            prediction,
            0,
            100
        )


        # ====================================================
        # GRADE
        # ====================================================

        if prediction >= 90:

            grade = "A+"
            message = "Excellent performance! 🌟"

        elif prediction >= 80:

            grade = "A"
            message = "Great performance! Keep it up! 👏"

        elif prediction >= 70:

            grade = "B"
            message = "Good performance! 👍"

        elif prediction >= 60:

            grade = "C"
            message = "Average performance. Keep improving! 📚"

        elif prediction >= 50:

            grade = "D"
            message = "More preparation is needed. 💪"

        else:

            grade = "F"
            message = "Try to improve your study routine. 📖"


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("📊 Prediction Result")


        st.metric(
            "Expected Exam Score",
            f"{prediction:.2f} / 100"
        )


        st.success(
            f"Grade: {grade} — {message}"
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader("📋 Student Summary")


        summary = pd.DataFrame({

            "Detail": [
                "Study Hours",
                "Attendance",
                "Sleep Hours",
                "Sleep Quality",
                "Study Method",
                "Facility Rating"
            ],

            "Value": [
                f"{study_hours} hours",
                f"{class_attendance}%",
                f"{sleep_hours} hours",
                sleep_quality,
                study_method,
                facility_rating
            ]

        })


        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Exam Score Prediction App | XGBoost Regression"
)