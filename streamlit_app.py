import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Exam Score Prediction")
st.write("Predict a student's exam score based on study habits, resources, and other factors.")

MODEL_FILES = ['xgb_model.pkl', 'onehot_columns.pkl', 'ordinal_mappings.pkl']
missing_files = [f for f in MODEL_FILES if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing model files: {', '.join(missing_files)}")
    st.info("Please make sure these files are in the same directory:")
    for file in missing_files:
        st.write(f"- `{file}`")
    st.stop()

try:
    xgb_model = joblib.load('xgb_model.pkl')
    onehot_cols = joblib.load('onehot_columns.pkl')
    ordinal_maps = joblib.load('ordinal_mappings.pkl')
    RMSE = 1.977
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
    previous_scores = st.number_input("Previous Exam Score", min_value=0, max_value=100, value=70)
    tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
    physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, value=2)
    peer_influence = st.selectbox("Peer Influence", ["Low", "Medium", "High"])

with col2:
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    gender = st.selectbox("Gender", ["Male", "Female"])

distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
binary_map = {'Yes': 1, 'No': 0}

input_dict = {
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Parental_Involvement': ordinal_map[parental_involvement],
    'Access_to_Resources': ordinal_map[access_to_resources],
    'Extracurricular_Activities': binary_map[extracurricular_activities],
    'Sleep_Hours': sleep_hours,
    'Previous_Scores': previous_scores,
    'Motivation_Level': ordinal_map[motivation_level],
    'Internet_Access': binary_map[internet_access],
    'Tutoring_Sessions': tutoring_sessions,
    'Family_Income': ordinal_map[family_income],
    'Peer_Influence': ordinal_map[peer_influence],
    'Physical_Activity': physical_activity,
    'Learning_Disabilities': binary_map[learning_disabilities],
    'Distance_from_Home': distance_map[distance_from_home],
    'Gender_Male': 1 if gender == 'Male' else 0,
    'School_Type_Public': 1 if school_type == 'Public' else 0
}

user_input = pd.DataFrame([input_dict])

if st.button("Predict Exam Score"):
    try:
        feature_order = xgb_model.feature_names_in_
        user_input_ordered = user_input[feature_order]
        predicted_score = xgb_model.predict(user_input_ordered)[0]
        
        st.success(f"Predicted Exam Score: {predicted_score:.1f}")
        st.info(f"Prediction Range: {predicted_score - RMSE:.1f} to {predicted_score + RMSE:.1f}")
        
        st.metric(
            label="Predicted Score",
            value=f"{predicted_score:.1f}",
            delta=f"± {RMSE:.1f} (RMSE)"
        )
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Check if features match the model's expected features.")

st.markdown("---")
st.caption("Note: Predictions are based on the trained XGBoost model. RMSE = 1.977")