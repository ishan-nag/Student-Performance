import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
xgb_model = joblib.load('xgb_model.pkl')
RMSE = 1.977

st.title("Student Exam Score Prediction")
st.write("Predict a student's exam score based on study habits, resources, and other factors.")

# two columns
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
    previous_scores = st.number_input("Previous Exam Score", min_value=0, max_value=100, value=70)
    tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
    physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, value=2)
    
    # Peer Influence
    peer_influence = st.selectbox("Peer Influence", ["Low", "Medium", "High"])

with col2:
    # Categorical inputs
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

# Mapping dictionaries
distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
binary_map = {'Yes': 1, 'No': 0}

input_dict = {
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Parental_Involvement': ordinal_map[parental_involvement],  # Position 3
    'Access_to_Resources': ordinal_map[access_to_resources],    # Position 4
    'Extracurricular_Activities': binary_map[extracurricular_activities],  # Position 5
    'Sleep_Hours': sleep_hours,  # Position 6
    'Previous_Scores': previous_scores,  # Position 7
    'Motivation_Level': ordinal_map[motivation_level],  # Position 8
    'Internet_Access': binary_map[internet_access],  # Position 9
    'Tutoring_Sessions': tutoring_sessions,  # Position 10
    'Family_Income': ordinal_map[family_income],  # Position 11
    'Peer_Influence': ordinal_map[peer_influence],  # Position 12
    'Physical_Activity': physical_activity,  # Position 13
    'Learning_Disabilities': binary_map[learning_disabilities],  # Position 14
    'Distance_from_Home': distance_map[distance_from_home],  # Position 15
    'Gender_Male': 1 if gender == 'Male' else 0,  # Position 16
    'School_Type_Public': 1 if school_type == 'Public' else 0  # Position 17
}

if st.checkbox("Show input features (debug)"):
    st.write("Features being sent to model:")
    for feature, value in input_dict.items():
        st.write(f"{feature}: {value}")
    
    st.write(f"Total features: {len(input_dict)}")
    st.write(f"Expected by model: {len(xgb_model.feature_names_in_)}")

feature_order = xgb_model.feature_names_in_
user_input = pd.DataFrame([input_dict])[feature_order]

if st.button("Predict Exam Score"):
    try:
        predicted_score = xgb_model.predict(user_input)[0]
        st.success(f"Predicted Exam Score: {predicted_score:.1f}")
        st.info(f"Prediction Range: {predicted_score - RMSE:.1f} to {predicted_score + RMSE:.1f}")
        
        # Optional: Show confidence interval
        st.metric(
            label="Predicted Score",
            value=f"{predicted_score:.1f}",
            delta=f"± {RMSE:.1f} (RMSE)"
        )
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Check if all required features are provided in the correct order.")

#feature importance visualization
if st.checkbox("Show Feature Importance"):
    try:
        import matplotlib.pyplot as plt
        
        feature_importance = pd.DataFrame({
            'Feature': xgb_model.feature_names_in_,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_importance['Feature'], feature_importance['Importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (XGBoost)')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")

st.markdown("---")
st.caption("Note: Predictions are based on the trained XGBoost model. RMSE = 1.977")