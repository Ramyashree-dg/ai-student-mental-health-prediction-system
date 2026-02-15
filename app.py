import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("mental_model.pkl", "rb"))

st.set_page_config(page_title="Student Mental Health AI", layout="wide")

st.title("🎓 AI-Based Student Mental Health Prediction System")
st.markdown("### Using Machine Learning to Predict Depression Risk")
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("📝 Student Information")

age = st.sidebar.slider("Age", 17, 30, 20)
cgpa = st.sidebar.slider("CGPA", 0.0, 4.0, 3.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

course = st.sidebar.selectbox(
    "Course",
    ["Engineering", "Science", "Arts", "Business", "Other"]
)

year = st.sidebar.selectbox(
    "Year of Study",
    ["Year 1", "Year 2", "Year 3", "Year 4"]
)

marital = st.sidebar.selectbox("Marital Status", ["No", "Yes"])
anxiety = st.sidebar.selectbox("Do you have Anxiety?", ["No", "Yes"])
panic = st.sidebar.selectbox("Do you have Panic Attack?", ["No", "Yes"])
treatment = st.sidebar.selectbox(
    "Did you seek specialist treatment?", ["No", "Yes"]
)

# Convert categorical inputs to numeric
gender = 1 if gender == "Male" else 0
marital = 1 if marital == "Yes" else 0
anxiety = 1 if anxiety == "Yes" else 0
panic = 1 if panic == "Yes" else 0
treatment = 1 if treatment == "Yes" else 0

course_map = {
    "Engineering": 0,
    "Science": 1,
    "Arts": 2,
    "Business": 3,
    "Other": 4
}

year_map = {
    "Year 1": 0,
    "Year 2": 1,
    "Year 3": 2,
    "Year 4": 3
}

course = course_map[course]
year = year_map[year]

# Final input array (9 features)
input_data = np.array([[age, cgpa, gender, course, year,
                        marital, anxiety, panic, treatment]])

# Prediction Button
if st.sidebar.button("🔍 Predict Depression Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.error("⚠ High Risk of Depression")
    else:
        st.success("✅ Low Risk of Depression")

    st.metric("Risk Probability", f"{probability:.2f}%")

    st.progress(int(probability))

    st.markdown("---")

    # Recommendations Section
    st.markdown("### 💡 Recommendations")

    if prediction == 1:
        st.write("• Consider speaking with a mental health professional.")
        st.write("• Practice stress management techniques.")
        st.write("• Maintain a healthy sleep schedule.")
        st.write("• Stay socially connected.")
    else:
        st.write("• Maintain a balanced lifestyle.")
        st.write("• Continue healthy habits.")
        st.write("• Monitor stress levels regularly.")

    st.markdown("---")

    # Feature Importance Graph
    if hasattr(model, "feature_importances_"):
        st.markdown("### 📈 Feature Importance")

        features = [
            "Age", "CGPA", "Gender", "Course", "Year",
            "Marital", "Anxiety", "Panic", "Treatment"
        ]

        importance = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

st.markdown("---")
st.caption("⚠ Disclaimer: This system is for academic purposes only and not a medical diagnosis tool.")