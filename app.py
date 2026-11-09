import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .healthy {
        background-color: #D4EDDA;
        color: #155724;
        border: 2px solid #C3E6CB;
    }
    .at-risk {
        background-color: #F8D7DA;
        color: #721C24;
        border: 2px solid #F5C6CB;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model_path = 'heart_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the model is in the same directory as this app.")
        return None
    return joblib.load(model_path)

model = load_model()

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application predicts the presence of heart disease based on various health parameters.
Please fill in all the fields below to get a prediction.
""")

if model is None:
    st.stop()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Patient Demographics")
    
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50,
        help="Patient's age in years (variable: age)"
    )
    
    sex = st.selectbox(
        "**sex:** Sex",
        options=[1, 0],
        format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
        help="Variable: sex | Biological sex - Male=1, Female=0"
    )
    
    st.subheader("💓 Cardiovascular Measurements")
    
    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps)",
        min_value=80,
        max_value=200,
        value=120,
        help="Resting blood pressure in mm Hg (normal: ~120/80)"
    )
    
    chol = st.number_input(
        "Serum Cholesterol (chol)",
        min_value=100,
        max_value=600,
        value=200,
        help="Total serum cholesterol in mg/dl (normal: <200)"
    )
    
    fbs = st.selectbox(
        "**fbs:** Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "Yes (1) - Elevated" if x == 1 else "No (0) - Normal",
        help="Variable: fbs | Whether fasting blood sugar is greater than 120 mg/dl"
    )
    
    thalach = st.number_input(
        "Maximum Heart Rate Achieved (thalach)",
        min_value=60,
        max_value=220,
        value=150,
        help="Maximum heart rate achieved during exercise in bpm"
    )

with col2:
    st.subheader("🏥 Clinical Tests")
    
    cp = st.selectbox(
        "**cp:** Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Typical Angina",
            1: "1 - Atypical Angina",
            2: "2 - Non-anginal Pain",
            3: "3 - Asymptomatic"
        }[x],
        help="Variable: cp | Type of chest pain: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic"
    )
    
    restecg = st.selectbox(
        "**restecg:** Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Normal",
            1: "1 - ST-T Wave Abnormality",
            2: "2 - Left Ventricular Hypertrophy"
        }[x],
        help="Variable: restecg | Resting electrocardiographic results: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy"
    )
    
    exang = st.selectbox(
        "**exang:** Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        help="Variable: exang | Presence of exercise-induced angina: 1=Yes, 0=No"
    )
    
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        format="%.2f",
        help="ST depression induced by exercise relative to rest (0-10, higher=worse)"
    )
    
    slope = st.selectbox(
        "**slope:** Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Upsloping (best)",
            1: "1 - Flat",
            2: "2 - Downsloping (worst)"
        }[x],
        help="Variable: slope | The slope of the peak exercise ST segment: 0=upsloping, 1=flat, 2=downsloping"
    )
    
    ca = st.selectbox(
        "**ca:** Number of Major Vessels Colored",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}",
        help="Variable: ca | Number of major vessels (0-3) colored by fluoroscopy (higher=worse)"
    )
    
    thal = st.selectbox(
        "**thal:** Thalium Stress Test Result",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Normal",
            1: "1 - Fixed Defect",
            2: "2 - Reversible Defect",
            3: "3 - Not Described"
        }[x],
        help="Variable: thal | Thalium stress test result: 0=normal, 1=fixed defect, 2=reversible defect, 3=not described"
    )

# Create a dataframe from inputs
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Prediction button
st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("## Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box at-risk">
                    ⚠️ HIGH RISK: Heart Disease Detected<br>
                    <span style="font-size: 1.2rem;">Confidence: {prediction_proba[1]*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.warning("""
                **Recommendation:** This prediction suggests a high risk of heart disease. 
                Please consult with a healthcare professional immediately for proper evaluation and treatment.
            """)
        else:
            st.markdown(f"""
                <div class="prediction-box healthy">
                    ✅ LOW RISK: No Heart Disease Detected<br>
                    <span style="font-size: 1.2rem;">Confidence: {prediction_proba[0]*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.success("""
                **Recommendation:** While the model suggests low risk, maintaining regular check-ups 
                and a healthy lifestyle is important for long-term cardiovascular health.
            """)
        
        # Display probability breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of No Disease", f"{prediction_proba[0]*100:.2f}%")
        with col2:
            st.metric("Probability of Disease", f"{prediction_proba[1]*100:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please ensure all input values are valid.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses a **Random Forest** machine learning model trained on heart disease data 
    to predict the likelihood of heart disease based on clinical parameters.
    
    ### Important Notes
    ⚠️ **Disclaimer:** This tool is for educational purposes only and should NOT replace professional 
    medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers.
    
    ### Key Features Influencing Prediction
    1. Chest Pain Type (cp)
    2. Thalium Stress Test (thal)
    3. Number of Major Vessels (ca)
    4. ST Depression - oldpeak
    5. Exercise Induced Angina (exang)
    
    ### Dataset Variables Reference
    
    **Demographics:**
    - **age**: Age in years (1-120)
    - **sex**: 1=Male, 0=Female
    
    **Cardiovascular:**
    - **trestbps**: Resting BP (mm Hg)
    - **chol**: Cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar >120 (1/0)
    - **thalach**: Max heart rate
    
    **Clinical Tests:**
    - **cp**: Chest pain type (0-3)
    - **restecg**: Resting ECG (0-2)
    - **exang**: Exercise angina (1/0)
    - **oldpeak**: ST depression (0-10)
    - **slope**: ST slope (0-2)
    - **ca**: Major vessels (0-3)
    - **thal**: Thalium test (0-3)
    
    **Target:**
    - **target**: Heart disease (0=No, 1=Yes)
    
    ### How to Use
    1. Fill in all patient information
    2. Click "Predict Heart Disease Risk"
    3. Review the results and recommendations
    """)
    
    st.markdown("---")
    st.markdown("**Created with ❤️ using Streamlit**")
