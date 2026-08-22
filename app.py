import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="centered"
)

# =========================================================
# LOAD TRAINED PIPELINE
# =========================================================

pipeline = joblib.load("adult_income_pipeline.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #0F4C81;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 25px;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    background-color: #0F4C81;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #1E88E5;
    color: white;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(" Project Details")

st.sidebar.success("Adult Income Prediction")

st.sidebar.write("###  Model")
st.sidebar.info("Random Forest Classifier")

st.sidebar.write("###  Encoding")
st.sidebar.info("OneHotEncoder")

st.sidebar.write("###  Scaling")
st.sidebar.info("StandardScaler")

st.sidebar.write("###  Pipeline")
st.sidebar.success("Scikit-Learn Pipeline")

st.sidebar.markdown("---")

st.sidebar.write("###  Target")
st.sidebar.write("Income")

st.sidebar.write("###  Problem Type")
st.sidebar.write("Binary Classification")

st.sidebar.markdown("---")

st.sidebar.caption("Machine Learning Project")

# =========================================================
# TITLE
# =========================================================

st.title("💰 Adult Income Prediction")

st.markdown(
    '<p class="subtitle">Predict whether annual income is <b><=50K</b> or <b>>50K</b></p>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# PERSONAL INFORMATION
# =========================================================

st.subheader(" Personal Information")

age = st.slider(
    "Age",
    min_value=17,
    max_value=90,
    value=30
)

sex = st.radio(
    "Gender",
    ["Female", "Male"],
    horizontal=True
)

race = st.selectbox(
    "Race",
    [
        "Amer-Indian-Eskimo",
        "Asian-Pac-Islander",
        "Black",
        "Other",
        "White"
    ]
)

relationship = st.selectbox(
    "Relationship",
    [
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Own-child",
        "Unmarried",
        "Wife"
    ]
)

# =========================================================
# EDUCATION INFORMATION
# =========================================================

st.subheader(" Education Information")

education = st.selectbox(
    "Education",
    [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool"
    ]
)

education_num = st.slider(
    "Education Number",
    min_value=1,
    max_value=16,
    value=10
)

# =========================================================
# WORK INFORMATION
# =========================================================

st.subheader(" Work Information")

workclass = st.selectbox(
    "Workclass",
    [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
        "?"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "?",
        "Adm-clerical",
        "Armed-Forces",
        "Craft-repair",
        "Exec-managerial",
        "Farming-fishing",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Other-service",
        "Priv-house-serv",
        "Prof-specialty",
        "Protective-serv",
        "Sales",
        "Tech-support",
        "Transport-moving"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse"
    ]
)

hours_per_week = st.slider(
    "Hours Per Week",
    min_value=1,
    max_value=99,
    value=40
)

# =========================================================
# FINANCIAL INFORMATION
# =========================================================

st.subheader(" Financial Information")

capital_gain = st.number_input(
    "Capital Gain",
    min_value=0,
    max_value=100000,
    value=0
)

capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    max_value=5000,
    value=0
)

fnlwgt = st.number_input(
    "Final Weight (fnlwgt)",
    min_value=10000,
    max_value=1500000,
    value=100000
)

# =========================================================
# LOCATION
# =========================================================

st.subheader(" Location")

native_country = st.selectbox(
    "Native Country",
    [
        "?",
        "Cambodia",
        "Canada",
        "China",
        "Columbia",
        "Cuba",
        "Dominican-Republic",
        "Ecuador",
        "El-Salvador",
        "England",
        "France",
        "Germany",
        "Greece",
        "Guatemala",
        "Haiti",
        "Holand-Netherlands",
        "Honduras",
        "Hong",
        "Hungary",
        "India",
        "Iran",
        "Ireland",
        "Italy",
        "Jamaica",
        "Japan",
        "Laos",
        "Mexico",
        "Nicaragua",
        "Outlying-US(Guam-USVI-etc)",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Puerto-Rico",
        "Scotland",
        "South",
        "Taiwan",
        "Thailand",
        "Trinadad&Tobago",
        "United-States",
        "Vietnam",
        "Yugoslavia"
    ]
)

# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")

if st.button(" Predict Income"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "fnlwgt": [fnlwgt],
        "education": [education],
        "education.num": [education_num],
        "marital.status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "sex": [sex],
        "capital.gain": [capital_gain],
        "capital.loss": [capital_loss],
        "hours.per.week": [hours_per_week],
        "native.country": [native_country]
    })

    # Prediction
    with st.spinner(" Analyzing the data..."):

        prediction = pipeline.predict(input_data)

        probability = pipeline.predict_proba(input_data)

    predicted_class = prediction[0]

    confidence = max(probability[0])

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.subheader(" Prediction Result")

    if predicted_class == ">50K":

        st.success(" Predicted Income: **>50K**")

        st.balloons()

    else:

        st.warning(" Predicted Income: **<=50K**")

    # =====================================================
    # CONFIDENCE
    # =====================================================

    st.subheader(" Prediction Confidence")

    st.progress(float(confidence))

    st.write(
        f"**Model Confidence: {confidence * 100:.2f}%**"
    )

    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader(" Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "Education",
            "Occupation",
            "Marital Status",
            "Hours Per Week",
            "Capital Gain",
            "Capital Loss",
            "Native Country"
        ],

        "Value": [
            age,
            sex,
            education,
            occupation,
            marital_status,
            hours_per_week,
            capital_gain,
            capital_loss,
            native_country
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<p style='text-align:center;'>"
    " Adult Income Prediction | Machine Learning Project"
    "</p>",
    unsafe_allow_html=True
)