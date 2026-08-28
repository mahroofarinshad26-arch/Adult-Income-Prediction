#  Adult Income Prediction 

##  Project Overview

This project focuses on predicting whether an individual's annual income is **less than or equal to $50K (<=50K)** or **greater than $50K (>50K)** using Machine Learning.

The project uses the **Adult Income Dataset**, which contains demographic, educational, employment, and financial-related information about individuals.

The project includes data preprocessing, exploratory data analysis (EDA), visualization, machine learning model training, evaluation, and deployment using Streamlit.

---

##  Objective

The main objective of this project is to build a Machine Learning classification model that can predict an individual's income category based on different personal, educational, employment, and financial features.

The prediction has two classes:

- **<=50K** – Income is less than or equal to $50,000
- **>50K** – Income is greater than $50,000

---

##  Dataset

The Adult Income Dataset contains information about individuals and their socio-economic characteristics.

### Features Used

- Age
- Workclass
- Final Weight (fnlwgt)
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours Per Week
- Native Country

### Target Variable

**Income**

- <=50K
- >50K

---

##  Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify patterns and relationships between variables.

The analysis included:

- Checking dataset shape
- Checking data types
- Checking missing values
- Checking duplicate values
- Statistical summary
- Distribution analysis
- Boxplots
- Correlation Heatmap
- Categorical feature analysis

###  Visualizations

The project includes visualizations such as:

- Age Boxplot
- Hours Per Week Boxplot
- Final Weight Boxplot
- Education Number Boxplot
- Capital Gain Boxplot
- Capital Loss Boxplot
- Correlation Heatmap
- Confusion Matrix

These visualizations helped in understanding the distribution of numerical features, outliers, and relationships between variables.

---

##  Data Preprocessing

The dataset contains both numerical and categorical features.

The preprocessing steps included:

1. Handling missing values
2. Checking and handling duplicate records
3. Separating features and target variable
4. Encoding categorical variables
5. Scaling numerical features

### Encoding

Categorical variables were converted into numerical format using encoding techniques.

**Label Encoding** was used where appropriate, while **One-Hot Encoding** was used for categorical features with multiple categories.

### Scaling

Numerical features were scaled using **StandardScaler** so that features with different numerical ranges could be handled effectively by the machine learning models.

---

##  Machine Learning Models

Three Machine Learning classification models were trained and evaluated.

### 1. Logistic Regression

Logistic Regression is a classification algorithm used to predict the probability of an observation belonging to a particular class.

In this project, it predicts whether the income is:

**<=50K or >50K**

---

### 2. Decision Tree Classifier

Decision Tree is a tree-based classification algorithm.

It makes decisions by splitting the dataset based on feature values.

The final prediction is obtained by following the decision path from the root node to a leaf node.

---

### 3. Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm.

It creates multiple decision trees and combines their predictions to produce a final prediction.

Random Forest was selected as the final model because it provided strong classification performance and works well with a mixture of numerical and categorical features.

---

##  Model Evaluation

The models were evaluated using different classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

These metrics were used to compare the performance of the different models.

---

##  Machine Learning Pipeline

A Machine Learning pipeline was created using Scikit-Learn.

The pipeline combines:

**Data Preprocessing → Encoding → Scaling → Model Prediction**

This ensures that the same preprocessing steps are automatically applied to new input data before making predictions.

The final pipeline was saved using **Joblib** as:

`adult_income_pipeline.pkl`

This saved pipeline is used directly by the Streamlit application.

---

##  Streamlit Deployment

The trained Machine Learning model was deployed using **Streamlit**.

The Streamlit application provides an interactive interface where users can enter individual details such as:

- Age
- Workclass
- Education
- Occupation
- Marital Status
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours Per Week
- Native Country

After entering the details, the user can click the **Predict** button.

The application then predicts the individual's income category:

###  Prediction

**<=50K** or **>50K**

---

##  Project Structure

```text
Adult-Income-Prediction/
│
├── app.py
├── adult_income_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore


## Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Joblib
Streamlit
GitHub


##  Conclusion


This project successfully developed an end-to-end Machine Learning system for predicting an individual's income category.

The project covers the complete Machine Learning workflow, starting from data preprocessing and exploratory data analysis to model training, evaluation, pipeline creation, and deployment.

Different classification models were trained and compared using evaluation metrics such as Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.

The final Machine Learning pipeline was integrated with a Streamlit application, allowing users to enter individual details and receive an income prediction interactively.

Overall, this project demonstrates how Machine Learning can be used to analyze demographic, educational…
