# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model
model = joblib.load('iris_model.pkl')

# Load dataset for visualizations
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Title
st.title(" Iris Flower Prediction App")
st.write("This app predicts the type of Iris flower based on input features.")

# Input sliders
st.sidebar.header("Input Features")
def user_input():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame([data])

input_df = user_input()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"Predicted Iris Flower: **{target_names[prediction[0]]}**")

st.subheader("Prediction Probability")
prob_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write(prob_df)

# Visualization
st.subheader("Feature Distribution")
iris_df = pd.DataFrame(iris.data, columns=feature_names)
iris_df['species'] = [target_names[i] for i in iris.target]

fig = sns.pairplot(iris_df, hue='species')
st.pyplot(fig)
