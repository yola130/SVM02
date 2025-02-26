import streamlit as st
import joblib
import pandas as pd


# Load the model
model = joblib.load('SVM02.pkl')

# Define feature names
feature_names = ["ALB", "Neutrophils", "HCO3", "APTT", "Fg", "BUN", "PT","LDH", "DBIL"]

# Streamlit user interface
st.title("COVID-19 Subphenotype Classifier")

# ALB: numerical input
ALB = st.number_input("ALB:", min_value=0, max_value=100, value=35)

# Neutrophils: numerical input
Neutrophils = st.number_input("Neutrophils:", min_value=0, max_value=100, value=6)

# HCO3: numerical input
HCO3 = st.number_input("HCO3:", min_value=0, max_value=100, value=25)

# APTT: numerical input
APTT = st.number_input("APTT:", min_value=0, max_value=100, value=40)

# Fg: numerical input
Fg = st.number_input("Fg:", min_value=0, max_value=50, value=3)

# BUN: numerical input
BUN = st.number_input("BUN:", min_value=0, max_value=200, value=5)

# PT: numerical input
PT = st.number_input("PT:", min_value=0, max_value=100, value=12)

# LDH: numerical input
LDH = st.number_input("LDH:", min_value=50, max_value=4000, value=270)

# DBIL: numerical input
DBIL = st.number_input("DBIL:", min_value=0, max_value=100, value=5)

 ["ALB", "Neutrophils", "HCO3", "APTT", "Fg", "BUN", "PT","LDH", "DBIL"]

# Process inputs and make predictions
# feature_values = [ALB, Neutrophils, HCO3, APTT, Fg, BUN, PT, LDH, DBIL]
# features = np.array([feature_values])
data = {"ALB": [ALB], "Neutrophils": [Neutrophils], "HCO3":[HCO3],  "APTT":[APTT], "Fg": [Fg], "BUN":[BUN], 
       "PT": [PT], "LDH":[LDH], "DBIL":[DBIL]}
features = pd.DataFrame(data)

if st.button("Predict"):
    # Predict probabilities
    predicted_proba = model.predict_proba(features)[0]
    st.text(predicted_proba)
    
    # 根据预测概率的最高值来确定预测类别（但这里我们直接根据概率阈值判断）  
    high_risk_threshold = 0.32  # 32% 的阈值  
    if predicted_proba[1] > high_risk_threshold:  # 假设模型输出的第二个概率是高风险类的概率  
        predicted_class = 1  # Cluster2 
    else:  
        predicted_class = 0  # Cluster1

    # 显示预测结果  
    text = f"Predicted Class: {'*Cluster 2*' if predicted_class == 1 else '*Cluster 1*'}"
    st.subheader(text, anchor=False)
        
    # 根据预测类别给出建议
    advice = f"Based on the model, predicted that the probability of Cluster 2 is *{predicted_proba[1] * 100:.1f}%*."

    st.subheader(advice, anchor=False)
