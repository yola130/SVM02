import streamlit as st
import joblib
import numpy as np


def main():
    # Load the model
    model = joblib.load('SVM02.pkl')

    # Define feature names
    feature_names = ["ALB", "Neutrophils", "HCO3", "APTT", "Fg", "BUN", "PT","LDH", "DBIL"]

    # Streamlit user interface
    st.title("COVID-19 Subphenotype Classifier")

    # age: numerical input
    Age = st.number_input("Age:", min_value=1, max_value=120, value=70)

    # Elevated RR: categorical selection
    Elevated_RR = st.selectbox("Elevated_RR:", options=[0, 1], format_func=lambda x: 'Normal (0)' if x == 0 else 'Elevated (1)')

    # Neutrophils: numerical input
    Neutrophils = st.number_input("Neutrophils:", min_value=0, max_value=50, value=6)

    # LDH: numerical input
    LDH = st.number_input("LDH:", min_value=50, max_value=4000, value=270)

    # Glucose: numerical input
    Glucose = st.number_input("Glucose:", min_value=1, max_value=40, value=8)

    # ALB: numerical input
    ALB = st.number_input("ALB:", min_value=0, max_value=100, value=35)

    # BUN: numerical input
    BUN = st.number_input("BUN:", min_value=0, max_value=100, value=7)

    # Process inputs and make predictions
    feature_values = [Age, Elevated_RR, Neutrophils, LDH, Glucose, ALB, BUN]
    features = np.array([feature_values])

    if st.button("Predict"):
        # Predict probabilities
        res = model.predict_proba(features)
        predicted_proba = model.predict_proba(features)[0]

        # 根据预测概率的最高值来确定预测类别（但这里我们直接根据概率阈值判断）
        high_risk_threshold = 0.74  # 74% 的阈值
        if predicted_proba[1] > high_risk_threshold:  # 假设模型输出的第二个概率是高风险类的概率
            predicted_class = 1  # 高风险
        else:
            predicted_class = 0  # 低风险

         # 显示预测结果
        text = f"Predicted Class: {'*High Risk*' if predicted_class == 1 else '*Low Risk*'}"
        st.subheader(text, anchor=False)

        # 根据预测类别给出建议
        advice = f"Based on the model, predicted that the probability of COVID-19 mortality is *{predicted_proba[1] * 100:.1f}%*."

        st.subheader(advice, anchor=False)


if __name__ == "__main__":
    main()
