import streamlit as st
import joblib
import numpy as np


def main():
    # Load the model
    model = joblib.load('SVM02.pkl')

    # Define feature names
    feature_names = ["Neutrophils", "PT", "APTT", "Fg", "ALB", "BUN", "HCO3", "LDH", "DBIL"]

    # Streamlit user interface
    st.title("COVID-19 Subphenotype Classifier")

    # Neutrophils: numerical input
    Neutrophils = st.number_input("Neutrophils:", min_value=0, max_value=50, value=6)
    
    # PT: numerical input
    PT = st.number_input("PT:", min_value=0, max_value=50, value=12)
    
    # APTT: numerical input
    APTT = st.number_input("APTT:", min_value=0, max_value=800, value=35)
        
    # Fg: numerical input
    Fg = st.number_input("Fg:", min_value=0, max_value=20, value=3)
    
    # ALB: numerical input
    ALB = st.number_input("ALB:", min_value=0, max_value=100, value=35)

    # BUN: numerical input
    BUN = st.number_input("BUN:", min_value=0, max_value=100, value=7)
    
    # HCO3: numerical input
    HCO3 = st.number_input("HCO3:", min_value=0, max_value=50, value=25)
    
    # LDH: numerical input
    LDH = st.number_input("LDH:", min_value=50, max_value=4000, value=270)

    # DBIL: numerical input
    DBIL = st.number_input("DBIL:", min_value=0, max_value=100, value=10)

    # Process inputs and make predictions
    feature_values = [Neutrophils, PT, APTT, Fg, ALB, BUN, HCO3, LDH, DBIL]
    features = np.array([feature_values])

    if st.button("Predict"):
        # Predict probabilities
        res = model.predict_proba(features)
        predicted_proba = model.predict_proba(features)[0]

        # 根据预测概率的最高值来确定预测类别（但这里我们直接根据概率阈值判断）
        high_risk_threshold = 0.32  # 74% 的阈值
        if predicted_proba[1] > high_risk_threshold:  # 假设模型输出的第二个概率是高风险类的概率
            predicted_class = 1  # 高风险
        else:
            predicted_class = 0  # 低风险

         # 显示预测结果
        text = f"Predicted Class: {'*Subphenotype 2*' if predicted_class == 1 else '*Subphenotype 1*'}"
        st.subheader(text, anchor=False)

        # 根据预测类别给出建议
        advice = f"Based on the model, predicted that the probability of Subphenotype 2 is *{predicted_proba[1] * 100:.1f}%*."

        st.subheader(advice, anchor=False)


if __name__ == "__main__":
    main()
