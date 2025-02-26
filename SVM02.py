import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 配置页面元信息
st.set_page_config(
    page_title="COVID-19亚型分类系统",
    page_icon="🦠",
    layout="centered"
)

# 常量定义
HIGH_RISK_THRESHOLD = 0.32  # 高风险阈值
FEATURE_RANGES = {
    "Neutrophils": (0, 50, 6),
    "PT": (10, 20, 12),
    "APTT": (0, 100, 40),
    "Fg": (0, 20, 3.0),
    "ALB": (10, 100, 35),
    "BUN": (0, 200, 5),
    "HCO3": (0, 100, 25),
    "LDH": (100, 3000, 270),
    "DBIL": (1, 100, 5)

}

def load_model(model_path='SVM02.pkl'):
    """安全加载机器学习模型"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

def validate_inputs(inputs):
    """输入数据验证"""
    alerts = []
    if inputs['LDH'] > 1000:
        alerts.append("⚠️ LDH>1000建议复查检测值")
    if inputs['Neutrophils'] > 15:
        alerts.append("⚠️ 中性粒细胞>15×10⁹/L提示严重感染")
    return alerts

def main():
    # 初始化模型
    model = load_model()
    
    st.title("COVID-19临床亚型智能分类系统")
    st.markdown("---")
    
    # 动态生成输入组件
    inputs = {}
    cols = st.columns(3)
    for idx, (feature, (min_val, max_val, default)) in enumerate(FEATURE_RANGES.items()):
        with cols[idx%3]:
            inputs[feature] = st.number_input(
                label=f"{feature}:",
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=0.1 if feature in ['Fg'] else 1
            )
    
    # 输入验证
    alerts = validate_inputs(inputs)
    if alerts:
        st.warning("\n\n".join(alerts))
    
    # 预测执行
    if st.button("开始分析", type="primary"):
        try:
            # 构建特征矩阵
            features = pd.DataFrame([inputs])
            
            # 获取预测概率
            proba = model.predict_proba(features)[0]
            risk_score = proba[1]  # 假设类别1为高风险
            
            # 结果可视化
            st.markdown("---")
            progress_bar = st.progress(0)
            for percent in range(0, int(risk_score*100)+1, 5):
                progress_bar.progress(percent/100)
            
            # 诊断结论
            if risk_score > HIGH_RISK_THRESHOLD:
                st.error(f"## 高危亚型 (Subphenotype II)\n"
                         f"**风险评估值**: {risk_score:.1%} "
                         f"(阈值 {HIGH_RISK_THRESHOLD:.0%})")
                st.markdown("**临床建议**:\n"
                            "- 立即启动抗炎治疗\n"
                            "- 监测凝血功能\n"
                            "- 建议ICU监护")
            else:
                st.success(f"## 标准亚型 (Subphenotype I)\n"
                          f"**风险评估值**: {risk_score:.1%}")
                st.markdown("**临床建议**:\n"
                            "- 常规抗病毒治疗\n"
                            "- 每日生命体征监测\n"
                            "- 营养支持治疗")
            
            # 显示特征重要性
            if hasattr(model, 'coef_'):
                st.markdown("### 关键预测因素")
                coef_df = pd.DataFrame({
                    '特征': FEATURE_RANGES.keys(),
                    '权重': model.coef_[0]
                }).sort_values('权重', ascending=False)
                st.bar_chart(coef_df.set_index('特征'))
                
        except Exception as e:
            st.error(f"预测过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()
