import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Demo dataset ---
data = {
    "age": [22,25,30,35,40,45,50,55,60],
    "weight": [55,60,68,72,78,82,86,90,95],
    "height": [165,170,172,168,170,169,167,168,170],
    "systolic": [118,120,124,128,132,136,140,145,150],
    "diastolic": [78,80,82,84,86,88,90,92,95]
}
df = pd.DataFrame(data)

X = df[['age','weight','height']]
y_sys = df['systolic']
y_dia = df['diastolic']

sys_model = LinearRegression().fit(X, y_sys)
dia_model = LinearRegression().fit(X, y_dia)

# --- Streamlit UI ---
st.title("PulseGuard AI – Blood Pressure Prediction")

age = st.slider("Age", 20, 80, 30)
weight = st.slider("Weight (kg)", 40, 120, 70)
height = st.slider("Height (cm)", 150, 200, 170)

user_data = np.array([[age, weight, height]])
pred_sys = sys_model.predict(user_data)[0]
pred_dia = dia_model.predict(user_data)[0]

st.subheader("Predicted Blood Pressure")
st.write(f"Systolic: {round(pred_sys)} mmHg")
st.write(f"Diastolic: {round(pred_dia)} mmHg")

if pred_sys < 120 and pred_dia < 80:
    st.success("Normal")
elif pred_sys < 140 and pred_dia < 90:
    st.warning("Elevated ⚠️")
else:
    st.error("High Risk 🚨")
