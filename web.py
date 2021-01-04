import streamlit as st
import pandas as pd
import joblib
import os

# load model
model = {}
for key in os.listdir('./model'):
    model[key] = joblib.load('./model/{}'.format(key))
print(model.keys())

# UI
st.subheader('請填入以下數值')

Input = pd.DataFrame(
    index=[0],
    columns=model['c620_icg.pkl'].x_col)

for cname in Input.columns:
    Input[cname] = st.number_input(cname)

st.subheader('輸入值為：')
st.write(Input)

if st.button('預測'):    
    output = model['c620_icg.pkl'].predict(Input)
    st.subheader('輸出值為：')
    st.write(output)


