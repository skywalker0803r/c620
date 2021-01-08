import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from config import config

# load
model_icg = joblib.load('model/c620_icg.pkl')

# functions
def save(row,log_path):
    try:
        log_df = pd.read_excel(log_path,index_col=0)
        log_df.append(row).to_excel(log_path)
    except:
        row.to_excel(log_path)

def show_progress():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.balloons()

def icg_predict(icg_input):
    while True:
        output = model_icg.predict(icg_input)
        if np.allclose(output['Simulation Case Conditions_C620 Distillate Rate_m3/hr'].values[0],0.01,atol=1e-2):
            info = 'current NA in Benzene_ppmw is:{} current Distillate Rate is:{} so NA in Benzene_ppmw -= 30'.format(
                icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values[0],
                output['Simulation Case Conditions_C620 Distillate Rate_m3/hr'].values[0])
            st.write(info)
            icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'] -= 30
        else:
            return output

# TAG
st.subheader('給這一次試算一個tag吧')
tag = st.text_input('輸入tag')
if st.button('確定'):
    st.write('tag名稱為:{}'.format(tag))

# ICG 
st.subheader('(ICG)請填入以下數值')
ICG_Input = pd.DataFrame(index=[tag],columns=model_icg.x_col)
for cname in ICG_Input.columns:
    min_ = model_icg.ss_x.data_min_[model_icg.x_col.index(cname)]
    max_ = model_icg.ss_x.data_max_[model_icg.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    ICG_Input[cname] = st.number_input(cname,min_value = float(min_),max_value = float(max_),value = float(mean_),step=1e-8,format='%.4f')
st.subheader('ICG 輸入值為：')
st.write(ICG_Input)
if st.button('ICG predict'):
    show_progress()
    icg_dist = icg_predict(ICG_Input)
    st.subheader('輸出值為：')
    st.write(icg_dist)
    save(ICG_Input.join(icg_dist),config['icg_log_path'])
