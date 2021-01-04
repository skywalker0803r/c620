import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from predict_class import Predict
predict = Predict()
#==================================================
# ICG INPUT
st.subheader('(ICG)請填入以下數值')
ICG_Input = pd.DataFrame(index=[0],columns=predict.model_icg.x_col)
for cname in ICG_Input.columns:
    min_ = predict.model_icg.ss_x.data_min_[predict.model_icg.x_col.index(cname)]
    max_ = predict.model_icg.ss_x.data_max_[predict.model_icg.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    ICG_Input[cname] = st.number_input(cname,min_value = min_,max_value = max_,value = mean_)
st.subheader('ICG 輸入值為：')
st.write(ICG_Input)

# ICG PREDICT
if st.button('ICG預測'):    
    icg_dist = predict.icg_dist(ICG_Input)
    st.subheader('輸出值為：')
    st.write(icg_dist)

#==================================================
# C620 INPUT
st.subheader('(C620)請填入以下數值')
C620_Input = pd.DataFrame(index=[0],columns=predict.model_c620_sf.x_col)
for cname in C620_Input.columns:
    min_ = predict.model_c620_sf.ss_x.data_min_[predict.model_c620_sf.x_col.index(cname)]
    max_ = predict.model_c620_sf.ss_x.data_max_[predict.model_c620_sf.x_col.index(cname)]
    if cname == C620_Input.filter(regex='Distillate Rate').columns[0]:
        mean_ = icg_dist.values[0][0]
        print(mean_)
    else:
        mean_ = min_ + (max_ - min_)/2
    C620_Input[cname] = st.number_input(cname,min_value = min_,max_value = max_,value = float(mean_))
st.subheader('C620 輸入值為：')
st.write(C620_Input)

# C620 PREDICT
if st.button('C620預測'):    
    c620_wt = predict.c660_wt(C620_Input)
    c620_op = predict.c660_op(C620_Input)
    st.subheader('C620_wt輸出值為：')
    st.write(c620_wt)
    st.subheader('C620_op輸出值為：')
    st.write(c620_op)

