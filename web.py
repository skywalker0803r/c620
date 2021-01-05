import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from predict_class import Predict
predict = Predict()

# path config
icg_log_path = './log/icg_log.xlsx'
c620_log_path = './log/c620_log.xlsx'
c660_log_path = './log/c660_log.xlsx'
c670_log_path = './log/c670_log.xlsx'

# functions
def check_log_or_create(tag,columns,log_path):
    log_df = pd.read_excel(log_path)
    if len(log_df.index) == 0:
        log_df = pd.DataFrame(index=[tag],columns=columns)
        log_df.to_excel(log_path)
    return log_df

def show_progress():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.balloons()

# enter tag
st.subheader('給這一次試算一個tag吧')
tag = st.text_input('輸入tag')
if st.button('確定'):
    st.balloons()
    st.write('tag名稱為:{}'.format(tag))

# instance log_df
icg_log_df = check_log_or_create(tag,predict.model_icg.x_col+predict.model_icg.y_col,icg_log_path)
c620_log_df = check_log_or_create(tag,predict.model_c620_sf.x_col+predict.model_c620_sf.y_col,c620_log_path)
c660_log_df = check_log_or_create(tag,predict.model_c660_sf.x_col+predict.model_c660_sf.y_col,c660_log_path)
c670_log_df = check_log_or_create(tag,predict.model_c670_sf.x_col+predict.model_c670_sf.y_col,c670_log_path)

# ICG INPUT
st.subheader('(ICG)請填入以下數值')
ICG_Input = pd.DataFrame(index=[tag],columns=predict.model_icg.x_col)
for cname in ICG_Input.columns:
    min_ = predict.model_icg.ss_x.data_min_[predict.model_icg.x_col.index(cname)]
    max_ = predict.model_icg.ss_x.data_max_[predict.model_icg.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    ICG_Input[cname] = st.number_input(cname,min_value = float(min_),max_value = float(max_),value = float(mean_),step=1e-8,format='%.4f')
st.subheader('ICG 輸入值為：')
st.write(ICG_Input)
# ICG PREDICT
if st.button('ICG預測'):
    show_progress()
    icg_dist = predict.icg_dist(ICG_Input)
    st.subheader('輸出值為：')
    st.write(icg_dist)
    # save as excel
    icg_log_df.loc[tag,ICG_Input.columns] = ICG_Input.loc[tag,ICG_Input.columns]
    icg_log_df.loc[tag,icg_dist.columns] = icg_dist.loc[tag,icg_dist.columns]
    icg_log_df.to_excel(icg_log_path)
#==================================================
# C620 INPUT
st.subheader('(C620)請填入以下數值')
C620_Input = pd.DataFrame(index=[tag],columns=predict.model_c620_sf.x_col)
for cname in C620_Input.columns:
    min_ = predict.model_c620_sf.ss_x.data_min_[predict.model_c620_sf.x_col.index(cname)]
    max_ = predict.model_c620_sf.ss_x.data_max_[predict.model_c620_sf.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    C620_Input[cname] = st.number_input(cname,min_value = float(min_),max_value = float(max_),value = float(mean_),step=1e-8,format='%.4f')
st.subheader('C620 輸入值為：')
st.write(C620_Input)
# C620 PREDICT
if st.button('C620預測'):
    show_progress()   
    c620_wt = predict.c620_wt(C620_Input)
    c620_op = predict.c620_op(C620_Input)
    st.subheader('C620_wt輸出值為：')
    st.write(c620_wt)
    st.subheader('C620_op輸出值為：')
    st.write(c620_op)
    # combined wt op results
    c620_output = c620_wt.join(c620_op)
    c620_log_df.loc[tag,C620_Input.columns] = C620_Input.loc[tag,C620_Input.columns]
    c620_log_df.loc[tag,c620_output.columns] = c620_output.loc[tag,c620_output.columns]
    c620_log_df.to_excel(c620_log_path)
#==================================================
# C660 INPUT
st.subheader('(C660)請填入以下數值')
C660_Input = pd.DataFrame(index=[tag],columns=predict.model_c660_sf.x_col)
for cname in C660_Input.columns:
    min_ = predict.model_c660_sf.ss_x.data_min_[predict.model_c660_sf.x_col.index(cname)]
    max_ = predict.model_c660_sf.ss_x.data_max_[predict.model_c660_sf.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    C660_Input[cname] = st.number_input(cname,min_value = float(min_),max_value = float(max_),value = float(mean_),step=1e-8,format='%.4f')
st.subheader('C660 輸入值為：')
st.write(C660_Input)
# C660 PREDICT
if st.button('C660預測'):
    show_progress() 
    c660_wt = predict.c660_wt(C660_Input)
    c660_op = predict.c660_op(C660_Input)
    st.subheader('C660_wt輸出值為：')
    st.write(c660_wt)
    st.subheader('C660_op輸出值為：')
    st.write(c660_op)
    # combined wt op results
    c660_output = c660_wt.join(c660_op)
    c660_log_df.loc[tag,C660_Input.columns] = C660_Input.loc[tag,C660_Input.columns]
    c660_log_df.loc[tag,c660_output.columns] = c660_output.loc[tag,c660_output.columns]
    c660_log_df.to_excel(c660_log_path)
#==================================================
# C670 INPUT
st.subheader('(C670)請填入以下數值')
C670_Input = pd.DataFrame(index=[tag],columns=predict.model_c670_sf.x_col)
for cname in C670_Input.columns:
    min_ = predict.model_c670_sf.ss_x.data_min_[predict.model_c670_sf.x_col.index(cname)]
    max_ = predict.model_c670_sf.ss_x.data_max_[predict.model_c670_sf.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    C670_Input[cname] = st.number_input(cname,min_value = float(min_),max_value = float(max_),value = float(mean_),step=1e-8,format='%.4f')
st.subheader('C670 輸入值為：')
st.write(C670_Input)
# C670 PREDICT
if st.button('C670預測'):
    show_progress() 
    c670_wt = predict.c670_wt(C670_Input)
    c670_op = predict.c670_op(C670_Input)
    st.subheader('C670_wt輸出值為：')
    st.write(c670_wt)
    st.subheader('C670_op輸出值為：')
    st.write(c670_op)
    # combined wt op results
    c670_output = c670_wt.join(c670_op)
    c670_log_df.loc[tag,C670_Input.columns] = C670_Input.loc[tag,C670_Input.columns]
    c670_log_df.loc[tag,c670_output.columns] = c670_output.loc[tag,c670_output.columns]
    c670_log_df.to_excel(c670_log_path)
#==================================================