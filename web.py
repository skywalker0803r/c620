import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from predict_class import Predict

#=====================================================================================================================
# instances predict 
predict = Predict()

#=====================================================================================================================
# file path config
icg_log_path = './log/icg_log.xlsx'
c620_log_path = './log/c620_log.xlsx'
c660_log_path = './log/c660_log.xlsx'
c670_log_path = './log/c670_log.xlsx'

#=====================================================================================================================
# functions
def save(row,log_path):
    log_df = pd.read_excel(log_path,index_col=0)
    if len(log_df) != 0:
        log_df = log_df.append(row)
    else:
        log_df = row
    log_df.to_excel(log_path)

def show_progress():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.balloons()

#=====================================================================================================================
# user enter tag
st.subheader('給這一次試算一個tag吧')
tag = st.text_input('輸入tag')
if st.button('確定'):
    st.balloons()
    st.write('tag名稱為:{}'.format(tag))

#=====================================================================================================================
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
    save(ICG_Input.join(icg_dist),icg_log_path)

#=====================================================================================================================
# C620 INPUT
st.subheader('(C620)請填入以下數值')
C620_Input = pd.DataFrame(index=[tag],columns=predict.model_c620_sf.x_col)
for cname in C620_Input.columns:
    min_ = predict.model_c620_sf.ss_x.data_min_[predict.model_c620_sf.x_col.index(cname)]
    max_ = predict.model_c620_sf.ss_x.data_max_[predict.model_c620_sf.x_col.index(cname)]
    mean_ = min_ + (max_ - min_)/2
    if cname == 'Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr':
        mean_ = pd.read_excel(icg_log_path,index_col=0)[predict.model_icg.y_col].loc[tag].values[0][0]
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
    # save
    save(C620_Input.join(c620_wt).join(c620_op),c620_log_path)
    c620_w3 = c620_wt.iloc[:,41*2:41*3]
    c620_w4 = c620_wt.iloc[:,41*3:41*4]
    st.subheader('C620_w3輸出值為：')
    st.write(c620_w3)
    st.subheader('C620_w4輸出值為：')
    st.write(c620_w4)
c620_w3 = c620_w3
c620_w4 = c620_w4

# T651_WT
st.subheader('(Feed_T651)請填入以下數值')
feed_t651 = pd.DataFrame(index=[tag],columns=joblib.load('./col_names/t651_col_names.pkl')['x41'])
for cname in feed_t651.columns:
    feed_t651[cname] = st.number_input(cname,step=1e-8,format='%.4f')
st.subheader('Feed_T651 輸入值為：')
st.write(feed_t651)

# T651_MF AND C620_MF
st.subheader('(T651_MF)請填入以下數值')
t651_mf = pd.DataFrame(index=[tag],columns=joblib.load('./col_names/t651_col_names.pkl')['MFR'])
for cname in t651_mf.columns:
    t651_mf[cname] = st.number_input(cname,step=1e-8,format='%.4f')
st.subheader('T651_MF 輸入值為：')
st.write(feed_t651)

st.subheader('(C620_MF)請填入以下數值')
c620_mf = pd.DataFrame(index=[tag],columns=['Tatoray Stripper C620 Operation_Sidedraw Production Rate and Composition_Mass Flow Rate_ton/hr'])
for cname in c620_mf.columns:
    c620_mf[cname] = st.number_input(cname,step=1e-8,format='%.4f') 
st.subheader('C620_MF 輸入值為：')
st.write(c620_mf)    
    
total_mf = t651_mf.values + c620_mf.values
t651_mfr = t651_mf/total_mf
c620_mfr = c620_mf/total_mf
#c620_w3 = pd.read_excel(c620_log_path,index_col=0)[joblib.load('./col_names/c620_col_names.pkl')['sidedraw_x']].loc[tag].values
#c620_w4 = pd.read_excel(c620_log_path,index_col=0)[joblib.load('./col_names/c620_col_names.pkl')['bottoms_x']].loc[tag].values
print(c620_w3)
print(c620_w4)
c660_feed = c620_w3.values*c620_mfr.values.reshape(-1,1) + feed_t651.values*t651_mfr.values.reshape(-1,1)
c660_feed = pd.DataFrame(c660_feed,index=[tag],columns=joblib.load('./col_names/c660_col_names.pkl')['x41'])
st.subheader('C660_combined_feed值為：')
st.write(c660_feed)

'''
#=====================================================================================================================
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
    # save
    save(C660_Input.join(c660_wt).join(c660_op),c660_log_path)

#=====================================================================================================================
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
    # save
    save(C670_Input.join(c670_wt).join(c670_op),c670_log_path)
#=====================================================================================================================
'''