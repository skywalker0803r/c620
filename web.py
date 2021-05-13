import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from FV2 import AllSystem
from configV2 import config
import xlrd
from tqdm import tqdm

# 一些函數
def save(row,log_path):
    try:
        log_df = pd.read_excel(log_path,index_col=0,engine='openpyxl')
        log_df.append(row).to_excel(log_path)
    except Exception as e:
        print(e)
        row.to_excel(log_path)

def show_progress():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    #st.balloons()

def let_user_input(title,default_input):
    st.subheader('{} 請填入以下數值'.format(title))
    for cname in default_input.columns:
        default_input[cname] = st.number_input(cname,value = float(default_input[cname].values[0]),step=1e-8,format='%.4f')
    st.subheader('{} 輸入值為：'.format(title))
    st.write(default_input)

# 載入模組
f =  AllSystem(config)

# 選擇模式
model_mode = st.radio("您想試算還是推薦？",('推薦', '試算'))
data_mode = st.radio("模擬數據還是現場數據？",('模擬', '現場'))

# 取得展示用資料
if data_mode == '模擬':
    demo = joblib.load('./data/demo.pkl')
    icg_input = demo['icg_input']
    c620_feed = demo['c620_feed']
    t651_feed = demo['t651_feed']

if data_mode == '現場':
    demo = joblib.load('./data/demo(real_data).pkl')
    idx = demo['icg_input'].index[0]
    icg_input = demo['icg_input'].loc[[idx]]
    c620_feed = demo['c620_feed'].loc[[idx]]
    t651_feed = demo['t651_feed'].loc[[idx]]
    

# 讓使用者輸入標籤
st.subheader('給這一次試算一個tag吧')
tag = st.text_input('輸入tag')
if st.button('確定'):
    st.write('tag名稱為:{}'.format(tag))

# 設置標籤
icg_input.index = [tag]
c620_feed.index = [tag]
t651_feed.index = [tag]

# 讓使用者輸入
let_user_input("ICG_INPUT",icg_input)
let_user_input("C620_feed",c620_feed)
let_user_input("T651_feed",t651_feed)

if st.button('Prediction'):
    if model_mode == '推薦':
        dist_rate ,SideDraw_in_BZ ,nainbz ,c620_op2 ,c660_op2 ,c670_op2 ,c620_op_Δ ,c660_op_Δ ,c670_op_Δ = f.recommend(icg_input.copy(),
                                                                                                                       c620_feed.copy(),
                                                                                                                       t651_feed.copy(),
                                                                                                                       real_data_mode = bool(data_mode == '現場')
                                                                                                                      )
    if model_mode == '試算':
        # 單純試算即可
        c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op,c620_side_體積流量2 = f.inference(icg_input.copy(),
                                                                                      c620_feed.copy(),
                                                                                      t651_feed.copy(),
                                                                                      real_data_mode = bool(data_mode == '現場')
                                                                                     )
    
    # 保存試算結果
    if model_mode == '試算':
        # save input
        save(icg_input.join(c620_feed).join(t651_feed),config['input_log_path'])
        # c620 output
        st.subheader('C620_wt')
        st.write(c620_wt)
        st.subheader('C620_op')
        st.write(c620_op)
        save(c620_wt,config['c620_wt_log_path'])
        save(c620_op,config['c620_op_log_path'])
        # c660 output
        st.subheader('C660_wt')
        st.write(c660_wt)
        st.subheader('C660_op')
        st.write(c660_op)
        save(c660_wt,config['c660_wt_log_path'])
        save(c660_op,config['c660_op_log_path'])
        # c670 output
        st.subheader('C670_wt')
        st.write(c670_wt)
        st.subheader('C670_op')
        st.write(c670_op)
        save(c670_wt,config['c670_wt_log_path'])
        save(c670_op,config['c670_op_log_path'])

    # 展示調幅建議
    if model_mode == '推薦':
        
        # 調幅
        st.subheader('c620_op調幅')
        st.write(c620_op_Δ)
        st.subheader('c660_op調幅')
        st.write(c660_op_Δ)
        st.subheader('c670_op調幅')
        st.write(c670_op_Δ)
        
        # op2
        st.subheader('c620_op2')
        st.write(c620_op2)
        st.subheader('c660_op2')
        st.write(c660_op2)
        st.subheader('c670_op2')
        st.write(c670_op2)
