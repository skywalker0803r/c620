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
        
        # 先就現有數據試算一遍,稱之為第一次試算
        c620_wt1,c620_op1,c660_wt1,c660_op1,c670_wt1,c670_op1,c660_mf1 = f.inference(icg_input.copy(),c620_feed.copy(),t651_feed.copy(),real_data_mode = bool(data_mode == '現場'))
        
        # 調整icg_input到使用者期望的規格,調整過後的icg_input稱之為icg_input2
        icg_input2 = icg_input.copy()
        
        icg_input2['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = 0.0 #從0看情況增加
        icg_input2['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'] = 980 # 980最理想
        icg_input2['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = 10 #10最理想
        
        if icg_input['Simulation Case Conditions_Feed Rate_Feed from T651_m3/hr'].values[0] + c660_mf1[0,0] > 150: # 兩種情況一種設為80第二種設為70
            icg_input2['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'] = 85
        else:
            icg_input2['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'] = 70
        
        
        history={}
        history['distrate'] = []
        history['nainbz'] = []
        step = 0.05

        # dist_rate開始遞增直到 nainbz <= 980
        for dist_rate in tqdm(np.arange(0,10,step)):
            icg_input2 = demo['icg_input'].copy()
            icg_input2['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = dist_rate
            c620_wt2,c620_op2,c660_wt2,c660_op2,c670_wt2,c670_op2,c660_mf2 = f.inference(icg_input2,demo['c620_feed'],demo['t651_feed'],real_data_mode = bool(data_mode == '現場'))
            na_idx = [1,2,3,4,5,6,8,9,11,13,14,15,20,22,29] 
            nainbz = c660_wt2.filter(regex='Side').filter(regex='wt%').iloc[:,na_idx].sum(axis=1).values[0]*10000
            history['distrate'].append(dist_rate)
            history['nainbz'].append(nainbz)
            st.write(f'dist_rate:{dist_rate} nainbz:{nainbz}') #打印訊息
            # 如果滿足以下條件跳出迴圈
            if nainbz <= 980:
                break
    
    if model_mode == '試算':
        # 單純試算即可
        c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op,c660_mf1 = f.inference(icg_input.copy(),c620_feed.copy(),t651_feed.copy(),real_data_mode = bool(data_mode == '現場'))
    
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
        st.write(c620_op2-c620_op1)
        st.subheader('c660_op調幅')
        st.write(c660_op2-c660_op1)
        st.subheader('c670_op調幅')
        st.write(c670_op2-c670_op1)

        # op1
        st.subheader('c620_op1')
        st.write(c620_op1)
        st.subheader('c660_op1')
        st.write(c660_op1)
        st.subheader('c670_op1')
        st.write(c670_op1)

        # op2
        st.subheader('c620_op2')
        st.write(c620_op2)
        st.subheader('c660_op2')
        st.write(c660_op2)
        st.subheader('c670_op2')
        st.write(c670_op2)
