import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from F import F
from config import config
import xlrd

# functions
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

# get F module
f = F(config)

# select mode
mode = st.radio("您想試算還是推薦？",('推薦', '試算'))
f.Recommended_mode = bool(mode == '推薦')

# get demo data
demo = joblib.load('./data/demo.pkl')
icg_input = demo['icg_input']
c620_feed = demo['c620_feed']
t651_feed = demo['t651_feed']

# USER NEED INPUT TAG
st.subheader('給這一次試算一個tag吧')
tag = st.text_input('輸入tag')
if st.button('確定'):
    st.write('tag名稱為:{}'.format(tag))

# setting tag 
icg_input.index = [tag]
c620_feed.index = [tag]
t651_feed.index = [tag]

# USER INPUT ALL
let_user_input("ICG_INPUT",icg_input)
let_user_input("C620_feed",c620_feed)
let_user_input("T651_feed",t651_feed)

if st.button('Prediction'):
    show_progress()
    c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op = f(icg_input,c620_feed,t651_feed)
    
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
