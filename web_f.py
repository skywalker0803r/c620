import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from F import F
from config import config

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

def let_user_input(title,default_input):
    st.subheader('{} 請填入以下數值'.format(title))
    for cname in default_input.columns:
        default_input[cname] = st.number_input(cname,value = float(default_input[cname].values[0]),step=1e-8,format='%.4f')
    st.subheader('{} 輸入值為：'.format(title))
    st.write(default_input)

# get F module
f = F(config)

# get demo data
demo = joblib.load('./data/demo.pkl')
c620_case = demo['c620_case']
c620_feed = demo['c620_feed']
t651_feed = demo['t651_feed']
t651_mf = demo['t651_mf'].to_frame()
c620_mf_side = demo['c620_mf_side'].to_frame()
c660_case = demo['c660_case']
c620_mf_bot = demo['c620_mf_bot'].to_frame()
c660_mf_bot = demo['c660_mf_bot'].to_frame()
c670_bf = demo['c670_bf']

# USER NEED INPUT TAG
st.subheader('給這一次試算一個tag吧')
tag = st.text_input('輸入tag')
if st.button('確定'):
    st.write('tag名稱為:{}'.format(tag))
    # setting tag 
    c620_case.index = [tag]
    c620_feed.index = [tag]
    t651_feed.index = [tag]
    t651_mf.index = [tag]
    c620_mf_side.index = [tag]
    c660_case.index = [tag]
    c620_mf_bot.index = [tag]
    c660_mf_bot.index = [tag]
    c670_bf.index = [tag]

# USER INPUT ALL
let_user_input("C620_CASE",c620_case)
let_user_input("C620_FEED",c620_feed)
let_user_input("T651_FEED",t651_feed)
let_user_input("T651_MF",t651_mf)
let_user_input("C620_MF_SIDE",c620_mf_side)
let_user_input("C660_CASE",c660_case)
let_user_input("C620_MF_BOT",c620_mf_bot)
let_user_input("C660_MF_BOT",c660_mf_bot)
let_user_input("C670_BF",c670_bf)

if st.button('Prediction'):
    c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op = f(c620_case,c620_feed,t651_feed,t651_mf,c620_mf_side,c660_case,c620_mf_bot,c660_mf_bot,c670_bf)
    
    st.subheader('C620_wt')
    st.write(c620_wt)
    st.subheader('C620_op')
    st.write(c620_op)

    st.subheader('C660_wt')
    st.write(c660_wt)
    st.subheader('C660_op')
    st.write(c660_op)

    st.subheader('C670_wt')
    st.write(c670_wt)
    st.subheader('C670_op')
    st.write(c670_op)