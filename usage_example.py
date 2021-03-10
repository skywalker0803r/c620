from F import F
from config import config
import joblib
import numpy as np
import pandas as pd
import os

def check_two_data(d1,d2):
  if np.allclose(d1.values,d2.values):
    print('一樣')
  else:
    print('不一樣')

if __name__ == '__main__':
    
    # load data
    tag = 'test001'
    demo = joblib.load('./data/demo(real_data).pkl')
    print(demo.keys())

    icg_input = demo['icg_input']
    c620_feed = demo['c620_feed']
    t651_feed = demo['t651_feed']

    idx1 = icg_input.index[0]
    idx2 = icg_input.index[1]

    # 1.確保給定不同的輸入 會有不同的輸出
    f = F(config)
    f.Recommended_mode = True #(True or False)
    f.real_data_mode = True
    # 輸入1
    c620_wt1,c620_op1,c660_wt1,c660_op1,c670_wt1,c670_op1 = f(icg_input.loc[[idx1]],c620_feed.loc[[idx1]],t651_feed.loc[[idx1]])
    # 輸入2
    c620_wt2,c620_op2,c660_wt2,c660_op2,c670_wt2,c670_op2 = f(icg_input.loc[[idx2]],c620_feed.loc[[idx2]],t651_feed.loc[[idx2]])
    check_two_data(c620_wt1,c620_wt2)
    check_two_data(c620_op1,c620_op2)
    check_two_data(c660_wt1,c660_wt2)
    check_two_data(c660_op1,c660_op2)
    check_two_data(c670_wt1,c670_wt2)
    check_two_data(c670_op1,c670_op2)

    # 2.確保變更模式 推薦 和 試算 模式 有不同的輸出
    # 試算模式
    f.Recommended_mode = False #(True or False)
    c620_wt1,c620_op1,c660_wt1,c660_op1,c670_wt1,c670_op1 = f(icg_input.loc[[idx1]],c620_feed.loc[[idx1]],t651_feed.loc[[idx1]])

    # 推薦模式
    f.Recommended_mode = True #(True or False)
    c620_wt2,c620_op2,c660_wt2,c660_op2,c670_wt2,c670_op2 = f(icg_input.loc[[idx1]],c620_feed.loc[[idx1]],t651_feed.loc[[idx1]])
    check_two_data(c620_wt1,c620_wt2)
    check_two_data(c620_op1,c620_op2)
    check_two_data(c660_wt1,c660_wt2)
    check_two_data(c660_op1,c660_op2)
    check_two_data(c670_wt1,c670_wt2)
    check_two_data(c670_op1,c670_op2)