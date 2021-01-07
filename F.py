import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autorch
from autorch.function import sp2wt

class F(object):
  def __init__(self,config):
    self.c620_model = joblib.load(config['c620_model_path'])
    self.c660_model = joblib.load(config['c660_model_path'])
    self.c670_model = joblib.load(config['c670_model_path'])
    self.c620_col = joblib.load(config['c620_col_path'])
    self.c660_col = joblib.load(config['c660_col_path'])
    self.c670_col = joblib.load(config['c670_col_path'])
  
  def __call__(self,c620_case,c620_feed,t651_feed,t651_mf,c620_mf_side,c660_case,c620_mf_bot,c660_mf_bot,c670_bf):
    # 取得index
    idx = c620_case.index
    
    # c620 預測
    c620_input = c620_case.join(c620_feed)
    c620_output = self.c620_model.predict(c620_input)
    c620_sp,c620_op = c620_output.iloc[:,:41*4],c620_output.iloc[:,41*4:]
    s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
    
    # c660 混合輸入
    total = t651_mf.values + c620_mf_side.values
    t651_mf ,c620_mf_side = t651_mf/total ,c620_mf_side/total
    c660_feed = c620_wt[self.c620_col['sidedraw_x']].values*c620_mf_side.values.reshape(-1,1) + t651_feed.values*t651_mf.values.reshape(-1,1)
    # c660 預測
    c660_input = c660_case.join(pd.DataFrame(c660_feed,index=idx,columns=self.c660_col['x41']))
    c660_output = self.c660_model.predict(c660_input)
    c660_sp,c660_op = c660_output.iloc[:,:41*4],c660_output.iloc[:,41*4:]
    s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
    
    # c670 混合輸入
    total = c620_mf_bot.values + c660_mf_bot.values
    c620_mf_bot,c660_mf_bot = c620_mf_bot/total , c660_mf_bot/total
    c670_feed = c620_wt[self.c620_col['bottoms_x']].values*c620_mf_bot.values.reshape(-1,1) + c660_wt[self.c660_col['bottoms_x']].values*c660_mf_bot.values.reshape(-1,1)
    c670_feed = pd.DataFrame(c670_feed,index=idx,columns=self.c670_col['combined'])
    # c670 預測
    c670_input = c670_feed.join(c670_bf)
    c670_output = self.c670_model.predict(c670_input)
    c670_sf,c670_op = c670_output.iloc[:,:41*2],c670_output.iloc[:,41*2:]
    s1 = c670_sf[self.c670_col['distillate_sf']].values
    s2 = c670_sf[self.c670_col['bottoms_sf']].values
    w1 = sp2wt(c670_feed,s1)
    w2 = sp2wt(c670_feed,s2)
    c670_wt = np.hstack((w1,w2))
    c670_wt = pd.DataFrame(c670_wt,index = idx,columns=self.c670_col['distillate_x']+self.c670_col['bottoms_x'])
    
    return c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op