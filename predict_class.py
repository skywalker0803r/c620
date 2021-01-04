import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autorch
from autorch.function import sp2wt

class Predict(object):
  def __init__(self):
#==============================================================================================================    
    # icg
    self.model_icg = joblib.load('./model/c620_icg.pkl')
    self.icg_col_names = joblib.load('./col_names/c620_c670.pkl')
    # c620
    self.model_c620_sf = joblib.load('./model/c620.pkl')
    self.model_c620_op = joblib.load('./model/c620.pkl')
    self.c620_col_names = joblib.load('./col_names/c620_col_names.pkl') 
    # c660
    self.model_c660_sf = joblib.load('./model/c660.pkl')
    self.model_c660_op = joblib.load('./model/c660.pkl')
    self.c660_col_names = joblib.load('./col_names/c660_col_names.pkl')
    # c670
    self.model_c670_sf = joblib.load('./model/c670.pkl')
    self.model_c670_op = joblib.load('./model/c670.pkl')
    self.c670_col_names = joblib.load('./col_names/c670_col_names.pkl')
    #t651
    self.t651_col_names = joblib.load('./col_names/t651_col_names.pkl')
#==============================================================================================================
# Predict ICG
  def icg_dist(self,icg_input):
    while True:
      output = self.model_icg.predict(icg_input)
      if np.allclose(output['Simulation Case Conditions_C620 Distillate Rate_m3/hr'].values[0],0.01,atol=1e-2):
        print('current Distillate Rate is:{} so NA in Benzene_ppmw -= 30'.format(output['Simulation Case Conditions_C620 Distillate Rate_m3/hr'].values[0]))
        icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'] -= 30
      else:
        return output
#==============================================================================================================     
  # Predict c620 WT
  def c620_wt(self,c620_x):
    idx = c620_x.index
    c620_sf = self.model_c620_sf.predict(c620_x).iloc[:,:41*4]
    x41 = c620_x[self.c620_col_names['x41']].values
    s1 = c620_sf[self.c620_col_names['vent_gas_sf']].values
    s2 = c620_sf[self.c620_col_names['distillate_sf']].values
    s3 = c620_sf[self.c620_col_names['sidedraw_sf']].values
    s4 = c620_sf[self.c620_col_names['bottoms_sf']].values
    w1 = sp2wt(x41,s1)
    w2 = sp2wt(x41,s2)
    w3 = sp2wt(x41,s3)
    w4 = sp2wt(x41,s4)
    c620_wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(
        c620_wt,
        index = idx,
        columns = 
        self.c620_col_names['vent_gas_x']+\
        self.c620_col_names['distillate_x']+\
        self.c620_col_names['sidedraw_x']+\
        self.c620_col_names['bottoms_x'])
    return c620_wt
  # Predict c620 OP
  def c620_op(self,c620_x):
    return self.model_c620_op.predict(c620_x).iloc[:,41*4:] 
#==============================================================================================================     
  # Predict c660 WT
  def c660_wt(self,c660_x):
    idx = c660_x.index
    c660_sf = self.model_c660_sf.predict(c660_x[self.model_c660_sf.x_col]).iloc[:,:41*4]
    x41 = c660_x.loc[idx,self.c660_col_names['x41']]
    s1 = c660_sf[self.c660_col_names['vent_gas_sf']].values
    s2 = c660_sf[self.c660_col_names['distillate_sf']].values
    s3 = c660_sf[self.c660_col_names['sidedraw_sf']].values
    s4 = c660_sf[self.c660_col_names['bottoms_sf']].values
    w1 = sp2wt(x41,s1)
    w2 = sp2wt(x41,s2)
    w3 = sp2wt(x41,s3)
    w4 = sp2wt(x41,s4)
    c660_wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(
        c660_wt,
        index=idx,
        columns=
        self.c660_col_names['vent_gas_x']+\
        self.c660_col_names['distillate_x']+\
        self.c660_col_names['sidedraw_x']+\
        self.c660_col_names['bottoms_x'])
    return c660_wt
  # Predict c660 OP
  def c660_op(self,c660_x):
    return self.model_c660_op.predict(c660_x).iloc[:,41*4:]
#==============================================================================================================   
  # Predict c670 WT
  def c670_wt(self,c670_x):
    idx = c670_x.index
    c670_sf = self.model_c670_sf.predict(c670_x).iloc[:,:41*4]
    x41 = c670_x.iloc[:,:41].values
    s1 = c670_sf[self.c670_col_names['distillate_sf']].values
    s2 = c670_sf[self.c670_col_names['bottoms_sf']].values
    w1 = sp2wt(x41,s1)
    w2 = sp2wt(x41,s2)
    c670_wt = np.hstack((w1,w2))
    c670_wt = pd.DataFrame(
        c670_wt,
        index = idx,
        columns=
        self.c670_col_names['distillate_x']+\
        self.c670_col_names['bottoms_x'])
    return c670_wt
  # Predict c670 OP
  def c670_op(self,c670_x):
    c670_op = self.model_c670_op.predict(c670_x).iloc[:,41*2:]
    return c670_op
#============================================================================================================== 