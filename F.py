import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autorch
from autorch.function import sp2wt

class F(object):
  def __init__(self,config):
    # simulation data model
    self.icg_model = joblib.load(config['icg_model_path'])
    self.c620_model = joblib.load(config['c620_model_path'])
    self.c660_model = joblib.load(config['c660_model_path'])
    self.c670_model = joblib.load(config['c670_model_path'])
    
    # real data model
    self.icg_real_data_model = joblib.load(config['icg_model_path_real_data'])
    self.c620_real_data_model = joblib.load(config['c620_model_path_real_data'])
    self.c660_real_data_model = joblib.load(config['c660_model_path_real_data'])
    self.c670_real_data_model = joblib.load(config['c670_model_path_real_data'])
    
    # real data linear model
    self.c620_real_data_model_linear = joblib.load(config['c620_model_path_real_data_linear'])
    self.c660_real_data_model_linear = joblib.load(config['c660_model_path_real_data_linear'])
    self.c670_real_data_model_linear = joblib.load(config['c670_model_path_real_data_linear'])
    
    # columns name
    self.icg_col = joblib.load(config['icg_col_path'])
    self.c620_col = joblib.load(config['c620_col_path'])
    self.c660_col = joblib.load(config['c660_col_path'])
    self.c670_col = joblib.load(config['c670_col_path'])
    
    # simple op_col
    self.c620_simple_op_col = joblib.load(config['c620_simple_op_col'])
    self.c660_simple_op_col = joblib.load(config['c660_simple_op_col'])
    self.c670_simple_op_col = joblib.load(config['c670_simple_op_col'])
    
    # other infomation
    self.c620_wt_always_same_split_factor_dict = joblib.load(config['c620_wt_always_same_split_factor_dict'])
    self.c660_wt_always_same_split_factor_dict = joblib.load(config['c660_wt_always_same_split_factor_dict'])
    self.c670_wt_always_same_split_factor_dict = joblib.load(config['c670_wt_always_same_split_factor_dict'])
    self.index_9999 = joblib.load(config['index_9999_path'])
    self.index_0001 = joblib.load(config['index_0001_path'])
    self.V615_density = 0.8626
    self.C820_density = 0.8731
    self.T651_density = 0.8749
    
    # user can set two mode
    self.Recommended_mode = False
    self.real_data_mode = False
    self._Post_processing = True
    self._linear_model = False

  def ICG_loop(self,Input):
    while True:
      if self.real_data_mode == True:
        output = pd.DataFrame(self.icg_real_data_model.predict(Input[self.icg_col['x']].values),
        index=Input.index,columns=['Simulation Case Conditions_C620 Distillate Rate_m3/hr'])
      if self.real_data_mode == False:
        output = pd.DataFrame(self.icg_model.predict(Input[self.icg_col['x']].values),
        index=Input.index,columns=['Simulation Case Conditions_C620 Distillate Rate_m3/hr'])
      dist_rate = output['Simulation Case Conditions_C620 Distillate Rate_m3/hr'].values[0]
      na_in_benzene = Input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values[0]
      print('current Distillate Rate_m3/hr:{} NA in Benzene_ppmw:{}'.format(dist_rate,na_in_benzene))
      if output['Simulation Case Conditions_C620 Distillate Rate_m3/hr'].values[0] > 0:
        return output,Input
      else:
        Input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'] -= 30
        print('NA in Benzene_ppmw -= 30')
  
  def __call__(self,icg_input,c620_feed,t651_feed):
    
    # get index
    idx = icg_input.index

    # c620_case 
    c620_case = pd.DataFrame(index=idx,columns=self.c620_col['case'])

    # c620_case(Receiver Temp_oC) = user input
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'].values
    
    if self.Recommended_mode == True:
      icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'] = 980.0 
      icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'] = 70.0
      icg_output,icg_input = self.ICG_loop(icg_input)
      c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_output.values
      c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'] = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values
    
    if self.Recommended_mode == False:
      c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values
      c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'] = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values
    
    # c620_input(c620_case&c620_feed)
    c620_input = c620_case.join(c620_feed)
    
    # c620 output(op&wt)
    c620_input = c620_case.join(c620_feed)
    c620_output = self.c620_model.predict(c620_input)
    c620_sp,c620_op = c620_output.iloc[:,:41*4],c620_output.iloc[:,41*4:]
    
    # update by c620 real data model?
    if self.real_data_mode == True:
      if self._linear_model == True:
        c620_op_real = self.c620_real_data_model_linear.predict(c620_input)[:,41*4:]
        c620_op_real = pd.DataFrame(c620_op_real,index=c620_input.index,columns=self.c620_simple_op_col)
        c620_sp_real = self.c620_real_data_model_linear.predict(c620_input)[:,:41*4]
        c620_sp_real = pd.DataFrame(c620_sp_real,index=c620_input.index,columns=c620_sp.columns)
      if self._linear_model == False:
        c620_op_real = self.c620_real_data_model.predict(c620_input).iloc[:,41*4:] 
        c620_sp_real = self.c620_real_data_model.predict(c620_input).iloc[:,:41*4] 
      c620_op.update(c620_op_real)
      c620_sp.update(c620_sp_real)
    
    # c620 sp後處理
    if self._Post_processing:
      for i in self.c620_wt_always_same_split_factor_dict.keys():
        c620_sp[i] = self.c620_wt_always_same_split_factor_dict[i]
    
    # 計算 c620_wt
    s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
    
    # 如果是線性模式就再update c620 wt一次,放在後處理前
    if self._linear_model:
      c620_wt_real = self.c620_real_data_model_linear.predict(c620_input).iloc[:,:41*4]
      c620_wt_real = pd.DataFrame(c620_wt_real,index=c620_input.index,columns=c620_wt.columns)
      c620_wt.update(c620_wt_real)
    
    # c620_wt 後處理 為了在輸出之前滿足業主給的約束條件
    if self._Post_processing:
      bz_idx = c620_wt.columns.tolist().index('Tatoray Stripper C620 Operation_Sidedraw Production Rate and Composition_Benzene_wt%')
      other_idx = [i for i in range(41*2,41*3,1) if i != bz_idx]
      other_total = (100 - c620_input['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'].values).reshape(-1,1)
      c620_wt.iloc[:,bz_idx] = c620_input['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'].values
      c620_wt.iloc[:,other_idx] = (c620_wt.iloc[:,other_idx].values /
                                   c620_wt.iloc[:,other_idx].values.sum(axis=1).reshape(-1,1))*other_total

    
    # c620 input mass flow rate m3 to ton
    V615_Btm_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from V615 Btm_m3/hr'].values.reshape(-1,1)
    C820_Dist_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from C820 Dist_m3/hr'].values.reshape(-1,1)
    V615_Btm_ton = V615_Btm_m3*self.V615_density
    C820_Dist_ton = C820_Dist_m3*self.C820_density
    c620_feed_rate_ton = V615_Btm_ton+C820_Dist_ton
    
    # c620 output mass flow ton
    c620_mf_side = np.sum(c620_feed_rate_ton*c620_feed.values*s3*0.01,axis=1,keepdims=True)
    c620_mf_bot = np.sum(c620_feed_rate_ton*c620_feed.values*s4*0.01,axis=1,keepdims=True)

    # t651 feed mass flow rate(ton)
    t651_mf = (icg_input['Simulation Case Conditions_Feed Rate_Feed from T651_m3/hr']*self.T651_density).values.reshape(-1,1)

    # c660 input mass flow(ton)
    c660_mf = t651_mf + c620_mf_side
    t651_mf_p ,c620_mf_side_p = t651_mf/c660_mf ,c620_mf_side/c660_mf

    # c660 input(feed & case)
    c660_feed = c620_wt[self.c620_col['sidedraw_x']].values*c620_mf_side_p + t651_feed.values*t651_mf_p
    c660_feed = pd.DataFrame(c660_feed,index=idx,columns=self.c660_col['x41'])
    c660_case = pd.DataFrame(index=idx,columns=self.c660_col['case'])
    c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'] = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values
    
    if self.Recommended_mode == True:
      # fix Toluene in Benzene_ppmw = 10
      c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = 10.0
    
    if self.Recommended_mode == False:
      # Toluene in Benzene_ppmw = user input
      c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values
    
    c660_input = c660_case.join(c660_feed)
    
    # c660 output(op&wt)
    c660_output = self.c660_model.predict(c660_input)
    c660_sp,c660_op = c660_output.iloc[:,:41*4],c660_output.iloc[:,41*4:]

    # update by c660 real data model?
    if self.real_data_mode == True:
      if self._linear_model == True:
        c660_op_real = self.c660_real_data_model_linear.predict(c660_input)[:,41*4:]
        c660_op_real = pd.DataFrame(c660_op_real,index=c660_input.index,columns=self.c660_simple_op_col)
        c660_sp_real = self.c660_real_data_model_linear.predict(c660_input)[:,:41*4]
        c660_sp_real = pd.DataFrame(c660_sp_real,index=c660_input.index,columns=c660_sp.columns)
      if self._linear_model == False:
        c660_op_real = self.c660_real_data_model.predict(c660_input).iloc[:,41*4:] #操作條件放後面
        c660_sp_real = self.c660_real_data_model.predict(c660_input).iloc[:,:41*4] #分離係數放前面
      c660_op.update(c660_op_real)
      c660_sp.update(c660_sp_real)
    
    # c660 sp後處理
    if self._Post_processing:
      for i in self.c660_wt_always_same_split_factor_dict.keys():
        c660_sp[i] = self.c660_wt_always_same_split_factor_dict[i]
    
    # 計算 c660_wt
    s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
    
    
    # 如果是線性模式就再update c660 wt一次,放在後處理前
    if self._linear_model:
      c660_wt_real = self.c660_real_data_model_linear.predict(c660_input).iloc[:,:41*4]
      c660_wt_real = pd.DataFrame(c660_wt_real,index=c660_input.index,columns=c660_wt.columns)
      c660_wt.update(c660_wt_real)
    
    # c660_wt 後處理 為了在輸出之前滿足業主給的約束條件
    if self._Post_processing:
      na_idx = [1,2,3,4,5,6,8,9,11,13,14,15,20,22,29] 
      other_idx = list(set([*range(41)])-set(na_idx))
      na_total = (c660_input['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'].values/10000).reshape(-1,1)
      other_total = 100 - na_total
      c660_wt.iloc[:,41*2:41*3].iloc[:,na_idx] = (c660_wt.iloc[:,41*2:41*3].iloc[:,na_idx].values/
                                                  c660_wt.iloc[:,41*2:41*3].iloc[:,na_idx].values.sum(axis=1).reshape(-1,1))*na_total
      c660_wt.iloc[:,41*2:41*3].iloc[:,other_idx] = (c660_wt.iloc[:,41*2:41*3].iloc[:,other_idx].values/
                                                     c660_wt.iloc[:,41*2:41*3].iloc[:,other_idx].values.sum(axis=1).reshape(-1,1))*other_total
    
    # c660 output mass flow (ton)
    c660_mf_bot = np.sum(c660_mf*c660_feed.values*s4*0.01,axis=1,keepdims=True)
    
    # c670 input mass flow
    c670_mf = c620_mf_bot + c660_mf_bot
    c620_mf_bot_p,c660_mf_bot_p = c620_mf_bot/c670_mf , c660_mf_bot/c670_mf
    
    # c670 feed wt%
    c670_feed = c620_wt[self.c620_col['bottoms_x']].values*c620_mf_bot_p + c660_wt[self.c660_col['bottoms_x']].values*c660_mf_bot_p
    c670_feed = pd.DataFrame(c670_feed,index=idx,columns=self.c670_col['combined'])

    c670_bf = pd.DataFrame(index=idx,columns=self.c670_col['upper_bf'])
    c620_bot_x = c620_wt[self.c620_col['bottoms_x']].values
    c660_bot_x = c660_wt[self.c660_col['bottoms_x']].values
    upper_bf = (c660_bot_x*c660_mf_bot)/(c620_bot_x*c620_mf_bot+c660_bot_x*c660_mf_bot)
    upper_bf = pd.DataFrame(upper_bf,index=idx,columns=self.c670_col['upper_bf'])
    upper_bf[list(set(self.index_9999)&set(upper_bf.columns))] = 0.9999
    upper_bf[list(set(self.index_0001)&set(upper_bf.columns))] = 0.0001
    
    # c670 input (feed%bf)
    c670_input = c670_feed.join(upper_bf)
    c670_output = self.c670_model.predict(c670_input)
    c670_sp,c670_op = c670_output.iloc[:,:41*2],c670_output.iloc[:,41*2:]

    # update by c670 real data model?
    if self.real_data_mode == True:
      if self._linear_model == True:
        c670_op_real = self.c670_real_data_model_linear.predict(c670_input)[:,41*2:]
        c670_op_real = pd.DataFrame(c670_op_real,index=c670_input.index,columns=self.c670_simple_op_col)
        c670_sp_real = self.c670_real_data_model_linear.predict(c670_input)[:,:41*2]
        c670_sp_real = pd.DataFrame(c670_sp_real,index=c670_input.index,columns=c670_sp.columns)
      if self._linear_model == False:
        c670_op_real = self.c670_real_data_model.predict(c670_input).iloc[:,41*2:] #操作條件放後面
        c670_sp_real = self.c670_real_data_model.predict(c670_input).iloc[:,:41*2] #分離係數放前面
      c670_op.update(c670_op_real)
      c670_sp.update(c670_sp_real)
    
    # c670 sp後處理
    if self._Post_processing:
      for i in self.c670_wt_always_same_split_factor_dict.keys():
        c670_sp[i] = self.c670_wt_always_same_split_factor_dict[i]
    
    s1 = c670_sp[self.c670_col['distillate_sf']].values
    s2 = c670_sp[self.c670_col['bottoms_sf']].values
    w1 = sp2wt(c670_feed,s1)
    w2 = sp2wt(c670_feed,s2)
    c670_wt = np.hstack((w1,w2))
    c670_wt = pd.DataFrame(c670_wt,index = idx,columns=self.c670_col['distillate_x']+self.c670_col['bottoms_x'])
    
    # 如果是線性模式就再update c670 wt一次,放在後處理前
    if self._linear_model:
      c670_wt_real = self.c670_real_data_model_linear.predict(c670_input).iloc[:,:41]
      c670_wt_real = pd.DataFrame(c670_wt_real,index=c670_input.index,columns=c670_wt.columns)
      c670_wt.update(c670_wt_real)
    
    # c670wt沒有後處理
    
    return c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op
