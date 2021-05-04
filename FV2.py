import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autorch
from autorch.function import sp2wt
import optuna
import torch
from tqdm import tqdm_notebook as tqdm

class AllSystem(object):
  def __init__(self,config):
    
    # C620 部份模組
    self.c620_G = joblib.load(config['c620_G'])
    self.c620_F = joblib.load(config['c620_F'])
    self.c620_op_min = joblib.load(config['c620_op_min'])
    self.c620_op_max = joblib.load(config['c620_op_max'])
    
    # C660 部份模組
    self.c660_G = joblib.load(config['c660_G'])
    self.c660_F = joblib.load(config['c660_F'])
    self.c660_op_min = joblib.load(config['c660_op_min'])
    self.c660_op_max = joblib.load(config['c660_op_max'])

    # C670 部份模組
    self.c670_M = joblib.load(config['c670_M'])
    
    # 用來修正現場儀表的誤差
    self.op_fix_model = joblib.load(config['op_fix_model'])
    self.op_bias = {'Benzene Column C660 Operation_Column Temp Profile_C660 Tray 23 (Control)_oC': -0.5616992712020874,
 'Benzene Column C660 Operation_Column Temp Profile_C660 Tray 6 (SD & Control)_oC': -0.5311859250068665,
 'Benzene Column C660 Operation_Yield Summary_Reflux Rate_m3/hr': -23.2098445892334,
 'Density_Bottoms Production Rate and Composition': 0.003680689726024866,
 'Density_Distillate (Benzene Drag) Production Rate and Composition': -0.002266152761876583,
 'Density_Distillate Production Rate and Composition': 0.00015364743012469262,
 'Density_Feed Properties': 0.0001229307526955381,
 'Density_Sidedraw (Benzene )Production Rate and Composition': 0.0012807963648810983,
 'Density_Sidedraw Production Rate and Composition': 0.0005597509443759918,
 'Density_Vent Gas Production Rate and Composition': 0.015023279003798962,
 'Tatoray Stripper C620 Operation_Column Temp Profile_C620 Tray 14 (Control)_oC': -14.776540756225586,
 'Tatoray Stripper C620 Operation_Column Temp Profile_C620 Tray 34 (Control)_oC': -12.766730308532715,
 'Tatoray Stripper C620 Operation_Yield Summary_Reflux Rate_m3/hr': -1.447864055633545,
 'Toluene Column C670 Operation_Column Temp Profile_C670 Btm Temp (Control)_oC': -2.3342103958129883,
 'Toluene Column C670 Operation_Column Temp Profile_C670 Tray 24 (Control)_oC': 0.29816916584968567,
 'Toluene Column C670 Operation_Yield \nSummary_Reflux Rate_m3/hr': -55.309078216552734}

    # 欄位名稱列表
    self.icg_col = joblib.load(config['icg_col_path'])
    self.c620_col = joblib.load(config['c620_col_path'])
    self.c660_col = joblib.load(config['c660_col_path'])
    self.c670_col = joblib.load(config['c670_col_path'])
    
    # 其他資訊
    self.c620_wt_always_same_split_factor_dict = joblib.load(config['c620_wt_always_same_split_factor_dict'])
    self.c660_wt_always_same_split_factor_dict = joblib.load(config['c660_wt_always_same_split_factor_dict'])
    self.c670_wt_always_same_split_factor_dict = joblib.load(config['c670_wt_always_same_split_factor_dict'])
    self.index_9999 = joblib.load(config['index_9999_path'])
    self.index_0001 = joblib.load(config['index_0001_path'])

    # 由廠區方提供的平均密度用來將體積流量轉成重量流量
    self.V615_density = 0.8626
    self.C820_density = 0.8731
    self.T651_density = 0.8749

    # 操作欄位名稱列表
    self.c620_op_col = self.c620_G.y_col # G系列模型本身就是預測操作條件
    self.c660_op_col = self.c660_G.y_col # G系列模型本身就是預測操作條件
    self.c670_op_col = self.c670_M.y_col[41*2:] #前面82個是分離係數後面都是操作條件

    # 廠區通常調整欄位列表
    self.c620_op_col_can_change = ['Tatoray Stripper C620 Operation_Column Temp Profile_C620 Tray 14 (Control)_oC','Tatoray Stripper C620 Operation_Column Temp Profile_C620 Tray 34 (Control)_oC']
    self.c660_op_col_can_change = ['Benzene Column C660 Operation_Column Temp Profile_C660 Tray 6 (SD & Control)_oC','Benzene Column C660 Operation_Column Temp Profile_C660 Tray 23 (Control)_oC']
  
  def inference(self,icg_input,c620_feed,t651_feed,
                real_data_mode = False #修正現場儀表誤差用
                ):
    
    # 紀錄樣本index
    idx = icg_input.index
    
    # 計算c620_case
    c620_case = pd.DataFrame(index=idx,columns=self.c620_col['case'])
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'] = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values
    
    # 預測c620操作條件和分離係數
    c620_op = self.c620_G.predict(c620_case.join(c620_feed))
    c620_sp = self.c620_F.predict(c620_case.join(c620_feed).join(c620_op))
    
    # 計算c620輸出組成
    s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
    
    #計算c660_feed
    V615_Btm_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from V615 Btm_m3/hr'].values.reshape(-1,1)
    C820_Dist_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from C820 Dist_m3/hr'].values.reshape(-1,1)
    V615_Btm_ton = V615_Btm_m3*self.V615_density
    C820_Dist_ton = C820_Dist_m3*self.C820_density
    c620_feed_rate_ton = V615_Btm_ton+C820_Dist_ton
    c620_mf_side = np.sum(c620_feed_rate_ton*c620_feed.values*s3*0.01,axis=1,keepdims=True)
    c620_mf_bot = np.sum(c620_feed_rate_ton*c620_feed.values*s4*0.01,axis=1,keepdims=True)
    t651_mf = (icg_input['Simulation Case Conditions_Feed Rate_Feed from T651_m3/hr']*self.T651_density).values.reshape(-1,1)
    c660_mf = t651_mf + c620_mf_side
    t651_mf_p ,c620_mf_side_p = t651_mf/c660_mf ,c620_mf_side/c660_mf
    c660_feed = c620_wt[self.c620_col['sidedraw_x']].values*c620_mf_side_p + t651_feed.values*t651_mf_p
    c660_feed = pd.DataFrame(c660_feed,index=idx,columns=self.c660_col['x41'])
    
    # 計算c660_case
    c660_case = pd.DataFrame(index=idx,columns=self.c660_col['case'])
    c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'] = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values
    c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values
    
    # 預測c660操作條件和分離係數
    c660_op = self.c660_G.predict(c660_case.join(c660_feed))
    c660_sp = self.c660_F.predict(c660_case.join(c660_feed).join(c660_op))
    
    # 計算c660輸出組成
    s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
    
    # 計算c670_feed
    c660_mf_bot = np.sum(c660_mf*c660_feed.values*s4*0.01,axis=1,keepdims=True)
    c670_mf = c620_mf_bot + c660_mf_bot
    c620_mf_bot_p,c660_mf_bot_p = c620_mf_bot/c670_mf , c660_mf_bot/c670_mf
    c670_feed = c620_wt[self.c620_col['bottoms_x']].values*c620_mf_bot_p + c660_wt[self.c660_col['bottoms_x']].values*c660_mf_bot_p
    c670_feed = pd.DataFrame(c670_feed,index=idx,columns=self.c670_col['combined'])
    
    # 計算c670_upper_bf
    c670_bf = pd.DataFrame(index=idx,columns=self.c670_col['upper_bf'])
    c620_bot_x = c620_wt[self.c620_col['bottoms_x']].values
    c660_bot_x = c660_wt[self.c660_col['bottoms_x']].values
    upper_bf = (c660_bot_x*c660_mf_bot)/(c620_bot_x*c620_mf_bot+c660_bot_x*c660_mf_bot)
    upper_bf = pd.DataFrame(upper_bf,index=idx,columns=self.c670_col['upper_bf'])
    upper_bf[list(set(self.index_9999)&set(upper_bf.columns))] = 0.9999
    upper_bf[list(set(self.index_0001)&set(upper_bf.columns))] = 0.0001
    
    # 直接預測分離係數和操作條件
    c670_output = self.c670_M.predict(c670_feed.join(upper_bf))
    c670_sp,c670_op = c670_output.iloc[:,:41*2],c670_output.iloc[:,41*2:]    
    
    # 計算輸出組成
    s1,s2 = c670_sp[self.c670_col['distillate_sf']].values,c670_sp[self.c670_col['bottoms_sf']].values
    w1,w2 = sp2wt(c670_feed,s1),sp2wt(c670_feed,s2)
    c670_wt = pd.DataFrame(np.hstack((w1,w2)),index = idx,columns=self.c670_col['distillate_x']+self.c670_col['bottoms_x'])
    
    # 是否修正操作條件 for 現場數據
    if real_data_mode == False:
      return c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op
    
    if real_data_mode == True:
      # 有些欄位現場數據沒有
      c620_op_col = c620_op.drop(['Tatoray Stripper C620 Operation_Heat Duty_Condenser Heat Duty_Mkcal/hr',
                                 'Tatoray Stripper C620 Operation_Heat Duty_Reboiler Heat Duty_Mkcal/hr'],
                                 axis=1).columns.tolist()
      c660_op_col = c660_op.drop(['Benzene Column C660 Operation_Heat Duty_Condenser Heat Duty_Mkcal/hr',
                                 'Benzene Column C660 Operation_Heat Duty_Reboiler Heat Duty_Mkcal/hr'],
                                 axis=1).columns.tolist()
      c670_op_col = c670_op.drop(['Toluene Column C670 Operation_Heat Duty_Condenser Heat Duty_Mkcal/hr',
                                 'Toluene Column C670 Operation_Heat Duty_Reboiler Heat Duty_Mkcal/hr'],
                                 axis=1).columns.tolist()
      
      # 直接放偏移上去
      op_pred = c620_op.join(c660_op).join(c670_op)
      for k,v in self.op_bias.items():
        op_pred[k] += v 
      
      # 新的op
      new_c620_op = op_pred.iloc[:,:8]
      new_c660_op = op_pred.iloc[:,8:16]
      new_c670_op = op_pred.iloc[:,-5:]
      new_c620_op.columns = c620_op_col
      new_c660_op.columns = c660_op_col
      new_c670_op.columns = c670_op_col
      
      # 更新op
      c620_op.update(new_c620_op)
      c660_op.update(new_c660_op)
      c670_op.update(new_c670_op)
      
      return c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op
  
  def recommend(self,icg_input,c620_feed,t651_feed,
                search_iteration = 300, # cma-es優化搜索次數
                real_data_mode = False, # 如果打開這個功能則會把操作條件在經過一個修正模組來配合現場儀表偏移
                auto_set_recommended_value = [70,980,10], #自動設定推薦值
                only_tune_temp = False, # 是否只調溫度,如果其他也可以調可能可以比較快找到理想的解,否則可能找不到理想的解
                ):
    
    self.only_tune_temp = only_tune_temp
    # 先用試算模式試算一遍
    idx = icg_input.index
    c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op = self.inference(icg_input,c620_feed,t651_feed,real_data_mode=real_data_mode) 

    # 紀錄該筆樣本原始的Benzene in C620 Sidedraw_wt%
    original_Benzene_in_C620_Sidedraw_wt = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values[0]
    
    # 是否自動設定推薦值
    if len(auto_set_recommended_value) == 3:
      icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'] = auto_set_recommended_value[0]
      icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'] = auto_set_recommended_value[1]
      icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = auto_set_recommended_value[2]
      print(f'系統已經自動設定推薦值[{auto_set_recommended_value[0]},{auto_set_recommended_value[1]},{auto_set_recommended_value[2]}]')
    
    # 根據推薦的Benzene in C620 Sidedraw_wt%和原始的Benzene in C620 Sidedraw_wt%去比較來判斷說溫度要往上調還是往下調
    if len(auto_set_recommended_value) == 3:
      # 代表現在工程師想要把原本的Benzene_in_C620_Sidedraw_wt往上拉所以溫度調幅應該往下
      if original_Benzene_in_C620_Sidedraw_wt < auto_set_recommended_value[0]:
        self.temp_adjust_direction = 'down'
      # 代表現在工程師想要把原本的Benzene_in_C620_Sidedraw_wt往下降所以溫度調幅應該往上
      if original_Benzene_in_C620_Sidedraw_wt > auto_set_recommended_value[0]:
        self.temp_adjust_direction = 'up'
      # 如果兩者一樣就不特別規定調幅要往上還往下,任意即可
      if original_Benzene_in_C620_Sidedraw_wt == auto_set_recommended_value[0]:
        self.temp_adjust_direction = 'arbitrary'
    
    # c620 case設置
    c620_case = pd.DataFrame(index=idx,columns=self.c620_col['case'])
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'] = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values
    
    # c660 case設置
    c660_case = pd.DataFrame(index=idx,columns=self.c660_col['case'])
    c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'] = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values
    c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values
    
    # cma-es 初始值設置 操作條件初始值來自於試算值
    x0 = {}
    for name in self.c620_op_col:
      x0[name] = c620_op[name].values[0]
    for name in self.c660_op_col:
      x0[name] = c660_op[name].values[0]
    
    # 這一樣Distillate Rate_m3/hr的初始值來自於使用者輸入
    x0['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values[0]
    
    # 建立 cma-es study 並賦予初始值X0
    sampler = optuna.samplers.CmaEsSampler(x0=x0,sigma0=1.0,restart_strategy='ipop',seed=42)
    study = optuna.create_study(sampler=sampler)
    
    # cma-es 搜索階段
    history = {}
    for i in tqdm(range(search_iteration)):
      trial = study.ask()
      
      # 1.在c620_op的向量空間作採樣
      c620_op_opt_dict = {}
      minimum_amplitude = 0.1
      for name in self.c620_op_col:
        # 如果這些欄位屬於可以調的 例如廠區通常調整溫度
        if name in self.c620_op_col_can_change:
          if self.temp_adjust_direction == 'down':
            try:
              c620_op_opt_dict[name] = trial.suggest_uniform(name,self.c620_op_min[name],c620_op[name].values[0]-minimum_amplitude) #最少應該降minimum_amplitude度
            except:
              print('溫度已經低於訓練數據的最小值不能再低了')
              c620_op_opt_dict[name] = self.c620_op_min[name] # 所以取最小值
          if self.temp_adjust_direction == 'up':
            try:
              c620_op_opt_dict[name] = trial.suggest_uniform(name,c620_op[name].values[0]+minimum_amplitude,self.c620_op_max[name]) #最少應該增minimum_amplitude度
            except:
              print('溫度已經低於訓練數據的最大值不能再高了')
              c620_op_opt_dict[name] = self.c620_op_max[name] # 所以取最大值
          if self.temp_adjust_direction == 'arbitrary':
            c620_op_opt_dict[name] = trial.suggest_uniform(name,self.c620_op_min[name],self.c620_op_max[name]) #任意調
        # 比較不能調的,看要不要調,可以調的話找到理想解的可能性會比較高
        else:
          if self.only_tune_temp == True:
            c620_op_opt_dict[name] = c620_op[name].values[0] 
          if self.only_tune_temp == False:
            c620_op_opt_dict[name] = trial.suggest_uniform(name,self.c620_op_min[name],self.c620_op_max[name]) #任意調
      c620_op_opt = pd.DataFrame(c620_op_opt_dict,index=idx)
      
      # 2.在c660_op的向量空間作採樣,但由於c660目前沒有跟c620一樣的苯和溫度呈現負相關這樣的限制,暫時沒有特別去設計調幅方向
      c660_op_opt_dict = {}
      for name in self.c660_op_col:
        if name in self.c660_op_col_can_change:
          c660_op_opt_dict[name] = trial.suggest_uniform(name,self.c660_op_min[name],self.c660_op_max[name])
        
        # 比較不能調的,看要不要調,可以調的話找到理想解的可能性會比較高
        else:
          if self.only_tune_temp == True:
            c660_op_opt_dict[name] = c660_op[name].values[0] 
          if self.only_tune_temp == False:
            c660_op_opt_dict[name] = trial.suggest_uniform(name,self.c660_op_min[name],self.c660_op_max[name]) #任意調
      c660_op_opt = pd.DataFrame(c660_op_opt_dict,index=idx)
      
      # 3.在Operation_Specifications_Spec 2 : Distillate Rate_m3/hr的向量空間做採樣,並將採樣結果代入c620_case
      c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = trial.suggest_float(
          'Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr',0,10.25)
      
      # 計算c620輸出wt%
      c620_sp = self.c620_F.predict(c620_case.join(c620_feed).join(c620_op_opt))
      s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
      w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
      wt = np.hstack((w1,w2,w3,w4))
      c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
      
      # 計算c660輸入_wt%
      V615_Btm_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from V615 Btm_m3/hr'].values.reshape(-1,1) # V615體積流量
      C820_Dist_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from C820 Dist_m3/hr'].values.reshape(-1,1) # C820體積流量
      V615_Btm_ton = V615_Btm_m3*self.V615_density # V615體積流量轉重量流量
      C820_Dist_ton = C820_Dist_m3*self.C820_density # C820體積流量轉重量流量
      c620_feed_rate_ton = V615_Btm_ton+C820_Dist_ton # V615和C820兩道油相加得到c620總入料的"質量流量"
      c620_mf_side = np.sum(c620_feed_rate_ton*c620_feed.values*s3*0.01,axis=1,keepdims=True) # 根據公式換算出c620_side流量
      c620_mf_bot = np.sum(c620_feed_rate_ton*c620_feed.values*s4*0.01,axis=1,keepdims=True) # 根據公式換算出c620_bott流量
      t651_mf = (icg_input['Simulation Case Conditions_Feed Rate_Feed from T651_m3/hr']*self.T651_density).values.reshape(-1,1) # t651質量流量轉成重量流量
      c660_mf = t651_mf + c620_mf_side # t651和c620_side兩道油的質量流量相加得到c660入料質量流量
      t651_mf_p ,c620_mf_side_p = t651_mf/c660_mf ,c620_mf_side/c660_mf # t651和c620_side兩道油對c660總入料質量流量的占比百分比
      c660_feed = c620_wt[self.c620_col['sidedraw_x']].values * c620_mf_side_p + t651_feed.values * t651_mf_p #根據公式換算c660混合入料組成
      c660_feed = pd.DataFrame(c660_feed,index=idx,columns=self.c660_col['x41']) # 將c660入料組成轉換成dataframe格式
      
      # 計算c660輸出wt%
      c660_sp = self.c660_F.predict(c660_case.join(c660_feed).join(c660_op_opt))
      s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
      w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
      wt = np.hstack((w1,w2,w3,w4))
      c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
      
      # 計算損失
      input_bzinside = c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'].values[0]
      input_nainbz = c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'].values[0]
      input_tol = c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values[0]
      input_dist_rate = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values[0]
      
      output_bzinside = c620_wt['Tatoray Stripper C620 Operation_Sidedraw Production Rate and Composition_Benzene_wt%'].values[0]
      output_nainbz = c660_wt.filter(regex='Side').filter(regex='wt%').iloc[:,[1,2,3,4,5,6,8,9,11,13,14,15,20,22,29]].sum(axis=1).values[0]*10000
      output_tol = c660_wt['Benzene Column C660 Operation_Sidedraw (Benzene )Production Rate and Composition_Toluene_wt%'].values[0]*10000
      output_dist_rate = c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values[0]
      
      bzinside_loss = abs(input_bzinside - output_bzinside) / (input_bzinside+1e-8) #絕對百分比誤差
      nainbz_loss = abs(input_nainbz - output_nainbz) / (input_nainbz+1e-8) #絕對百分比誤差
      tol_loss = abs(input_tol - output_tol) / (input_tol+1e-8) #絕對百分比誤差
      distrate_loss = max(output_dist_rate - input_dist_rate,0) # distrate根據廠區說法愈小愈好,因此如果採樣出的如果採樣出的dist_rate大於input_dist_rate就會有loss,否則為0
      total_loss = np.max([bzinside_loss,nainbz_loss,tol_loss,distrate_loss])
      # 把總損失告訴study供下一次採樣的依據
      study.tell(trial,total_loss)

      if i%10 == 0:
        print(f'epoch:{i}')
        performance = pd.DataFrame(index=[i],columns=['output_bzinside','output_nainbz','output_tol','output_dist_rate'])
        performance['output_bzinside'] = output_bzinside
        performance['output_nainbz'] = output_nainbz
        performance['output_tol'] = output_tol
        performance['output_dist_rate'] = output_dist_rate
        performance['max_loss'] = np.max([bzinside_loss,nainbz_loss,tol_loss,distrate_loss])
        print(performance)

      # 如果滿足以下條件就算提早成功
      if (bzinside_loss <= 0.01) and (nainbz_loss <= 0.01) and (tol_loss <= 0.01) and (distrate_loss == 0):
        print('Congratulations Early Success find optimal op')
        break
    
    # cma-es搜索迴圈跑完或是提早結束,調出study.best_params並製作成best_params_df
    best_params_df = pd.DataFrame(
        study.best_params,
        index = idx,
        columns = self.c620_op_col + self.c660_op_col + ['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'])
    
    # 優化過的c620_op,c660_op,c620_case(dist_rate)
    c620_op_opt = c620_op.drop(self.c620_op_col_can_change,axis=1).join(best_params_df[self.c620_op_col][self.c620_op_col_can_change])
    c660_op_opt = c660_op.drop(self.c660_op_col_can_change,axis=1).join(best_params_df[self.c660_op_col][self.c660_op_col_can_change])
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = best_params_df['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr']
    
    # 計算c620輸出wt%
    c620_sp = self.c620_F.predict(c620_case.join(c620_feed).join(c620_op_opt))
    s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
    
    # 計算c660_feed_wt%
    V615_Btm_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from V615 Btm_m3/hr'].values.reshape(-1,1) #體積流量
    C820_Dist_m3 = icg_input['Simulation Case Conditions_Feed Rate_Feed from C820 Dist_m3/hr'].values.reshape(-1,1) #體積流量
    V615_Btm_ton = V615_Btm_m3*self.V615_density # 體積流量轉重量流量
    C820_Dist_ton = C820_Dist_m3*self.C820_density # 體積流量轉重量流量
    c620_feed_rate_ton = V615_Btm_ton+C820_Dist_ton # 兩股相加得到c620總入料重量流量
    c620_mf_side = np.sum(c620_feed_rate_ton*c620_feed.values*s3*0.01,axis=1,keepdims=True) # c620_side 流量
    c620_mf_bot = np.sum(c620_feed_rate_ton*c620_feed.values*s4*0.01,axis=1,keepdims=True) # c620_bott 流量
    t651_mf = (icg_input['Simulation Case Conditions_Feed Rate_Feed from T651_m3/hr']*self.T651_density).values.reshape(-1,1) # t651重量流量
    c660_mf = t651_mf + c620_mf_side # 兩股相加得到c660入料重量流量
    t651_mf_p ,c620_mf_side_p = t651_mf/c660_mf ,c620_mf_side/c660_mf # 兩股油源的占比百分比
    c660_feed = c620_wt[self.c620_col['sidedraw_x']].values*c620_mf_side_p + t651_feed.values*t651_mf_p #計算c660入料組成
    c660_feed = pd.DataFrame(c660_feed,index=idx,columns=self.c660_col['x41']) #計算c660入料組成(dataframe格式)

    # 計算c660輸出wt%
    c660_sp = self.c660_F.predict(c660_case.join(c660_feed).join(c660_op_opt))
    s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
    
    # 計算損失
    input_bzinside = c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'].values[0] 
    input_nainbz = c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'].values[0] 
    input_tol = c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values[0]
    input_dist_rate = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values[0]
    
    output_bzinside = c620_wt['Tatoray Stripper C620 Operation_Sidedraw Production Rate and Composition_Benzene_wt%'].values[0] 
    output_nainbz = c660_wt.filter(regex='Side').filter(regex='wt%').iloc[:,[1,2,3,4,5,6,8,9,11,13,14,15,20,22,29]].sum(axis=1).values[0]*10000 
    output_tol = c660_wt['Benzene Column C660 Operation_Sidedraw (Benzene )Production Rate and Composition_Toluene_wt%'].values[0]*10000
    output_dist_rate = c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values[0]
    
    bzinside_loss = abs(input_bzinside - output_bzinside) / input_bzinside
    nainbz_loss = abs(input_nainbz - output_nainbz) / input_nainbz
    tol_loss = abs(input_tol - output_tol) / input_tol
    distrate_loss = max(output_dist_rate - input_dist_rate,0)
    
    # 打印優化結果
    print('bzinside_loss:',bzinside_loss)
    print('nainbz_loss:',nainbz_loss)
    print('tol_loss:',tol_loss)
    print('distrate_loss:',distrate_loss)

    # c670 部份
    c660_mf_bot = np.sum(c660_mf*c660_feed.values*s4*0.01,axis=1,keepdims=True) # c660_bot 重量流量
    c670_mf = c620_mf_bot + c660_mf_bot # c620_bot 重量流量 加上 c660_bot 重量流量 得到c670總入料重量流量
    c620_mf_bot_p,c660_mf_bot_p = c620_mf_bot/c670_mf , c660_mf_bot/c670_mf # 兩股油源的百分比占比
    c670_feed = c620_wt[self.c620_col['bottoms_x']].values*c620_mf_bot_p + c660_wt[self.c660_col['bottoms_x']].values*c660_mf_bot_p # c670入料組成
    c670_feed = pd.DataFrame(c670_feed,index=idx,columns=self.c670_col['combined'])
    
    # c670 upper_bf 計算
    c670_bf = pd.DataFrame(index=idx,columns=self.c670_col['upper_bf']) 
    c620_bot_x = c620_wt[self.c620_col['bottoms_x']].values
    c660_bot_x = c660_wt[self.c660_col['bottoms_x']].values
    upper_bf = (c660_bot_x*c660_mf_bot)/(c620_bot_x*c620_mf_bot+c660_bot_x*c660_mf_bot)
    upper_bf = pd.DataFrame(upper_bf,index=idx,columns=self.c670_col['upper_bf'])
    upper_bf[list(set(self.index_9999)&set(upper_bf.columns))] = 0.9999
    upper_bf[list(set(self.index_0001)&set(upper_bf.columns))] = 0.0001
    
    # c670因為沒有指定wt數值要多少所以直接預測操作條件和分離係數即可
    c670_output = self.c670_M.predict(c670_feed.join(upper_bf))
    c670_sp,c670_op_opt = c670_output.iloc[:,:41*2],c670_output.iloc[:,41*2:]    
    
    # 計算c670_wt
    s1 = c670_sp[self.c670_col['distillate_sf']].values
    s2 = c670_sp[self.c670_col['bottoms_sf']].values
    w1 = sp2wt(c670_feed,s1)
    w2 = sp2wt(c670_feed,s2)
    c670_wt = np.hstack((w1,w2))
    c670_wt = pd.DataFrame(c670_wt,index = idx,columns=self.c670_col['distillate_x']+self.c670_col['bottoms_x'])
    
    # 是否修正操作條件 for 現場數據
    if real_data_mode == False:
      return c620_wt,c620_op_opt,c660_wt,c660_op_opt,c670_wt,c670_op_opt,bzinside_loss,nainbz_loss,tol_loss
    
    if real_data_mode == True:
      # 有些欄位現場數據沒有 因此drop掉
      c620_op_col = c620_op.drop(['Tatoray Stripper C620 Operation_Heat Duty_Condenser Heat Duty_Mkcal/hr',
                                 'Tatoray Stripper C620 Operation_Heat Duty_Reboiler Heat Duty_Mkcal/hr'],
                                 axis=1).columns.tolist()
      c660_op_col = c660_op.drop(['Benzene Column C660 Operation_Heat Duty_Condenser Heat Duty_Mkcal/hr',
                                 'Benzene Column C660 Operation_Heat Duty_Reboiler Heat Duty_Mkcal/hr'],
                                 axis=1).columns.tolist()
      c670_op_col = c670_op.drop(['Toluene Column C670 Operation_Heat Duty_Condenser Heat Duty_Mkcal/hr',
                                 'Toluene Column C670 Operation_Heat Duty_Reboiler Heat Duty_Mkcal/hr'],
                                 axis=1).columns.tolist()
      
      # 直接放偏移上去
      op_pred = c620_op.join(c660_op).join(c670_op)
      for k,v in self.op_bias.items():
        op_pred[k] += v 
      
      # 新的op
      new_c620_op = op_pred.iloc[:,:8]
      new_c660_op = op_pred.iloc[:,8:16]
      new_c670_op = op_pred.iloc[:,-5:]
      new_c620_op.columns = c620_op_col
      new_c660_op.columns = c660_op_col
      new_c670_op.columns = c670_op_col
      
      # 更新op
      c620_op_opt.update(new_c620_op)
      c660_op_opt.update(new_c660_op)
      c670_op_opt.update(new_c670_op)
      
      return c620_wt,c620_op_opt,c660_wt,c660_op_opt,c670_wt,c670_op_opt,bzinside_loss,nainbz_loss,tol_loss
