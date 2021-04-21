import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autorch
from autorch.function import sp2wt
import optuna

class AllSystem(object):
  def __init__(self,config):
    
    # C620 part
    self.c620_G = joblib.load(config['c620_G'])
    self.c620_F = joblib.load(config['c620_F'])
    self.c620_op_min = joblib.load(config['c620_op_min'])
    self.c620_op_max = joblib.load(config['c620_op_max'])
    
    # C660 part
    self.c660_G = joblib.load(config['c660_G'])
    self.c660_F = joblib.load(config['c660_F'])
    self.c660_op_min = joblib.load(config['c660_op_min'])
    self.c660_op_max = joblib.load(config['c660_op_max'])

    # C670 part
    self.c670_M = joblib.load(config['c670_M'])

    # columns name
    self.icg_col = joblib.load(config['icg_col_path'])
    self.c620_col = joblib.load(config['c620_col_path'])
    self.c660_col = joblib.load(config['c660_col_path'])
    self.c670_col = joblib.load(config['c670_col_path'])
    
    # other infomation
    self.c620_wt_always_same_split_factor_dict = joblib.load(config['c620_wt_always_same_split_factor_dict'])
    self.c660_wt_always_same_split_factor_dict = joblib.load(config['c660_wt_always_same_split_factor_dict'])
    self.c670_wt_always_same_split_factor_dict = joblib.load(config['c670_wt_always_same_split_factor_dict'])
    self.index_9999 = joblib.load(config['index_9999_path'])
    self.index_0001 = joblib.load(config['index_0001_path'])
    self.V615_density = 0.8626
    self.C820_density = 0.8731
    self.T651_density = 0.8749
  
  def inference(self,icg_input,c620_feed,t651_feed):
    idx = icg_input.index
    #c620
    c620_case = pd.DataFrame(index=idx,columns=self.c620_col['case'])
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'] = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values
    c620_op = self.c620_G.predict(c620_case.join(c620_feed))
    c620_sp = self.c620_F.predict(c620_case.join(c620_feed).join(c620_op))
    s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
    
    #c660
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
    c660_case = pd.DataFrame(index=idx,columns=self.c660_col['case'])
    c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'] = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values
    c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values
    c660_op = self.c660_G.predict(c660_case.join(c660_feed))
    c660_sp = self.c660_F.predict(c660_case.join(c660_feed).join(c660_op))
    s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
    
    #c670
    c660_mf_bot = np.sum(c660_mf*c660_feed.values*s4*0.01,axis=1,keepdims=True)
    c670_mf = c620_mf_bot + c660_mf_bot
    c620_mf_bot_p,c660_mf_bot_p = c620_mf_bot/c670_mf , c660_mf_bot/c670_mf
    c670_feed = c620_wt[self.c620_col['bottoms_x']].values*c620_mf_bot_p + c660_wt[self.c660_col['bottoms_x']].values*c660_mf_bot_p
    c670_feed = pd.DataFrame(c670_feed,index=idx,columns=self.c670_col['combined'])
    c670_bf = pd.DataFrame(index=idx,columns=self.c670_col['upper_bf'])
    c620_bot_x = c620_wt[self.c620_col['bottoms_x']].values
    c660_bot_x = c660_wt[self.c660_col['bottoms_x']].values
    upper_bf = (c660_bot_x*c660_mf_bot)/(c620_bot_x*c620_mf_bot+c660_bot_x*c660_mf_bot)
    upper_bf = pd.DataFrame(upper_bf,index=idx,columns=self.c670_col['upper_bf'])
    upper_bf[list(set(self.index_9999)&set(upper_bf.columns))] = 0.9999
    upper_bf[list(set(self.index_0001)&set(upper_bf.columns))] = 0.0001
    c670_output = self.c670_M.predict(c670_feed.join(upper_bf))
    c670_sp,c670_op = c670_output.iloc[:,:41*2],c670_output.iloc[:,41*2:]    
    s1 = c670_sp[self.c670_col['distillate_sf']].values
    s2 = c670_sp[self.c670_col['bottoms_sf']].values
    w1 = sp2wt(c670_feed,s1)
    w2 = sp2wt(c670_feed,s2)
    c670_wt = np.hstack((w1,w2))
    c670_wt = pd.DataFrame(c670_wt,index = idx,columns=self.c670_col['distillate_x']+self.c670_col['bottoms_x'])
    return c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op
  
  def recommend(self,icg_input,c620_feed,t651_feed,search_iteration=300):
    idx = icg_input.index
    c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op = self.inference(icg_input,c620_feed,t651_feed)

    # c620 part
    c620_case = pd.DataFrame(index=idx,columns=self.c620_col['case'])
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 1 : Receiver Temp_oC'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'] = icg_input['Tatoray Stripper C620 Operation_Specifications_Spec 2 : Distillate Rate_m3/hr'].values
    c620_case['Tatoray Stripper C620 Operation_Specifications_Spec 3 : Benzene in Sidedraw_wt%'] = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values
    c620_op_col = c620_op.columns.tolist()
    def c620_objective(trial):
      c620_op_dict = {}
      for name in c620_op_col:
        c620_op_dict[name] = trial.suggest_uniform(name,self.c620_op_min[name],self.c620_op_max[name])
      c620_op = pd.DataFrame(c620_op_dict,index=idx)
      輸入端bz = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values[0]
      c620_sp = self.c620_F.predict(c620_case.join(c620_feed).join(c620_op))
      s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
      w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
      wt = np.hstack((w1,w2,w3,w4))
      c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
      輸出端bz = c620_wt['Tatoray Stripper C620 Operation_Sidedraw Production Rate and Composition_Benzene_wt%'].values[0]
      loss = (輸入端bz - 輸出端bz)**2
      return loss
    study = optuna.create_study()
    study.optimize(c620_objective, n_trials=search_iteration)
    c620_op_opt = pd.DataFrame(study.best_params,index=idx)
    c620_op_delta = c620_op_opt - c620_op
    c620_sp = self.c620_F.predict(c620_case.join(c620_feed).join(c620_op_opt))
    s1,s2,s3,s4 = c620_sp.iloc[:,:41].values,c620_sp.iloc[:,41:41*2].values,c620_sp.iloc[:,41*2:41*3].values,c620_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c620_feed,s1),sp2wt(c620_feed,s2),sp2wt(c620_feed,s3),sp2wt(c620_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c620_wt = pd.DataFrame(wt,index=idx,columns=self.c620_col['vent_gas_x']+self.c620_col['distillate_x']+self.c620_col['sidedraw_x']+self.c620_col['bottoms_x'])
    輸入端bz = icg_input['Simulation Case Conditions_Spec 1 : Benzene in C620 Sidedraw_wt%'].values[0]
    輸出端bz = c620_wt['Tatoray Stripper C620 Operation_Sidedraw Production Rate and Composition_Benzene_wt%'].values[0]
    bz_error = abs(輸入端bz-輸出端bz)
    print('bz_error:',bz_error)
    
    # c660 part
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
    c660_case = pd.DataFrame(index=idx,columns=self.c660_col['case'])
    c660_case['Benzene Column C660 Operation_Specifications_Spec 2 : NA in Benzene_ppmw'] = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw']
    c660_case['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'] = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw']
    c660_op_col = c660_op.columns.tolist()
    def c660_objective(trial):
      c660_op_dict = {}
      for name in c660_op_col:
        c660_op_dict[name] = trial.suggest_uniform(name,self.c660_op_min[name],self.c660_op_max[name])
      c660_op = pd.DataFrame(c660_op_dict,index=idx)
      輸入端nainbz = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values[0]
      輸入端tol = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values[0]
      c660_sp = self.c660_F.predict(c660_case.join(c660_feed).join(c660_op))
      s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
      w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
      wt = np.hstack((w1,w2,w3,w4))
      c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
      na_idx = [1,2,3,4,5,6,8,9,11,13,14,15,20,22,29] 
      輸出端nainbz = c660_wt.filter(regex='Side').filter(regex='wt%').iloc[:,na_idx].sum(axis=1).values[0]*10000
      輸出端tol = c660_wt['Benzene Column C660 Operation_Sidedraw (Benzene )Production Rate and Composition_Toluene_wt%'].values[0]*10000
      loss1 = (輸入端nainbz - 輸出端nainbz)**2
      loss2 = (輸入端tol - 輸出端tol)**2
      return loss1+loss2
    study = optuna.create_study()
    study.optimize(c660_objective, n_trials=search_iteration)
    c660_op_opt = pd.DataFrame(study.best_params,index=idx)
    c660_op_delta = c660_op_opt - c660_op
    c660_sp = self.c660_F.predict(c660_case.join(c660_feed).join(c660_op_opt))
    s1,s2,s3,s4 = c660_sp.iloc[:,:41].values,c660_sp.iloc[:,41:41*2].values,c660_sp.iloc[:,41*2:41*3].values,c660_sp.iloc[:,41*3:41*4].values
    w1,w2,w3,w4 = sp2wt(c660_feed,s1),sp2wt(c660_feed,s2),sp2wt(c660_feed,s3),sp2wt(c660_feed,s4)
    wt = np.hstack((w1,w2,w3,w4))
    c660_wt = pd.DataFrame(wt,index=idx,columns=self.c660_col['vent_gas_x']+self.c660_col['distillate_x']+self.c660_col['sidedraw_x']+self.c660_col['bottoms_x'])
    輸入端nainbz = icg_input['Simulation Case Conditions_Spec 2 : NA in Benzene_ppmw'].values[0]
    輸入端tol = icg_input['Benzene Column C660 Operation_Specifications_Spec 3 : Toluene in Benzene_ppmw'].values[0]
    na_idx = [1,2,3,4,5,6,8,9,11,13,14,15,20,22,29] 
    輸出端nainbz = c660_wt.filter(regex='Side').filter(regex='wt%').iloc[:,na_idx].sum(axis=1).values[0]*10000
    輸出端tol = c660_wt['Benzene Column C660 Operation_Sidedraw (Benzene )Production Rate and Composition_Toluene_wt%'].values[0]*10000
    nainbz_error = abs(輸入端nainbz-輸出端nainbz)
    tol_error = abs(輸入端tol-輸出端tol)
    print('nainbz_error:',nainbz_error)
    print('tol_error:',tol_error)

    # c670 part 
    
    c660_mf_bot = np.sum(c660_mf*c660_feed.values*s4*0.01,axis=1,keepdims=True)
    c670_mf = c620_mf_bot + c660_mf_bot
    c620_mf_bot_p,c660_mf_bot_p = c620_mf_bot/c670_mf , c660_mf_bot/c670_mf
    c670_feed = c620_wt[self.c620_col['bottoms_x']].values*c620_mf_bot_p + c660_wt[self.c660_col['bottoms_x']].values*c660_mf_bot_p
    c670_feed = pd.DataFrame(c670_feed,index=idx,columns=self.c670_col['combined'])
    c670_bf = pd.DataFrame(index=idx,columns=self.c670_col['upper_bf'])
    c620_bot_x = c620_wt[self.c620_col['bottoms_x']].values
    c660_bot_x = c660_wt[self.c660_col['bottoms_x']].values
    upper_bf = (c660_bot_x*c660_mf_bot)/(c620_bot_x*c620_mf_bot+c660_bot_x*c660_mf_bot)
    upper_bf = pd.DataFrame(upper_bf,index=idx,columns=self.c670_col['upper_bf'])
    upper_bf[list(set(self.index_9999)&set(upper_bf.columns))] = 0.9999
    upper_bf[list(set(self.index_0001)&set(upper_bf.columns))] = 0.0001
    c670_output = self.c670_M.predict(c670_feed.join(upper_bf))
    c670_sp,c670_op_opt = c670_output.iloc[:,:41*2],c670_output.iloc[:,41*2:]    
    s1 = c670_sp[self.c670_col['distillate_sf']].values
    s2 = c670_sp[self.c670_col['bottoms_sf']].values
    w1 = sp2wt(c670_feed,s1)
    w2 = sp2wt(c670_feed,s2)
    c670_wt = np.hstack((w1,w2))
    c670_wt = pd.DataFrame(c670_wt,index = idx,columns=self.c670_col['distillate_x']+self.c670_col['bottoms_x'])
    c670_op_delta = c670_op_opt - c670_op
    
    return c620_wt,c620_op_opt,c660_wt,c660_op_opt,c670_wt,c670_op_opt,bz_error,nainbz_error,tol_error
