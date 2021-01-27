from F import F
from config import config
import joblib
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    
    # load data
    demo = joblib.load('./data/demo.pkl')
    print(demo.keys())
    icg_input = demo['icg_input']
    c620_Receiver_Temp = demo['c620_Receiver_Temp']
    c620_feed = demo['c620_feed']
    t651_feed = demo['t651_feed']
    c620_dist_rate = demo['c620_case_Distillate_Rate']
    c660_case_Toluene_in_Benzene_ppmw = demo['c660_case_Toluene_in_Benzene_ppmw']

    # instance f
    f = F(config)
    f.Recommended_mode = True #(True or False)
    
    # call f
    if f.Recommended_mode == True:
        c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op = f(icg_input,c620_Receiver_Temp,c620_feed,t651_feed)
    
    if f.Recommended_mode == False:
        other_args =  {
            'c620_case_Distillate_Rate':c620_dist_rate,
            'c660_case_Toluene_in_Benzene_ppmw':c660_case_Toluene_in_Benzene_ppmw,
            }
        c620_wt,c620_op,c660_wt,c660_op,c670_wt,c670_op = f(icg_input,c620_Receiver_Temp,c620_feed,t651_feed,other_args)
    
    # print(input and output)
    print('icg_input',icg_input)
    print('c620_Receiver_Temp',c620_Receiver_Temp)
    print('c620_feed',c620_feed)
    print('t651_feed',t651_feed)
    print('c620_wt',c620_wt)
    print('c620_op',c620_op)
    print('c660_wt',c660_wt)
    print('c660_op',c660_op)
    print('c670_wt',c670_wt)
    print('c670_op',c670_op)
    print('Recommended_mode = ',f.Recommended_mode)
    print('test ok !')