# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 09:27:48 2022

@author: Sia Stroemsnes & Marie Staveland
"""

import pandas as pd
from datetime import datetime,timedelta


wave_fol="/Volumes/LaCie/Master Thesis/Wave_parquet/"
wind_fol="/Volumes/LaCie/Master Thesis/Wind_parquet/"

timestart=datetime(2008,1,1,0,0,0)

for timestart in pd.date_range(start='2008/4/3', end='2020/1/1',inclusive="left",freq="D"):
    
    timestart=timestart.date().strftime("%Y-%m-%d")
    
    wave_data=pd.read_parquet(wave_fol+"voyage_started_"+timestart)
    
    wind_data=pd.read_parquet(wind_fol+"voyage_started_"+timestart)
    wind_data=wind_data[["time_xa","route_name","starting_date",'app_wind_dir','wind_speed_kts','time_wind_xa']]
    merged=pd.merge(wave_data,wind_data,on=["time_xa","route_name","starting_date"],how="left")
    
    
    merged.to_parquet("/Volumes/LaCie/Master Thesis/Merged_Routes/voyage_started_"+timestart)