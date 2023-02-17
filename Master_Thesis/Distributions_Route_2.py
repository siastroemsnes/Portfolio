# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 09:50:52 2022

@author: Marie Log Staveland & Sia Benedikte Strømsnes

"""
import pandas as pd
import glob
import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import joblib

# Uploading the parquet files
files = glob.glob("/Volumes/LaCie/Master Thesis/merged_files")
df = pd.concat([pd.read_parquet(fp) for fp in files])

#Dropping year 2020 considering the wind data ends in 2019
df['year'] = df['time_xa'].dt.year
df = df[df.year != 2020]

#Creating a column for load based on route leg, where 1 is for laden, 0 for ballast
load = []
for l in df['leg']:
    if l <= 1:
        load.append(0)
    else: load.append(1)
df['load'] = load

# Making a dataframe of only Route 2 and removing unessesary columns
df_pipapav = df[df['full_route'] == "rendezvous-samarinda-pipapav"].drop(columns=['full_route','vmdr','vmdr_sw1','wave_box_fr_x','time_wave_xa','time_wind_xa','wave_box_to_x','wave_box_fr_y','wave_box_to_y','lon_wp','lat_wp','heading_to'])

## Splitting route 2 into four periods   
# Creating a column for month to be able to define our periods 
df_pipapav['month'] = df_pipapav['time_xa'].dt.month
# Removing rows with starting dates from or later than 2019-11-01 in order to prevent unfinished voyages
df_pipapav = df_pipapav[(df_pipapav['starting_date'] <= '2019-11-01')]

#Creating a column called season in order to get the mode of periods based on what day the voyage start
season=[]
for x in df_pipapav["month"]:
    if x < 1.5:
        season.append(1)
    elif x >= 1.5 and x < 2.5:
        season.append(1)
    elif x >= 2.5 and x < 3.5:
        season.append(2)
    elif x >= 3.5 and x < 4.5:
        season.append(2)
    elif x >= 4.5 and x < 5.5:
        season.append(2)
    elif x >=5.5 and x < 6.5:
        season.append(3)
    elif x >=6.5 and x < 7.5:
        season.append(3)
    elif x >=7.5 and x < 8.5:
        season.append(3)
    elif x >=8.5 and x < 9.5:
        season.append(4)
    elif x >=9.5 and x < 10.5:
        season.append(4)
    elif x >=10.5 and x < 11.5:
       season.append(4)
    else:
        season.append(1)
df_pipapav['season'] = season

# Fill in NAN with mean of the month observations for wind speed in knots, and the mode for apparent wind speed
df_pipapav['wind_speed_kts'] = df_pipapav.groupby(['month','year'])['wind_speed_kts'].transform(lambda x: x.fillna(x.mean()))
df_pipapav['app_wind_dir'] = df_pipapav.groupby(['month','year'])['app_wind_dir'].transform(lambda x: x.fillna(x.value_counts().index[0]))

# Creating a new column in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df_pipapav["voyage_season"] = (df_pipapav.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Making a new dataframe of route 2 of months December, January & February
pipapav_DJF = df_pipapav[df_pipapav['voyage_season'] == 1]

# Making a new dataframe of route 2 of months March, April & May
pipapav_MAM = df_pipapav[df_pipapav['voyage_season'] == 2]

# Making a new dataframe of route 2 of months June, July & August
pipapav_JJA = df_pipapav[df_pipapav['voyage_season'] == 3]

# Making a new dataframe of route 2 of months September, October & November
pipapav_SON = df_pipapav[df_pipapav['voyage_season'] == 4]

####################################################
#### Getting a new dataset to merge later with the fuel consumption column for further work
pipapav_DJF = pipapav_DJF.reset_index(drop=True)
pipapav_DJF_1 = pipapav_DJF.groupby(['starting_date',(pipapav_DJF['time_xa'].dt.year),(pipapav_DJF['time_xa'].dt.month),(pipapav_DJF['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

pipapav_MAM = pipapav_MAM.reset_index(drop=True)
pipapav_MAM_1 = pipapav_MAM.groupby(['starting_date',(pipapav_MAM['time_xa'].dt.year),(pipapav_MAM['time_xa'].dt.month),(pipapav_MAM['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

pipapav_JJA = pipapav_JJA.reset_index(drop=True)
pipapav_JJA_1 = pipapav_JJA.groupby(['starting_date',(pipapav_JJA['time_xa'].dt.year),(pipapav_JJA['time_xa'].dt.month),(pipapav_JJA['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

pipapav_SON = pipapav_SON.reset_index(drop=True)
pipapav_SON_1 = pipapav_SON.groupby(['starting_date',(pipapav_SON['time_xa'].dt.year),(pipapav_SON['time_xa'].dt.month),(pipapav_SON['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

####################################################
## Aggregating the dataframes for all periods in order to get one observation per day, equal to the training set used for creating the machine learning model 
pipapav_DJF = pipapav_DJF.reset_index(drop=True)
agg_pipapav_DJF = pipapav_DJF.groupby(['starting_date',(pipapav_DJF['time_xa'].dt.year),(pipapav_DJF['time_xa'].dt.month),(pipapav_DJF['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

pipapav_MAM = pipapav_MAM.reset_index(drop=True)
agg_pipapav_MAM= pipapav_MAM.groupby(['starting_date',(pipapav_MAM['time_xa'].dt.year),(pipapav_MAM['time_xa'].dt.month),(pipapav_MAM['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

pipapav_JJA = pipapav_JJA.reset_index(drop=True)
agg_pipapav_JJA = pipapav_JJA.groupby(['starting_date',(pipapav_JJA['time_xa'].dt.year),(pipapav_JJA['time_xa'].dt.month),(pipapav_JJA['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

pipapav_SON = pipapav_SON.reset_index(drop=True)
agg_pipapav_SON = pipapav_SON.groupby(['starting_date',(pipapav_SON['time_xa'].dt.year),(pipapav_SON['time_xa'].dt.month),(pipapav_SON['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

####################################################
## Deleting unessesary columns
agg_pipapav_DJF=agg_pipapav_DJF.drop(columns=['year','month','starting_date']).reset_index(drop=True)
agg_pipapav_MAM=agg_pipapav_MAM.drop(columns=['year','month','starting_date']).reset_index(drop=True)
agg_pipapav_JJA=agg_pipapav_JJA.drop(columns=['year','month','starting_date']).reset_index(drop=True)
agg_pipapav_SON=agg_pipapav_SON.drop(columns=['year','month','starting_date']).reset_index(drop=True)

## Making dataframe with identical columns to the dataset used to train the machine learning model
# Making a column for the design (we are only operating with Dolphine 64 which is listed as 1 in the machine learning dataframe)
agg_pipapav_DJF["design"] = 1
agg_pipapav_MAM["design"] = 1
agg_pipapav_JJA["design"] = 1
agg_pipapav_SON["design"] = 1

# Making a column for speed over ground, whuch is a fixed value at 12.5, which is the speed of the vessel
agg_pipapav_DJF["speed_over_ground"] = 12.5
agg_pipapav_MAM["speed_over_ground"] = 12.5
agg_pipapav_JJA["speed_over_ground"] = 12.5
agg_pipapav_SON["speed_over_ground"] = 12.5

#########################################
### Fuel consumption for Route 2, DJF
# Making dummy variables of the apparent swell, wind and waves columns
pipapav_DJF_fuel=agg_pipapav_DJF.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(pipapav_DJF_fuel.loc[:,columns_to_one_hot])
pipapav_DJF_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
pipapav_DJF_fuel = pd.concat([pipapav_DJF_fuel,pipapav_DJF_fuel_encoded],axis=1)
pipapav_DJF_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = pipapav_DJF_fuel['vhm0'].mean() + 3*pipapav_DJF_fuel['vhm0'].std()
lower_limit_vh = pipapav_DJF_fuel['vhm0'].mean() - 3*pipapav_DJF_fuel['vhm0'].std()
pipapav_DJF_fuel['vhm0'] = np.where(
    pipapav_DJF_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        pipapav_DJF_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        pipapav_DJF_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = pipapav_DJF_fuel['vhm0_sw1'].mean() + 3*pipapav_DJF_fuel['vhm0_sw1'].std()
lower_limit_sw = pipapav_DJF_fuel['vhm0_sw1'].mean() - 3*pipapav_DJF_fuel['vhm0_sw1'].std()
pipapav_DJF_fuel['vhm0_sw1'] = np.where(
    pipapav_DJF_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        pipapav_DJF_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        pipapav_DJF_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = pipapav_DJF_fuel['speed_over_ground'].mean() + 3*pipapav_DJF_fuel['speed_over_ground'].std()
lower_limit_sog = pipapav_DJF_fuel['speed_over_ground'].mean() - 3*pipapav_DJF_fuel['speed_over_ground'].std()
pipapav_DJF_fuel['speed_over_ground'] = np.where(
    pipapav_DJF_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        pipapav_DJF_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        pipapav_DJF_fuel['speed_over_ground']
    )
)
upper_limit_ws = pipapav_DJF_fuel['wind_speed_kts'].mean() + 3*pipapav_DJF_fuel['wind_speed_kts'].std()
lower_limit_ws = pipapav_DJF_fuel['wind_speed_kts'].mean() - 3*pipapav_DJF_fuel['wind_speed_kts'].std()
pipapav_DJF_fuel['wind_speed_kts'] = np.where(
    pipapav_DJF_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        pipapav_DJF_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        pipapav_DJF_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_pipapav_DJF=model.predict(pipapav_DJF_fuel)
pipapav_DJF_fuel['fuel_consumption'] = predict_pipapav_DJF

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
pipapav_DJF_fuel['ID']= np.arange(len(pipapav_DJF_fuel))
pipapav_DJF_1['ID']=np.arange(len(pipapav_DJF_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
pipapav_DJF_both = pd.merge(pipapav_DJF_fuel,pipapav_DJF_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)

# Group the dataframe by starting date to find the total fuel consumption per voyage
pipapav_DJF_voy_fuel = pipapav_DJF_both.groupby([(pipapav_DJF_both['starting_date'].dt.year),(pipapav_DJF_both['starting_date'].dt.month),(pipapav_DJF_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
pipapav_DJF_voy_fuel=pipapav_DJF_voy_fuel

# Making a plot of the total fuel consumption for December, January and February
sns.distplot(pipapav_DJF_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='DJF')
plt.legend()
plt.title("FC - Route 2 - DJF")
sns.plt.show()

#########################################
### Fuel consumption for Route 2, MAM
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
pipapav_MAM_fuel=agg_pipapav_MAM.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(pipapav_MAM_fuel.loc[:,columns_to_one_hot])
pipapav_MAM_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
pipapav_MAM_fuel = pd.concat([pipapav_MAM_fuel,pipapav_MAM_fuel_encoded],axis=1)
pipapav_MAM_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = pipapav_MAM_fuel['vhm0'].mean() + 3*pipapav_MAM_fuel['vhm0'].std()
lower_limit_vh = pipapav_MAM_fuel['vhm0'].mean() - 3*pipapav_MAM_fuel['vhm0'].std()
pipapav_MAM_fuel['vhm0'] = np.where(
    pipapav_MAM_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        pipapav_MAM_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        pipapav_MAM_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = pipapav_MAM_fuel['vhm0_sw1'].mean() + 3*pipapav_MAM_fuel['vhm0_sw1'].std()
lower_limit_sw = pipapav_MAM_fuel['vhm0_sw1'].mean() - 3*pipapav_MAM_fuel['vhm0_sw1'].std()
pipapav_MAM_fuel['vhm0_sw1'] = np.where(
    pipapav_MAM_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        pipapav_MAM_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        pipapav_MAM_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = pipapav_MAM_fuel['speed_over_ground'].mean() + 3*pipapav_MAM_fuel['speed_over_ground'].std()
lower_limit_sog = pipapav_MAM_fuel['speed_over_ground'].mean() - 3*pipapav_MAM_fuel['speed_over_ground'].std()
pipapav_MAM_fuel['speed_over_ground'] = np.where(
    pipapav_MAM_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        pipapav_MAM_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        pipapav_MAM_fuel['speed_over_ground']
    )
)
upper_limit_ws = pipapav_MAM_fuel['wind_speed_kts'].mean() + 3*pipapav_MAM_fuel['wind_speed_kts'].std()
lower_limit_ws = pipapav_MAM_fuel['wind_speed_kts'].mean() - 3*pipapav_MAM_fuel['wind_speed_kts'].std()
pipapav_MAM_fuel['wind_speed_kts'] = np.where(
    pipapav_MAM_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        pipapav_MAM_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        pipapav_MAM_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_pipapav_MAM=model.predict(pipapav_MAM_fuel)
pipapav_MAM_fuel['fuel_consumption'] = predict_pipapav_MAM

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
pipapav_MAM_fuel['ID']= np.arange(len(pipapav_MAM_fuel))
pipapav_MAM_1['ID']=np.arange(len(pipapav_MAM_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
pipapav_MAM_both = pd.merge(pipapav_MAM_fuel,pipapav_MAM_1,on=['ID','vhm0','wind_speed_kts'],how="left")

# Group the dataframe by starting date to find the total fuel consumption per voyage
pipapav_MAM_voy_fuel = pipapav_MAM_both.groupby([(pipapav_MAM_both['starting_date'].dt.year),(pipapav_MAM_both['starting_date'].dt.month),(pipapav_MAM_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
pipapav_MAM_voy_fuel=pipapav_MAM_voy_fuel

# Making a plot of the total fuel consumption for March, April and May
sns.distplot(pipapav_MAM_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='MAM')
plt.legend()
plt.title("FC - Route 2 - MAM")
sns.plt.show()

#########################################
### Fuel consumption for Route 2, JJA
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
pipapav_JJA_fuel=agg_pipapav_JJA.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(pipapav_JJA_fuel.loc[:,columns_to_one_hot])
pipapav_JJA_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
pipapav_JJA_fuel = pd.concat([pipapav_JJA_fuel,pipapav_JJA_fuel_encoded],axis=1)
pipapav_JJA_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = pipapav_JJA_fuel['vhm0'].mean() + 3*pipapav_JJA_fuel['vhm0'].std()
lower_limit_vh = pipapav_JJA_fuel['vhm0'].mean() - 3*pipapav_JJA_fuel['vhm0'].std()
pipapav_JJA_fuel['vhm0'] = np.where(
    pipapav_JJA_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        pipapav_JJA_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        pipapav_JJA_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = pipapav_JJA_fuel['vhm0_sw1'].mean() + 3*pipapav_JJA_fuel['vhm0_sw1'].std()
lower_limit_sw = pipapav_JJA_fuel['vhm0_sw1'].mean() - 3*pipapav_JJA_fuel['vhm0_sw1'].std()
pipapav_JJA_fuel['vhm0_sw1'] = np.where(
    pipapav_JJA_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        pipapav_JJA_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        pipapav_JJA_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = pipapav_JJA_fuel['speed_over_ground'].mean() + 3*pipapav_JJA_fuel['speed_over_ground'].std()
lower_limit_sog = pipapav_JJA_fuel['speed_over_ground'].mean() - 3*pipapav_JJA_fuel['speed_over_ground'].std()
pipapav_JJA_fuel['speed_over_ground'] = np.where(
    pipapav_JJA_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        pipapav_JJA_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        pipapav_JJA_fuel['speed_over_ground']
    )
)
upper_limit_ws = pipapav_JJA_fuel['wind_speed_kts'].mean() + 3*pipapav_JJA_fuel['wind_speed_kts'].std()
lower_limit_ws = pipapav_JJA_fuel['wind_speed_kts'].mean() - 3*pipapav_JJA_fuel['wind_speed_kts'].std()
pipapav_JJA_fuel['wind_speed_kts'] = np.where(
    pipapav_JJA_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        pipapav_JJA_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        pipapav_JJA_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_pipapav_JJA=model.predict(pipapav_JJA_fuel)
pipapav_JJA_fuel['fuel_consumption'] = predict_pipapav_JJA

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
pipapav_JJA_fuel['ID']= np.arange(len(pipapav_JJA_fuel))
pipapav_JJA_1['ID']=np.arange(len(pipapav_JJA_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
pipapav_JJA_both = pd.merge(pipapav_JJA_fuel,pipapav_JJA_1,on=['ID','vhm0','wind_speed_kts'],how="left")

# Group the dataframe by starting date to find the total fuel consumption per voyage
pipapav_JJA_voy_fuel = pipapav_JJA_both.groupby([(pipapav_JJA_both['starting_date'].dt.year),(pipapav_JJA_both['starting_date'].dt.month),(pipapav_JJA_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
pipapav_JJA_voy_fuel=pipapav_JJA_voy_fuel

# Making a plot of the total fuel consumption for June, July and August
sns.distplot(pipapav_JJA_voy_fuel["fuel_consumption"], hist = False, kde = True, label='JJA')
plt.legend()
plt.title("FC - Route 2 - JJA")
sns.plt.show()

#########################################
### Fuel consumption for Route 2, SON
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
pipapav_SON_fuel=agg_pipapav_SON.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(pipapav_SON_fuel.loc[:,columns_to_one_hot])
pipapav_SON_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
pipapav_SON_fuel = pd.concat([pipapav_SON_fuel,pipapav_SON_fuel_encoded],axis=1)
pipapav_SON_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = pipapav_SON_fuel['vhm0'].mean() + 3*pipapav_SON_fuel['vhm0'].std()
lower_limit_vh = pipapav_SON_fuel['vhm0'].mean() - 3*pipapav_SON_fuel['vhm0'].std()
pipapav_SON_fuel['vhm0'] = np.where(
    pipapav_SON_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        pipapav_SON_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        pipapav_SON_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = pipapav_SON_fuel['vhm0_sw1'].mean() + 3*pipapav_SON_fuel['vhm0_sw1'].std()
lower_limit_sw = pipapav_SON_fuel['vhm0_sw1'].mean() - 3*pipapav_SON_fuel['vhm0_sw1'].std()
pipapav_SON_fuel['vhm0_sw1'] = np.where(
    pipapav_SON_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        pipapav_SON_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        pipapav_SON_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = pipapav_SON_fuel['speed_over_ground'].mean() + 3*pipapav_SON_fuel['speed_over_ground'].std()
lower_limit_sog = pipapav_SON_fuel['speed_over_ground'].mean() - 3*pipapav_SON_fuel['speed_over_ground'].std()
pipapav_SON_fuel['speed_over_ground'] = np.where(
    pipapav_SON_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        pipapav_SON_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        pipapav_SON_fuel['speed_over_ground']
    )
)
upper_limit_ws = pipapav_SON_fuel['wind_speed_kts'].mean() + 3*pipapav_SON_fuel['wind_speed_kts'].std()
lower_limit_ws = pipapav_SON_fuel['wind_speed_kts'].mean() - 3*pipapav_SON_fuel['wind_speed_kts'].std()
pipapav_SON_fuel['wind_speed_kts'] = np.where(
    pipapav_SON_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        pipapav_SON_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        pipapav_SON_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_pipapav_SON=model.predict(pipapav_SON_fuel)
pipapav_SON_fuel['fuel_consumption'] = predict_pipapav_SON

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
pipapav_SON_fuel['ID']= np.arange(len(pipapav_SON_fuel))
pipapav_SON_1['ID']=np.arange(len(pipapav_SON_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
pipapav_SON_both = pd.merge(pipapav_SON_fuel,pipapav_SON_1,on=['ID','vhm0','wind_speed_kts'],how="left")

# Group the dataframe by starting date to find the total fuel consumption per voyage
pipapav_SON_voy_fuel = pipapav_SON_both.groupby([(pipapav_SON_both['starting_date'].dt.year),(pipapav_SON_both['starting_date'].dt.month),(pipapav_SON_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
pipapav_SON_voy_fuel=pipapav_SON_voy_fuel

# Making a plot of the total fuel consumption for September, October and November
sns.distplot(pipapav_SON_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='SON')
plt.legend()
plt.title("FC - Route 2 - SON")
sns.plt.show()

######################################### Total Fuel Consumption in one plot #########################################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(pipapav_DJF_voy_fuel["fuel_consumption"], label='DJF', ax=ax)
sns.kdeplot(pipapav_MAM_voy_fuel["fuel_consumption"], label='MAM', ax=ax)
sns.kdeplot(pipapav_JJA_voy_fuel["fuel_consumption"], label='JJA', ax=ax)
sns.kdeplot(pipapav_SON_voy_fuel["fuel_consumption"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - Route 2", fontsize=12)
plt.savefig('fc_r2.png', dpi = 300)
sns.plt.show()

######################################### Calculating TCE for Route 2 #########################################
pipapav_DJF_case = pipapav_DJF_voy_fuel
pipapav_MAM_case = pipapav_MAM_voy_fuel
pipapav_JJA_case = pipapav_JJA_voy_fuel
pipapav_SON_case = pipapav_SON_voy_fuel

################# TCE calculation simulation, scenario 1 (low bunker price) ####################
##### DJF
# Used the random seed of 42 when simulating to get the results used for the thesis

# For this route, we used the distribution of total fuel consumption per voyage and to calculate the TCE, we used  
# np.random.choice to retrieve one random variable for the TCE equation and then repeating this
# for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

random.seed(42)
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_DJF_case['fuel_consumption'])
    time_horizon = len(pipapav_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_DJF_case["TCE_DJF_c1"] = simulation(9,379.33,69598,35.28)

##### MAM
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_MAM_case['fuel_consumption'])
    time_horizon = len(pipapav_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_MAM_case["TCE_MAM_c1"] = simulation(9,379.33,69598,35.28)

##### JJA
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_JJA_case['fuel_consumption'])
    time_horizon = len(pipapav_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_JJA_case["TCE_JJA_c1"] = simulation(9,379.33,69598,35.28)

##### SON
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_SON_case['fuel_consumption'])
    time_horizon = len(pipapav_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_SON_case["TCE_SON_c1"] = simulation(9,379.33,69598,35.28)

################# Plot of TCE distribution, scenario 1 (low bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(pipapav_DJF_case["TCE_DJF_c1"], label='DJF', ax=ax)
sns.kdeplot(pipapav_MAM_case["TCE_MAM_c1"], label='MAM', ax=ax)
sns.kdeplot(pipapav_JJA_case["TCE_JJA_c1"], label='JJA', ax=ax)
sns.kdeplot(pipapav_SON_case["TCE_SON_c1"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 2 - Scenario 1", fontsize=12)
plt.savefig('TCE_r2_c1.png', dpi = 300)
sns.plt.show()

################# TCE calculation simulation, scenario 2 (mean bunker price) ####################
##### DJF

# For this route, we used the distribution of total fuel consumption per voyage and to calculate the TCE, we used  
# np.random.choice to retrieve one random variable for the TCE equation and then repeating this
# for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_DJF_case['fuel_consumption'])
    time_horizon = len(pipapav_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_DJF_case["TCE_DJF_c2"] = simulation(9.2,479,69598,35.28)

##### MAM
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_MAM_case['fuel_consumption'])
    time_horizon = len(pipapav_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_MAM_case["TCE_MAM_c2"] = simulation(9.2,479,69598,35.28)

##### JJA
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_JJA_case['fuel_consumption'])
    time_horizon = len(pipapav_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_JJA_case["TCE_JJA_c2"] = simulation(9.2,479,69598,35.28)

##### SON
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_SON_case['fuel_consumption'])
    time_horizon = len(pipapav_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_SON_case["TCE_SON_c2"] = simulation(9.2,479,69598,35.28)

################# Plot of TCE distribution, scenario 2 (mean bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(pipapav_DJF_case["TCE_DJF_c2"], label='DJF', ax=ax)
sns.kdeplot(pipapav_MAM_case["TCE_MAM_c2"], label='MAM', ax=ax)
sns.kdeplot(pipapav_JJA_case["TCE_JJA_c2"], label='JJA', ax=ax)
sns.kdeplot(pipapav_SON_case["TCE_SON_c2"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 2 - Scenario 2", fontsize=12)
plt.savefig('TCE_r2_c2.png', dpi = 300)
sns.plt.show()

################# TCE calculation simulation, scenario 3 (high bunker price) ####################
##### DJF

# For this route, we used the distribution of total fuel consumption per voyage and to calculate the TCE, we used  
# np.random.choice to retrieve one random variable for the TCE equation and then repeating this
# for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_DJF_case['fuel_consumption'])
    time_horizon = len(pipapav_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_DJF_case["TCE_DJF_c3"] = simulation(11.4,549.21,69598,35.28)

##### MAM
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_MAM_case['fuel_consumption'])
    time_horizon = len(pipapav_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_MAM_case["TCE_MAM_c3"] = simulation(11.4,549.21,69598,35.28)

##### JJA
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_JJA_case['fuel_consumption'])
    time_horizon = len(pipapav_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_JJA_case["TCE_JJA_c3"] = simulation(11.4,549.21,69598,35.28)

##### SON
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (pipapav_SON_case['fuel_consumption'])
    time_horizon = len(pipapav_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
pipapav_SON_case["TCE_SON_c3"] = simulation(11.4,549.21,69598,35.28)

################# Plot of TCE distribution, scenario 3 (high bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(pipapav_DJF_case["TCE_DJF_c3"], label='DJF', ax=ax)
sns.kdeplot(pipapav_MAM_case["TCE_MAM_c3"], label='MAM', ax=ax)
sns.kdeplot(pipapav_JJA_case["TCE_JJA_c3"], label='JJA', ax=ax)
sns.kdeplot(pipapav_SON_case["TCE_SON_c3"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 2 - Scenario 3", fontsize=12)
plt.savefig('TCE_r2_c3.png', dpi = 300)
sns.plt.show()

# Retrieving statistics of mean, min and max fuel consumption and TCE values
route2_desc_DJF = pipapav_DJF_case.describe()
route2_desc_MAM = pipapav_MAM_case.describe()
route2_desc_JJA = pipapav_JJA_case.describe()
route2_desc_SON = pipapav_SON_case.describe()

# Using joblib to gather the dataframes in seperate scripts for comparing fuel consumption and TCE
import joblib
joblib.dump(pipapav_DJF_case,"pipapav_DJF_case.joblib")
joblib.dump(pipapav_MAM_case,"pipapav_MAM_case.joblib")
joblib.dump(pipapav_JJA_case,"pipapav_JJA_case.joblib")
joblib.dump(pipapav_SON_case,"pipapav_SON_case.joblib")




