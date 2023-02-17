# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:11:36 2022

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

# Making a dataframe of only Route 3 and removing unessesary columns
df_mundra = df[df['full_route'] == "rendezvous-richards_bay-mundra"].drop(columns=["full_route",'vmdr','vmdr_sw1','wave_box_fr_x','time_wave_xa','time_wind_xa','wave_box_to_x','wave_box_fr_y','wave_box_to_y','lon_wp','lat_wp','heading_to'])

## Splitting route 2 into four periods   
# Creating a column for month to be able to define our periods 
df_mundra['month'] = df_mundra['time_xa'].dt.month
# Removing rows with starting dates from or later than 2019-11-01 in order to prevent unfinished voyages
df_mundra = df_mundra[(df_mundra['starting_date'] <= '2019-11-01')]

#Creating a column called season in order to get the mode of periods based on what day the voyage start
season=[]
for seas in df_mundra["month"]:
    if seas < 1.5:
        season.append(1)
    elif seas >= 1.5 and seas < 2.5:
        season.append(1)
    elif seas >= 2.5 and seas < 3.5:
        season.append(2)
    elif seas >= 3.5 and seas < 4.5:
        season.append(2)
    elif seas >= 4.5 and seas < 5.5:
         season.append(2)
    elif seas >=5.5 and seas < 6.5:
        season.append(3)
    elif seas >=6.5 and seas < 7.5:
        season.append(3)
    elif seas >=7.5 and seas < 8.5:
        season.append(3)
    elif seas >=8.5 and seas < 9.5:
        season.append(4)
    elif seas >=9.5 and seas < 10.5:
        season.append(4)
    elif seas >=10.5 and seas < 11.5:
        season.append(4)
    else:
        season.append(1)
df_mundra['season'] = season

# Fill in NAN with mean of the month observations for wind speed in knots, and the mode for apparent wind speed
df_mundra['wind_speed_kts'] = df_mundra.groupby(['month','year'])['wind_speed_kts'].transform(lambda x: x.fillna(x.mean()))
df_mundra['app_wind_dir'] = df_mundra.groupby(['month','year'])['app_wind_dir'].transform(lambda x: x.fillna(x.value_counts().index[0]))

# Creating a new column in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df_mundra["voyage_season"] = (df_mundra.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Making a new dataframe of route 3 of months December, January & February
mundra_DJF = df_mundra[df_mundra['voyage_season'] == 1]

# Making a new dataframe of route 3 of months March, April & May
mundra_MAM = df_mundra[df_mundra['voyage_season'] == 2]

# Making a new dataframe of route 3 of months June, July & August
mundra_JJA = df_mundra[df_mundra['voyage_season'] == 3]

# Making a new dataframe of route 3 of months September, October & November
mundra_SON = df_mundra[df_mundra['voyage_season'] == 4]

####################################################
#### Getting a new dataset to merge later with the fuel consumption column for further work
mundra_DJF = mundra_DJF.reset_index(drop=True)
mundra_DJF_1 = mundra_DJF.groupby(['starting_date',(mundra_DJF['time_xa'].dt.year),(mundra_DJF['time_xa'].dt.month),(mundra_DJF['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

mundra_MAM = mundra_MAM.reset_index(drop=True)
mundra_MAM_1 = mundra_MAM.groupby(['starting_date',(mundra_MAM['time_xa'].dt.year),(mundra_MAM['time_xa'].dt.month),(mundra_MAM['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

mundra_JJA = mundra_JJA.reset_index(drop=True)
mundra_JJA_1 = mundra_JJA.groupby(['starting_date',(mundra_JJA['time_xa'].dt.year),(mundra_JJA['time_xa'].dt.month),(mundra_JJA['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

mundra_SON = mundra_SON.reset_index(drop=True)
mundra_SON_1 = mundra_SON.groupby(['starting_date',(mundra_SON['time_xa'].dt.year),(mundra_SON['time_xa'].dt.month),(mundra_SON['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

####################################################
## Aggregating the dataframes for all periods in order to get one observation per day, equal to the training set used for creating the machine learning model 
mundra_DJF = mundra_DJF.reset_index(drop=True)
agg_mundra_DJF = mundra_DJF.groupby(['starting_date',(mundra_DJF['time_xa'].dt.year),(mundra_DJF['time_xa'].dt.month),(mundra_DJF['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

mundra_MAM = mundra_MAM.reset_index(drop=True)
agg_mundra_MAM = mundra_MAM.groupby(['starting_date',(mundra_MAM['time_xa'].dt.year),(mundra_MAM['time_xa'].dt.month),(mundra_MAM['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

mundra_JJA = mundra_JJA.reset_index(drop=True)
agg_mundra_JJA = mundra_JJA.groupby(['starting_date',(mundra_JJA['time_xa'].dt.year),(mundra_JJA['time_xa'].dt.month),(mundra_JJA['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

mundra_SON = mundra_SON.reset_index(drop=True)
agg_mundra_SON = mundra_SON.groupby(['starting_date',(mundra_SON['time_xa'].dt.year),(mundra_SON['time_xa'].dt.month),(mundra_SON['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

####################################################
## Deleting unessesary columns
agg_mundra_DJF=agg_mundra_DJF.drop(columns=['month','year','starting_date']).reset_index(drop=True)
agg_mundra_MAM=agg_mundra_MAM.drop(columns=['month','year','starting_date']).reset_index(drop=True)
agg_mundra_JJA=agg_mundra_JJA.drop(columns=['month','year','starting_date']).reset_index(drop=True)
agg_mundra_SON=agg_mundra_SON.drop(columns=['month','year','starting_date']).reset_index(drop=True)

## Making dataframe with identical columns to the dataset used to train the machine learning model
# Making a column for the design (we are only operating with Dolphine 64 which is listed as 1 in the machine learning dataframe)
agg_mundra_DJF["design"] = 1
agg_mundra_MAM["design"] = 1
agg_mundra_JJA["design"] = 1
agg_mundra_SON["design"] = 1

# Making a column for speed over ground, whuch is a fixed value at 12.5, which is the speed of the vessel
agg_mundra_DJF["speed_over_ground"] = 12.5
agg_mundra_MAM["speed_over_ground"] = 12.5
agg_mundra_JJA["speed_over_ground"] = 12.5
agg_mundra_SON["speed_over_ground"] = 12.5

#########################################
### Fuel consumption for Route 3, DJF
# Making dummy variables of the apparent swell, wind and waves columns
mundra_DJF_fuel=agg_mundra_DJF.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(mundra_DJF_fuel.loc[:,columns_to_one_hot])
mundra_DJF_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
mundra_DJF_fuel = pd.concat([mundra_DJF_fuel,mundra_DJF_fuel_encoded],axis=1)
mundra_DJF_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = mundra_DJF_fuel['vhm0'].mean() + 3*mundra_DJF_fuel['vhm0'].std()
lower_limit_vh = mundra_DJF_fuel['vhm0'].mean() - 3*mundra_DJF_fuel['vhm0'].std()
mundra_DJF_fuel['vhm0'] = np.where(
    mundra_DJF_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        mundra_DJF_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        mundra_DJF_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = mundra_DJF_fuel['vhm0_sw1'].mean() + 3*mundra_DJF_fuel['vhm0_sw1'].std()
lower_limit_sw = mundra_DJF_fuel['vhm0_sw1'].mean() - 3*mundra_DJF_fuel['vhm0_sw1'].std()
mundra_DJF_fuel['vhm0_sw1'] = np.where(
    mundra_DJF_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        mundra_DJF_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        mundra_DJF_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = mundra_DJF_fuel['speed_over_ground'].mean() + 3*mundra_DJF_fuel['speed_over_ground'].std()
lower_limit_sog = mundra_DJF_fuel['speed_over_ground'].mean() - 3*mundra_DJF_fuel['speed_over_ground'].std()
mundra_DJF_fuel['speed_over_ground'] = np.where(
    mundra_DJF_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        mundra_DJF_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        mundra_DJF_fuel['speed_over_ground']
    )
)
upper_limit_ws = mundra_DJF_fuel['wind_speed_kts'].mean() + 3*mundra_DJF_fuel['wind_speed_kts'].std()
lower_limit_ws = mundra_DJF_fuel['wind_speed_kts'].mean() - 3*mundra_DJF_fuel['wind_speed_kts'].std()
mundra_DJF_fuel['wind_speed_kts'] = np.where(
    mundra_DJF_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        mundra_DJF_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        mundra_DJF_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_mundra_DJF=model.predict(mundra_DJF_fuel)
mundra_DJF_fuel['fuel_consumption'] = predict_mundra_DJF

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
mundra_DJF_fuel['ID']= np.arange(len(mundra_DJF_fuel))
mundra_DJF_1['ID']=np.arange(len(mundra_DJF_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
mundra_DJF_both = pd.merge(mundra_DJF_fuel,mundra_DJF_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)

# Group the dataframe by starting date to find the total fuel consumption per voyage
mundra_DJF_voy_fuel = mundra_DJF_both.groupby([(mundra_DJF_both['starting_date'].dt.year),(mundra_DJF_both['starting_date'].dt.month),(mundra_DJF_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
mundra_DJF_voy_fuel=mundra_DJF_voy_fuel

# Making a plot of the total fuel consumption for December, January and February
sns.distplot(mundra_DJF_voy_fuel["fuel_consumption"], hist = False, kde = True, label='DJF')
plt.legend()
plt.title("FC - Route 3 - DJF")
sns.plt.show()

#########################################
### Fuel consumption for Route 3, MAM
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
mundra_MAM_fuel=agg_mundra_MAM.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(mundra_MAM_fuel.loc[:,columns_to_one_hot])
mundra_MAM_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
mundra_MAM_fuel = pd.concat([mundra_MAM_fuel,mundra_MAM_fuel_encoded],axis=1)
mundra_MAM_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = mundra_MAM_fuel['vhm0'].mean() + 3*mundra_MAM_fuel['vhm0'].std()
lower_limit_vh = mundra_MAM_fuel['vhm0'].mean() - 3*mundra_MAM_fuel['vhm0'].std()
mundra_MAM_fuel['vhm0'] = np.where(
    mundra_MAM_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        mundra_MAM_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        mundra_MAM_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = mundra_MAM_fuel['vhm0_sw1'].mean() + 3*mundra_MAM_fuel['vhm0_sw1'].std()
lower_limit_sw = mundra_MAM_fuel['vhm0_sw1'].mean() - 3*mundra_MAM_fuel['vhm0_sw1'].std()
mundra_MAM_fuel['vhm0_sw1'] = np.where(
    mundra_MAM_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        mundra_MAM_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        mundra_MAM_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = mundra_MAM_fuel['speed_over_ground'].mean() + 3*mundra_MAM_fuel['speed_over_ground'].std()
lower_limit_sog = mundra_MAM_fuel['speed_over_ground'].mean() - 3*mundra_MAM_fuel['speed_over_ground'].std()
mundra_MAM_fuel['speed_over_ground'] = np.where(
    mundra_MAM_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        mundra_MAM_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        mundra_MAM_fuel['speed_over_ground']
    )
)
upper_limit_ws = mundra_MAM_fuel['wind_speed_kts'].mean() + 3*mundra_MAM_fuel['wind_speed_kts'].std()
lower_limit_ws = mundra_MAM_fuel['wind_speed_kts'].mean() - 3*mundra_MAM_fuel['wind_speed_kts'].std()
mundra_MAM_fuel['wind_speed_kts'] = np.where(
    mundra_MAM_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        mundra_MAM_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        mundra_MAM_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_mundra_MAM=model.predict(mundra_MAM_fuel)
mundra_MAM_fuel['fuel_consumption'] = predict_mundra_MAM

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
mundra_MAM_fuel['ID']= np.arange(len(mundra_MAM_fuel))
mundra_MAM_1['ID']=np.arange(len(mundra_MAM_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
mundra_MAM_both = pd.merge(mundra_MAM_fuel,mundra_MAM_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)

# Group the dataframe by starting date to find the total fuel consumption per voyage
mundra_MAM_voy_fuel = mundra_MAM_both.groupby([(mundra_MAM_both['starting_date'].dt.year),(mundra_MAM_both['starting_date'].dt.month),(mundra_MAM_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
mundra_MAM_voy_fuel=mundra_MAM_voy_fuel

# Making a plot of the total fuel consumption for March, April and May
sns.distplot(mundra_MAM_voy_fuel["fuel_consumption"], hist = False, kde = True, label='MAM')
plt.legend()
plt.title("FC - Route 3 - MAM")
sns.plt.show()

#########################################
### Fuel consumption for Route 3, JJA
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
mundra_JJA_fuel=agg_mundra_JJA.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(mundra_JJA_fuel.loc[:,columns_to_one_hot])
mundra_JJA_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
mundra_JJA_fuel = pd.concat([mundra_JJA_fuel,mundra_JJA_fuel_encoded],axis=1)
mundra_JJA_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = mundra_JJA_fuel['vhm0'].mean() + 3*mundra_JJA_fuel['vhm0'].std()
lower_limit_vh = mundra_JJA_fuel['vhm0'].mean() - 3*mundra_JJA_fuel['vhm0'].std()
mundra_JJA_fuel['vhm0'] = np.where(
    mundra_JJA_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        mundra_JJA_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        mundra_JJA_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = mundra_JJA_fuel['vhm0_sw1'].mean() + 3*mundra_JJA_fuel['vhm0_sw1'].std()
lower_limit_sw = mundra_JJA_fuel['vhm0_sw1'].mean() - 3*mundra_JJA_fuel['vhm0_sw1'].std()
mundra_JJA_fuel['vhm0_sw1'] = np.where(
    mundra_JJA_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        mundra_JJA_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        mundra_JJA_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = mundra_JJA_fuel['speed_over_ground'].mean() + 3*mundra_JJA_fuel['speed_over_ground'].std()
lower_limit_sog = mundra_JJA_fuel['speed_over_ground'].mean() - 3*mundra_JJA_fuel['speed_over_ground'].std()
mundra_JJA_fuel['speed_over_ground'] = np.where(
    mundra_JJA_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        mundra_JJA_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        mundra_JJA_fuel['speed_over_ground']
    )
)
upper_limit_ws = mundra_JJA_fuel['wind_speed_kts'].mean() + 3*mundra_JJA_fuel['wind_speed_kts'].std()
lower_limit_ws = mundra_JJA_fuel['wind_speed_kts'].mean() - 3*mundra_JJA_fuel['wind_speed_kts'].std()
mundra_JJA_fuel['wind_speed_kts'] = np.where(
    mundra_JJA_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        mundra_JJA_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        mundra_JJA_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_mundra_JJA=model.predict(mundra_JJA_fuel)
mundra_JJA_fuel['fuel_consumption'] = predict_mundra_JJA

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
mundra_JJA_fuel['ID']= np.arange(len(mundra_JJA_fuel))
mundra_JJA_1['ID']=np.arange(len(mundra_JJA_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
mundra_JJA_both = pd.merge(mundra_JJA_fuel,mundra_JJA_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)

# Group the dataframe by starting date to find the total fuel consumption per voyage
mundra_JJA_voy_fuel = mundra_JJA_both.groupby([(mundra_JJA_both['starting_date'].dt.year),(mundra_JJA_both['starting_date'].dt.month),(mundra_JJA_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
mundra_JJA_voy_fuel=mundra_JJA_voy_fuel

# Making a plot of the total fuel consumption for June, July and August
sns.distplot(mundra_JJA_voy_fuel["fuel_consumption"], hist = False, kde = True, label='JJA')
plt.legend()
plt.title("FC - Route 3 - JJA")
sns.plt.show()

#########################################
### Fuel consumption for Route 3, SON
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
mundra_SON_fuel=agg_mundra_SON.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(mundra_SON_fuel.loc[:,columns_to_one_hot])
mundra_SON_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
mundra_SON_fuel = pd.concat([mundra_SON_fuel,mundra_SON_fuel_encoded],axis=1)
mundra_SON_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)

#Getting rid of outliers from wave height
upper_limit_vh = mundra_SON_fuel['vhm0'].mean() + 3*mundra_SON_fuel['vhm0'].std()
lower_limit_vh = mundra_SON_fuel['vhm0'].mean() - 3*mundra_SON_fuel['vhm0'].std()
mundra_SON_fuel['vhm0'] = np.where(
    mundra_SON_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        mundra_SON_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        mundra_SON_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = mundra_SON_fuel['vhm0_sw1'].mean() + 3*mundra_SON_fuel['vhm0_sw1'].std()
lower_limit_sw = mundra_SON_fuel['vhm0_sw1'].mean() - 3*mundra_SON_fuel['vhm0_sw1'].std()
mundra_SON_fuel['vhm0_sw1'] = np.where(
    mundra_SON_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        mundra_SON_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        mundra_SON_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = mundra_SON_fuel['speed_over_ground'].mean() + 3*mundra_SON_fuel['speed_over_ground'].std()
lower_limit_sog = mundra_SON_fuel['speed_over_ground'].mean() - 3*mundra_SON_fuel['speed_over_ground'].std()
mundra_SON_fuel['speed_over_ground'] = np.where(
    mundra_SON_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        mundra_SON_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        mundra_SON_fuel['speed_over_ground']
    )
)
upper_limit_ws = mundra_SON_fuel['wind_speed_kts'].mean() + 3*mundra_SON_fuel['wind_speed_kts'].std()
lower_limit_ws = mundra_SON_fuel['wind_speed_kts'].mean() - 3*mundra_SON_fuel['wind_speed_kts'].std()
mundra_SON_fuel['wind_speed_kts'] = np.where(
    mundra_SON_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        mundra_SON_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        mundra_SON_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_mundra_SON=model.predict(mundra_SON_fuel)
mundra_SON_fuel['fuel_consumption'] = predict_mundra_SON

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
mundra_SON_fuel['ID']= np.arange(len(mundra_SON_fuel))
mundra_SON_1['ID']=np.arange(len(mundra_SON_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
mundra_SON_both = pd.merge(mundra_SON_fuel,mundra_SON_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)

# Group the dataframe by starting date to find the total fuel consumption per voyage
mundra_SON_voy_fuel = mundra_SON_both.groupby([(mundra_SON_both['starting_date'].dt.year),(mundra_SON_both['starting_date'].dt.month),(mundra_SON_both['starting_date'].dt.day)],as_index=False)['fuel_consumption'].aggregate({'starting_date':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
mundra_SON_voy_fuel=mundra_SON_voy_fuel

# Making a plot of the total fuel consumption for September, October and November
sns.distplot(mundra_SON_voy_fuel["fuel_consumption"], hist = False, kde = True, label='SON')
plt.legend()
plt.title("FC - Route 3 - SON")
sns.plt.show()

######################################### Total Fuel Consumption in one plot #########################################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(mundra_DJF_voy_fuel["fuel_consumption"], label='DJF', ax=ax)
sns.kdeplot(mundra_MAM_voy_fuel["fuel_consumption"], label='MAM', ax=ax)
sns.kdeplot(mundra_JJA_voy_fuel["fuel_consumption"], label='JJA', ax=ax)
sns.kdeplot(mundra_SON_voy_fuel["fuel_consumption"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - Route 3", fontsize=12)
plt.savefig('fc_r3.png', dpi = 300)
sns.plt.show()

######################################### Calculating TCE for Route 3 #########################################
mundra_DJF_case = mundra_DJF_voy_fuel
mundra_MAM_case = mundra_MAM_voy_fuel
mundra_JJA_case = mundra_JJA_voy_fuel
mundra_SON_case = mundra_SON_voy_fuel

################# TCE calculation simulation, scenario 1 (low bunker price) ####################
##### DJF
# Used the random seed of 42 when simulating to get the results used for the thesis

# For this route, we used the distribution of total fuel consumption per voyage and to calculate the TCE, we used  
# np.random.choice to retrieve one random variable for the TCE equation and then repeating this
# for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

random.seed(42)
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = mundra_DJF_case['fuel_consumption']
    time_horizon = len(mundra_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_DJF_case["TCE_DJF_c1"] = simulation(12,379.33,90166,19.53)

##### MAM
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_MAM_case['fuel_consumption'])
    time_horizon = len(mundra_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_MAM_case["TCE_MAM_c1"] = simulation(12,379.33,90166,19.53)

##### JJA
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_JJA_case['fuel_consumption'])
    time_horizon = len(mundra_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_JJA_case["TCE_JJA_c1"] = simulation(12,379.33,90166,19.53)

##### SON
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_SON_case['fuel_consumption'])
    time_horizon = len(mundra_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=(((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration)
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_SON_case["TCE_SON_c1"] = simulation(12,379.33,90166,19.53)

################# Plot of TCE distribution, scenario 1 (low bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(mundra_DJF_case["TCE_DJF_c1"], label='DJF', ax=ax)
sns.kdeplot(mundra_MAM_case["TCE_MAM_c1"], label='MAM', ax=ax)
sns.kdeplot(mundra_JJA_case["TCE_JJA_c1"], label='JJA', ax=ax)
sns.kdeplot(mundra_SON_case["TCE_SON_c1"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 3 - Scenario 1", fontsize=12)
plt.savefig('TCE_r3_c1_50.png', dpi = 300)
sns.plt.show()

################# TCE calculation simulation, scenario 2 (mean bunker price) ####################
##### DJF

# For this route, we used the distribution of total fuel consumption per voyage and to calculate the TCE, we used  
# np.random.choice to retrieve one random variable for the TCE equation and then repeating this
# for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_DJF_case['fuel_consumption'])
    time_horizon = len(mundra_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_DJF_case["TCE_DJF_c2"] = simulation(15,479,90166,19.53)

##### MAM
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_MAM_case['fuel_consumption'])
    time_horizon = len(mundra_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_MAM_case["TCE_MAM_c2"] = simulation(15,479,90166,19.53)

##### JJA
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_JJA_case['fuel_consumption'])
    time_horizon = len(mundra_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_JJA_case["TCE_JJA_c2"] = simulation(15,479,90166,19.53)

##### SON
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_SON_case['fuel_consumption'])
    time_horizon = len(mundra_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_SON_case["TCE_SON_c2"] = simulation(15,479,90166,19.53)

################# Plot of TCE distribution, scenario 2 (mean bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(mundra_DJF_case["TCE_DJF_c2"], label='DJF', ax=ax)
sns.kdeplot(mundra_MAM_case["TCE_MAM_c2"], label='MAM', ax=ax)
sns.kdeplot(mundra_JJA_case["TCE_JJA_c2"], label='JJA', ax=ax)
sns.kdeplot(mundra_SON_case["TCE_SON_c2"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 3 - Scenario 2", fontsize=12)
plt.savefig('TCE_r3_c2_50.png', dpi = 300)
sns.plt.show()

################# TCE calculation simulation, scenario 3 (high bunker price) ####################
##### DJF

# For this route, we used the distribution of total fuel consumption per voyage and to calculate the TCE, we used  
# np.random.choice to retrieve one random variable for the TCE equation and then repeating this
# for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_DJF_case['fuel_consumption'])
    time_horizon = len(mundra_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_DJF_case["TCE_DJF_c3"] = simulation(15.25,549.21,90166,19.53)

##### MAM
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_MAM_case['fuel_consumption'])
    time_horizon = len(mundra_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_MAM_case["TCE_MAM_c3"] = simulation(15.25,549.21,90166,19.53)

##### JJA
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_JJA_case['fuel_consumption'])
    time_horizon = len(mundra_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_JJA_case["TCE_JJA_c3"] = simulation(15.25,549.21,90166,19.53)

##### SON
def simulation(freightrate,bunkercost,othercosts,duration):
    res1 = pd.DataFrame()
    f_c = (mundra_SON_case['fuel_consumption'])
    time_horizon = len(mundra_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*60000)-((np.random.choice(f_c))*bunkercost)-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
mundra_SON_case["TCE_SON_c3"] = simulation(15.25,549.21,90166,19.53)

################# Plot of TCE distribution, scenario 3 (high bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(mundra_DJF_case["TCE_DJF_c3"], label='DJF', ax=ax)
sns.kdeplot(mundra_MAM_case["TCE_MAM_c3"], label='MAM', ax=ax)
sns.kdeplot(mundra_JJA_case["TCE_JJA_c3"], label='JJA', ax=ax)
sns.kdeplot(mundra_SON_case["TCE_SON_c3"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 3 - Scenario 3", fontsize=12)
plt.savefig('TCE_r3_c3_50.png', dpi = 300)
sns.plt.show()

# Retrieving statistics of mean, min and max fuel consumption and TCE values
route3_desc_DJF_50 = mundra_DJF_case.describe()
route3_desc_MAM_50 = mundra_MAM_case.describe()
route3_desc_JJA_50 = mundra_JJA_case.describe()
route3_desc_SON_50 = mundra_SON_case.describe()

# Using joblib to gather the dataframes in seperate scripts for comparing fuel consumption and TCE
import joblib
joblib.dump(mundra_DJF_case,"mundra_DJF_case.joblib")
joblib.dump(mundra_MAM_case,"mundra_MAM_case.joblib")
joblib.dump(mundra_JJA_case,"mundra_JJA_case.joblib")
joblib.dump(mundra_SON_case,"mundra_SON_case.joblib")

