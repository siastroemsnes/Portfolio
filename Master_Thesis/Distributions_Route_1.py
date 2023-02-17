# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:20:52 2022

@author: Sia Stroemsnes and Marie Log Staveland
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

# Making a dataframe of only Route 1 and removing unessesary columns
df_rotterdam= df[df['full_route'] == "rendezvous-houston-rotterdam"].drop(columns=["full_route",'vmdr','vmdr_sw1','wave_box_fr_x','time_wave_xa','time_wind_xa','wave_box_to_x','wave_box_fr_y','wave_box_to_y','lon_wp','lat_wp','heading_to'])

## Splitting route 1 into four periods   
# Creating a column for month to be able to define our periods
df_rotterdam['month'] = df_rotterdam['time_xa'].dt.month
# Removing rows with starting dates from or later than 2019-11-01 in order to prevent unfinished voyages
df_rotterdam = df_rotterdam[(df_rotterdam['starting_date'] <= '2019-11-01')]

#Creating a column called season in order to get the mode of periods based on what day the voyage start
season=[]
for x in df_rotterdam["month"]:
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
df_rotterdam['season'] = season

# Fill in NAN with mean of the month observations for wind speed in knots, and the mode for apparent wind speed
df_rotterdam['wind_speed_kts'] = df_rotterdam.groupby(['month','year'])['wind_speed_kts'].transform(lambda x: x.fillna(x.mean()))
df_rotterdam['app_wind_dir'] = df_rotterdam.groupby(['month','year'])['app_wind_dir'].transform(lambda x: x.fillna(x.value_counts().index[0]))

# Creating a new column in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df_rotterdam["voyage_season"] = (df_rotterdam.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Making a new dataframe of route 1 of months December, January & February
rotterdam_DJF = df_rotterdam[df_rotterdam['voyage_season'] == 1]

# Making a new dataframe of route 1 of months March, April & May
rotterdam_MAM = df_rotterdam[df_rotterdam['voyage_season'] == 2]

# Making a new dataframe of route 1 of months June, July & August
rotterdam_JJA = df_rotterdam[df_rotterdam['voyage_season'] == 3]

# Making a new dataframe of route 1 of months September, October & November
rotterdam_SON = df_rotterdam[df_rotterdam['voyage_season'] == 4]

####################################################
#### Getting a new dataset to merge later with the fuel consumption column for further work
rotterdam_DJF = rotterdam_DJF.reset_index(drop=True)
rotterdam_DJF_1 = rotterdam_DJF.groupby(['starting_date',(rotterdam_DJF['time_xa'].dt.year),(rotterdam_DJF['time_xa'].dt.month),(rotterdam_DJF['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

rotterdam_MAM = rotterdam_MAM.reset_index(drop=True)
rotterdam_MAM_1 = rotterdam_MAM.groupby(['starting_date',(rotterdam_MAM['time_xa'].dt.year),(rotterdam_MAM['time_xa'].dt.month),(rotterdam_MAM['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

rotterdam_JJA = rotterdam_JJA.reset_index(drop=True)
rotterdam_JJA_1 = rotterdam_JJA.groupby(['starting_date',(rotterdam_JJA['time_xa'].dt.year),(rotterdam_JJA['time_xa'].dt.month),(rotterdam_JJA['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

rotterdam_SON = rotterdam_SON.reset_index(drop=True)
rotterdam_SON_1 = rotterdam_SON.groupby(['starting_date',(rotterdam_SON['time_xa'].dt.year),(rotterdam_SON['time_xa'].dt.month),(rotterdam_SON['time_xa'].dt.day)]).aggregate({'vhm0':'mean','wind_speed_kts':'mean','eca':lambda x: x.value_counts().index[0],'leg':lambda x: x.value_counts().index[0],'starting_date':lambda x: x.value_counts().index[0],'time_xa':lambda x: x.value_counts().index[0]})

####################################################
## Aggregating the dataframes for all periods in order to get one observation per day, equal to the training set used for creating the machine learning model 
rotterdam_DJF = rotterdam_DJF.reset_index(drop=True)
agg_rotterdam_DJF = rotterdam_DJF.groupby(['starting_date',(rotterdam_DJF['time_xa'].dt.year),(rotterdam_DJF['time_xa'].dt.month),(rotterdam_DJF['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

rotterdam_MAM = rotterdam_MAM.reset_index(drop=True)
agg_rotterdam_MAM = rotterdam_MAM.groupby(['starting_date',(rotterdam_MAM['time_xa'].dt.year),(rotterdam_MAM['time_xa'].dt.month),(rotterdam_MAM['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

rotterdam_JJA = rotterdam_JJA.reset_index(drop=True)
agg_rotterdam_JJA = rotterdam_JJA.groupby(['starting_date',(rotterdam_JJA['time_xa'].dt.year),(rotterdam_JJA['time_xa'].dt.month),(rotterdam_JJA['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

rotterdam_SON = rotterdam_SON.reset_index(drop=True)
agg_rotterdam_SON = rotterdam_SON.groupby(['starting_date',(rotterdam_SON['time_xa'].dt.year),(rotterdam_SON['time_xa'].dt.month),(rotterdam_SON['time_xa'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'app_wind_dir':lambda x: x.value_counts().index[0],'load':lambda x: x.value_counts().index[0],'month':lambda x: x.value_counts().index[0],'year':lambda x: x.value_counts().index[0],'app_wave_dir':lambda x: x.value_counts().index[0],'app_swell_dir':lambda x: x.value_counts().index[0],'vhm0':'mean','vhm0_sw1':'mean','wind_speed_kts':'mean'})

####################################################
## Deleting unessesary columns
agg_rotterdam_DJF=agg_rotterdam_DJF.drop(columns=['month','year','starting_date']).reset_index(drop=True)
agg_rotterdam_MAM=agg_rotterdam_MAM.drop(columns=['month','year','starting_date']).reset_index(drop=True)
agg_rotterdam_JJA=agg_rotterdam_JJA.drop(columns=['month','year','starting_date']).reset_index(drop=True)
agg_rotterdam_SON=agg_rotterdam_SON.drop(columns=['month','year','starting_date']).reset_index(drop=True)

## Making dataframe with identical columns to the dataset used to train the machine learning model
# Making a column for the design (we are only operating with Dolphine 64 which is listed as 1 in the machine learning dataframe)
agg_rotterdam_DJF["design"] = 1
agg_rotterdam_MAM["design"] = 1
agg_rotterdam_JJA["design"] = 1
agg_rotterdam_SON["design"] = 1

# Making a column for speed over ground, whuch is a fixed value at 12.5, which is the speed of the vessel
agg_rotterdam_DJF["speed_over_ground"] = 12.5
agg_rotterdam_MAM["speed_over_ground"] = 12.5
agg_rotterdam_JJA["speed_over_ground"] = 12.5
agg_rotterdam_SON["speed_over_ground"] = 12.5

#########################################
### Fuel consumption for Route 1, December, January & February
# Making dummy variables of the apparent swell, wind and waves columns
rotterdam_DJF_fuel=agg_rotterdam_DJF.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(rotterdam_DJF_fuel.loc[:,columns_to_one_hot])
rotterdam_DJF_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
rotterdam_DJF_fuel = pd.concat([rotterdam_DJF_fuel,rotterdam_DJF_fuel_encoded],axis=1)
rotterdam_DJF_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)
rotterdam_DJF_fuel

#Getting rid of outliers from wave height
upper_limit_vh = rotterdam_DJF_fuel['vhm0'].mean() + 3*rotterdam_DJF_fuel['vhm0'].std()
lower_limit_vh = rotterdam_DJF_fuel['vhm0'].mean() - 3*rotterdam_DJF_fuel['vhm0'].std()
rotterdam_DJF_fuel['vhm0'] = np.where(
    rotterdam_DJF_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        rotterdam_DJF_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        rotterdam_DJF_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = rotterdam_DJF_fuel['vhm0_sw1'].mean() + 3*rotterdam_DJF_fuel['vhm0_sw1'].std()
lower_limit_sw = rotterdam_DJF_fuel['vhm0_sw1'].mean() - 3*rotterdam_DJF_fuel['vhm0_sw1'].std()
rotterdam_DJF_fuel['vhm0_sw1'] = np.where(
    rotterdam_DJF_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        rotterdam_DJF_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        rotterdam_DJF_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = rotterdam_DJF_fuel['speed_over_ground'].mean() + 3*rotterdam_DJF_fuel['speed_over_ground'].std()
lower_limit_sog = rotterdam_DJF_fuel['speed_over_ground'].mean() - 3*rotterdam_DJF_fuel['speed_over_ground'].std()
rotterdam_DJF_fuel['speed_over_ground'] = np.where(
    rotterdam_DJF_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        rotterdam_DJF_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        rotterdam_DJF_fuel['speed_over_ground']
    )
)
upper_limit_ws = rotterdam_DJF_fuel['wind_speed_kts'].mean() + 3*rotterdam_DJF_fuel['wind_speed_kts'].std()
lower_limit_ws = rotterdam_DJF_fuel['wind_speed_kts'].mean() - 3*rotterdam_DJF_fuel['wind_speed_kts'].std()
rotterdam_DJF_fuel['wind_speed_kts'] = np.where(
    rotterdam_DJF_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        rotterdam_DJF_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        rotterdam_DJF_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_rotterdam_DJF=model.predict(rotterdam_DJF_fuel)
rotterdam_DJF_fuel['fuel_consumption'] = predict_rotterdam_DJF

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
rotterdam_DJF_fuel['ID']= np.arange(len(rotterdam_DJF_fuel))
rotterdam_DJF_1['ID']=np.arange(len(rotterdam_DJF_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
rotterdam_DJF_both = pd.merge(rotterdam_DJF_fuel,rotterdam_DJF_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)
# Divide the df into what is ECA and what is not
rotterdam_DJF_ulsfo = rotterdam_DJF_both[rotterdam_DJF_both['eca']==1]
rotterdam_DJF_hfso = rotterdam_DJF_both[rotterdam_DJF_both['eca']==0]

#Grouping by starting date for each of the fuel types to get the total amount of fuel consumption for each type of fuel for each route
rotterdam_DJF_ulsfo = rotterdam_DJF_ulsfo.groupby([(rotterdam_DJF_ulsfo['starting_date'].dt.year),(rotterdam_DJF_ulsfo['starting_date'].dt.month),(rotterdam_DJF_ulsfo['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_DJF_hfso = rotterdam_DJF_hfso.groupby([(rotterdam_DJF_hfso['starting_date'].dt.year),(rotterdam_DJF_hfso['starting_date'].dt.month),(rotterdam_DJF_hfso['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})

# Group the dataframe by starting date to find the total consumption of fuel both in ECA and outside per voyage
rotterdam_DJF_voy_fuel = rotterdam_DJF_both.groupby([(rotterdam_DJF_both['starting_date'].dt.year),(rotterdam_DJF_both['starting_date'].dt.month),(rotterdam_DJF_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_DJF_voy_fuel=rotterdam_DJF_voy_fuel

# Making a plot of the total fuel consumption, regardless of fuel type for December, January and February
sns.distplot(rotterdam_DJF_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='DJF')
plt.legend()
plt.title("FC - Route 1 - DJF")
sns.plt.show()

#########################################
### Fuel consumption for Route 1, March, April & May
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
rotterdam_MAM_fuel=agg_rotterdam_MAM.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(rotterdam_MAM_fuel.loc[:,columns_to_one_hot])
rotterdam_MAM_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
rotterdam_MAM_fuel = pd.concat([rotterdam_MAM_fuel,rotterdam_MAM_fuel_encoded],axis=1)
rotterdam_MAM_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)
rotterdam_MAM_fuel

#Getting rid of outliers from wave height
upper_limit_vh = rotterdam_MAM_fuel['vhm0'].mean() + 3*rotterdam_MAM_fuel['vhm0'].std()
lower_limit_vh = rotterdam_MAM_fuel['vhm0'].mean() - 3*rotterdam_MAM_fuel['vhm0'].std()
rotterdam_MAM_fuel['vhm0'] = np.where(
    rotterdam_MAM_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        rotterdam_MAM_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        rotterdam_MAM_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = rotterdam_MAM_fuel['vhm0_sw1'].mean() + 3*rotterdam_MAM_fuel['vhm0_sw1'].std()
lower_limit_sw = rotterdam_MAM_fuel['vhm0_sw1'].mean() - 3*rotterdam_MAM_fuel['vhm0_sw1'].std()
rotterdam_MAM_fuel['vhm0_sw1'] = np.where(
    rotterdam_MAM_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        rotterdam_MAM_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        rotterdam_MAM_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = rotterdam_MAM_fuel['speed_over_ground'].mean() + 3*rotterdam_MAM_fuel['speed_over_ground'].std()
lower_limit_sog = rotterdam_MAM_fuel['speed_over_ground'].mean() - 3*rotterdam_MAM_fuel['speed_over_ground'].std()
rotterdam_MAM_fuel['speed_over_ground'] = np.where(
    rotterdam_MAM_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        rotterdam_MAM_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        rotterdam_MAM_fuel['speed_over_ground']
    )
)
upper_limit_ws = rotterdam_MAM_fuel['wind_speed_kts'].mean() + 3*rotterdam_MAM_fuel['wind_speed_kts'].std()
lower_limit_ws = rotterdam_MAM_fuel['wind_speed_kts'].mean() - 3*rotterdam_MAM_fuel['wind_speed_kts'].std()
rotterdam_MAM_fuel['wind_speed_kts'] = np.where(
    rotterdam_MAM_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        rotterdam_MAM_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        rotterdam_MAM_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_rotterdam_MAM=model.predict(rotterdam_MAM_fuel)
rotterdam_MAM_fuel['fuel_consumption'] = predict_rotterdam_MAM

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
rotterdam_MAM_fuel['ID']= np.arange(len(rotterdam_MAM_fuel))
rotterdam_MAM_1['ID']=np.arange(len(rotterdam_MAM_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
rotterdam_MAM_both = pd.merge(rotterdam_MAM_fuel,rotterdam_MAM_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)
# Divide the df into what is ECA and what is not
rotterdam_MAM_ulsfo = rotterdam_MAM_both[rotterdam_MAM_both['eca']==1]
rotterdam_MAM_hfso = rotterdam_MAM_both[rotterdam_MAM_both['eca']==0]

#Grouping by starting date for each of the fuel types to get the total amount of fuel consumption for each type of fuel for each route
rotterdam_MAM_ulsfo = rotterdam_MAM_ulsfo.groupby([(rotterdam_MAM_ulsfo['starting_date'].dt.year),(rotterdam_MAM_ulsfo['starting_date'].dt.month),(rotterdam_MAM_ulsfo['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_MAM_hfso = rotterdam_MAM_hfso.groupby([(rotterdam_MAM_hfso['starting_date'].dt.year),(rotterdam_MAM_hfso['starting_date'].dt.month),(rotterdam_MAM_hfso['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})

# Group the dataframe by starting date to find the total consumption of fuel both in ECA and outside per voyage
rotterdam_MAM_voy_fuel = rotterdam_MAM_both.groupby([(rotterdam_MAM_both['starting_date'].dt.year),(rotterdam_MAM_both['starting_date'].dt.month),(rotterdam_MAM_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_MAM_voy_fuel=rotterdam_MAM_voy_fuel
# Making a plot of the total fuel consumption, regardless of fuel type for March, April and May
sns.distplot(rotterdam_MAM_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='MAM')
plt.legend()
plt.title("FC - Route 1 - MAM")
sns.plt.show()

#########################################
### Fuel consumption for Route 1, June, July & August
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
rotterdam_JJA_fuel=agg_rotterdam_JJA.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(rotterdam_JJA_fuel.loc[:,columns_to_one_hot])
rotterdam_JJA_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
rotterdam_JJA_fuel = pd.concat([rotterdam_JJA_fuel,rotterdam_JJA_fuel_encoded],axis=1)
rotterdam_JJA_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)
rotterdam_JJA_fuel

#Getting rid of outliers from wave height
upper_limit_vh = rotterdam_JJA_fuel['vhm0'].mean() + 3*rotterdam_JJA_fuel['vhm0'].std()
lower_limit_vh = rotterdam_JJA_fuel['vhm0'].mean() - 3*rotterdam_JJA_fuel['vhm0'].std()
rotterdam_JJA_fuel['vhm0'] = np.where(
    rotterdam_JJA_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        rotterdam_JJA_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        rotterdam_JJA_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = rotterdam_JJA_fuel['vhm0_sw1'].mean() + 3*rotterdam_JJA_fuel['vhm0_sw1'].std()
lower_limit_sw = rotterdam_JJA_fuel['vhm0_sw1'].mean() - 3*rotterdam_JJA_fuel['vhm0_sw1'].std()
rotterdam_JJA_fuel['vhm0_sw1'] = np.where(
    rotterdam_JJA_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        rotterdam_JJA_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        rotterdam_JJA_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = rotterdam_JJA_fuel['speed_over_ground'].mean() + 3*rotterdam_JJA_fuel['speed_over_ground'].std()
lower_limit_sog = rotterdam_MAM_fuel['speed_over_ground'].mean() - 3*rotterdam_JJA_fuel['speed_over_ground'].std()
rotterdam_JJA_fuel['speed_over_ground'] = np.where(
    rotterdam_JJA_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        rotterdam_JJA_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        rotterdam_JJA_fuel['speed_over_ground']
    )
)
upper_limit_ws = rotterdam_JJA_fuel['wind_speed_kts'].mean() + 3*rotterdam_JJA_fuel['wind_speed_kts'].std()
lower_limit_ws = rotterdam_JJA_fuel['wind_speed_kts'].mean() - 3*rotterdam_JJA_fuel['wind_speed_kts'].std()
rotterdam_JJA_fuel['wind_speed_kts'] = np.where(
    rotterdam_JJA_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        rotterdam_JJA_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        rotterdam_JJA_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_rotterdam_JJA=model.predict(rotterdam_JJA_fuel)
rotterdam_JJA_fuel['fuel_consumption'] = predict_rotterdam_JJA

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
rotterdam_JJA_fuel['ID']= np.arange(len(rotterdam_JJA_fuel))
rotterdam_JJA_1['ID']=np.arange(len(rotterdam_JJA_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
rotterdam_JJA_both = pd.merge(rotterdam_JJA_fuel,rotterdam_JJA_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)
# Divide the df into what is ECA and what is not
rotterdam_JJA_ulsfo = rotterdam_JJA_both[rotterdam_JJA_both['eca']==1]
rotterdam_JJA_hfso = rotterdam_JJA_both[rotterdam_JJA_both['eca']==0]

#Grouping by starting date for each of the fuel types to get the total amount of fuel consumption for each type of fuel for each route
rotterdam_JJA_ulsfo = rotterdam_JJA_ulsfo.groupby([(rotterdam_JJA_ulsfo['starting_date'].dt.year),(rotterdam_JJA_ulsfo['starting_date'].dt.month),(rotterdam_JJA_ulsfo['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_JJA_hfso = rotterdam_JJA_hfso.groupby([(rotterdam_JJA_hfso['starting_date'].dt.year),(rotterdam_JJA_hfso['starting_date'].dt.month),(rotterdam_JJA_hfso['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})

# Group the dataframe by starting date to find the total consumption of fuel both in ECA and outside per voyage
rotterdam_JJA_voy_fuel = rotterdam_JJA_both.groupby([(rotterdam_JJA_both['starting_date'].dt.year),(rotterdam_JJA_both['starting_date'].dt.month),(rotterdam_JJA_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_JJA_voy_fuel=rotterdam_JJA_voy_fuel

# Making a plot of the total fuel consumption, regardless of fuel type for June, July and August
sns.distplot(rotterdam_JJA_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='JJA')
plt.legend()
plt.title("FC - Route 1 - JJA")
sns.plt.show()


#########################################
### Fuel consumption for Route 1, September, October & November
## Making dataframe with identical columns as the one used to train machine learning model
# Making dummy variables of the apparent swell, wind and waves columns
rotterdam_SON_fuel=agg_rotterdam_SON.reset_index(drop=True)
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['app_wind_dir', 'app_swell_dir', 'app_wave_dir']
encoded_array = enc.fit_transform(rotterdam_SON_fuel.loc[:,columns_to_one_hot])
rotterdam_SON_fuel_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
rotterdam_SON_fuel = pd.concat([rotterdam_SON_fuel,rotterdam_SON_fuel_encoded],axis=1)
rotterdam_SON_fuel.drop(labels= columns_to_one_hot,axis=1,inplace=True)
rotterdam_SON_fuel

#Getting rid of outliers from wave height
upper_limit_vh = rotterdam_SON_fuel['vhm0'].mean() + 3*rotterdam_SON_fuel['vhm0'].std()
lower_limit_vh = rotterdam_SON_fuel['vhm0'].mean() - 3*rotterdam_SON_fuel['vhm0'].std()
rotterdam_SON_fuel['vhm0'] = np.where(
    rotterdam_SON_fuel['vhm0']>upper_limit_vh,
    upper_limit_vh,
    np.where(
        rotterdam_SON_fuel['vhm0']<lower_limit_vh,
        lower_limit_vh,
        rotterdam_SON_fuel['vhm0']
    )
)
#Getting rid of outliers from swell height
upper_limit_sw = rotterdam_SON_fuel['vhm0_sw1'].mean() + 3*rotterdam_SON_fuel['vhm0_sw1'].std()
lower_limit_sw = rotterdam_SON_fuel['vhm0_sw1'].mean() - 3*rotterdam_SON_fuel['vhm0_sw1'].std()
rotterdam_SON_fuel['vhm0_sw1'] = np.where(
    rotterdam_SON_fuel['vhm0_sw1']>upper_limit_sw,
    upper_limit_sw,
    np.where(
        rotterdam_SON_fuel['vhm0_sw1']<lower_limit_sw,
        lower_limit_sw,
        rotterdam_SON_fuel['vhm0_sw1']
    )
)
#Getting rid of outliers from wind speed in knots
upper_limit_sog = rotterdam_SON_fuel['speed_over_ground'].mean() + 3*rotterdam_SON_fuel['speed_over_ground'].std()
lower_limit_sog = rotterdam_MAM_fuel['speed_over_ground'].mean() - 3*rotterdam_SON_fuel['speed_over_ground'].std()
rotterdam_SON_fuel['speed_over_ground'] = np.where(
    rotterdam_SON_fuel['speed_over_ground']>upper_limit_sog,
    upper_limit_sog,
    np.where(
        rotterdam_SON_fuel['speed_over_ground']<lower_limit_sog,
        lower_limit_sog,
        rotterdam_SON_fuel['speed_over_ground']
    )
)
upper_limit_ws = rotterdam_SON_fuel['wind_speed_kts'].mean() + 3*rotterdam_SON_fuel['wind_speed_kts'].std()
lower_limit_ws = rotterdam_SON_fuel['wind_speed_kts'].mean() - 3*rotterdam_SON_fuel['wind_speed_kts'].std()
rotterdam_SON_fuel['wind_speed_kts'] = np.where(
    rotterdam_SON_fuel['wind_speed_kts']>upper_limit_ws,
    upper_limit_ws,
    np.where(
        rotterdam_SON_fuel['wind_speed_kts']<lower_limit_ws,
        lower_limit_ws,
        rotterdam_SON_fuel['wind_speed_kts']
    )
)
model=joblib.load("/Volumes/LaCie/Master Thesis/Machine Learning Models/random_forest_model.joblib")
predict_rotterdam_SON=model.predict(rotterdam_SON_fuel)
rotterdam_SON_fuel['fuel_consumption'] = predict_rotterdam_SON

#After retrieving the fuel consumption and adding it to the dataframe as a column, we make an ID for each of the dataframes to merge them
rotterdam_SON_fuel['ID']= np.arange(len(rotterdam_SON_fuel))
rotterdam_SON_1['ID']=np.arange(len(rotterdam_SON_1))

# Merge to get starting date and fuel in order to group by start day to get information on consumption and cost per voyage
rotterdam_SON_both = pd.merge(rotterdam_SON_fuel,rotterdam_SON_1,on=['ID','vhm0','wind_speed_kts'],how="left").dropna(axis=0)
# Divide the df into what is ECA and what is not
rotterdam_SON_ulsfo = rotterdam_SON_both[rotterdam_SON_both['eca']==1]
rotterdam_SON_hfso = rotterdam_SON_both[rotterdam_SON_both['eca']==0]

#Grouping by starting date for each of the fuel types to get the total amount of fuel consumption for each type of fuel for each route
rotterdam_SON_ulsfo = rotterdam_SON_ulsfo.groupby([(rotterdam_SON_ulsfo['starting_date'].dt.year),(rotterdam_SON_ulsfo['starting_date'].dt.month),(rotterdam_SON_ulsfo['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_SON_hfso = rotterdam_SON_hfso.groupby([(rotterdam_SON_hfso['starting_date'].dt.year),(rotterdam_SON_hfso['starting_date'].dt.month),(rotterdam_SON_hfso['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})

# Group the dataframe by starting date to find the total consumption of fuel both in ECA and outside per voyage
rotterdam_SON_voy_fuel = rotterdam_SON_both.groupby([(rotterdam_SON_both['starting_date'].dt.year),(rotterdam_SON_both['starting_date'].dt.month),(rotterdam_SON_both['starting_date'].dt.day)],as_index=False).aggregate({'starting_date':lambda x: x.value_counts().index[0],'eca':lambda x: x.value_counts().index[0],'fuel_consumption':'sum'})
rotterdam_SON_voy_fuel=rotterdam_SON_voy_fuel

# Making a plot of the total fuel consumption, regardless of fuel type for September, October and November
sns.distplot(rotterdam_SON_voy_fuel["fuel_consumption"], hist = False, bins=10,kde = True, label='SON')
plt.legend()
plt.title("FC - Route 1 - SON")
sns.plt.show()

######################################### Total Fuel Consumption in one plot #########################################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(rotterdam_DJF_voy_fuel["fuel_consumption"], label='DJF', ax=ax)
sns.kdeplot(rotterdam_MAM_voy_fuel["fuel_consumption"], label='MAM', ax=ax)
sns.kdeplot(rotterdam_JJA_voy_fuel["fuel_consumption"], label='JJA', ax=ax)
sns.kdeplot(rotterdam_SON_voy_fuel["fuel_consumption"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - Route 1", fontsize=12)
plt.savefig('fc_r1.png', dpi = 300)
sns.plt.show()

######################################### Calculating TCE for Route 1 #########################################
rotterdam_DJF_case = pd.merge(rotterdam_DJF_ulsfo,rotterdam_DJF_hfso,on=['starting_date'],how='left')
rotterdam_MAM_case = pd.merge(rotterdam_MAM_ulsfo,rotterdam_MAM_hfso,on=['starting_date'],how='left')
rotterdam_JJA_case = pd.merge(rotterdam_JJA_ulsfo,rotterdam_JJA_hfso,on=['starting_date'],how='left')
rotterdam_SON_case = pd.merge(rotterdam_SON_ulsfo,rotterdam_SON_hfso,on=['starting_date'],how='left')

################# TCE calculation simulation, scenario 1 (low bunker price) ####################
##### DJF
# Used the random seed of 42 when simulating to get the results used for the thesis

# For this route, we used the total fuel consumption per voyage for both areas within the emisson control ares and outside
# We used both of these for calculating the TCE, using np.random.choice to retrieve one random variable for the TCE equation
# and then repeating this for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

random.seed(42)
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_DJF_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_DJF_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_DJF_case["TCE_DJF_c1"] = simulation(18,379.33,457.85,155278,46.07)

##### MAM
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_MAM_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_MAM_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_MAM_case["TCE_MAM_c1"] = simulation(18,379.33,457.85,155278,46.07)

##### JJA
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_JJA_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_JJA_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_JJA_case["TCE_JJA_c1"] = simulation(18,379.33,457.85,155278,46.07)

##### SON
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_SON_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_SON_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_SON_case["TCE_SON_c1"] = simulation(18,379.33,457.85,155278,46.07)

################# Plot of TCE distribution, scenario 1 (low bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(rotterdam_DJF_case["TCE_DJF_c1"], label='DJF', ax=ax)
sns.kdeplot(rotterdam_MAM_case["TCE_MAM_c1"], label='MAM', ax=ax)
sns.kdeplot(rotterdam_JJA_case["TCE_JJA_c1"], label='JJA', ax=ax)
sns.kdeplot(rotterdam_SON_case["TCE_SON_c1"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 1 - Scenario 1", fontsize=12)
plt.savefig('TCE_r1_c1.png', dpi = 300)
sns.plt.show()

################# TCE calculation simulation, scenario 2 (mean bunker price) ####################
##### DJF

# For this route, we used the total fuel consumption per voyage for both areas within the emisson control ares and outside
# We used both of these for calculating the TCE, using np.random.choice to retrieve one random variable for the TCE equation
# and then repeating this for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_DJF_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_DJF_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_DJF_case["TCE_DJF_c2"] = simulation(13.25,479,571,155278,46.07)

##### MAM
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_MAM_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_MAM_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_MAM_case["TCE_MAM_c2"] = simulation(13.25,479,571,155278,46.07)

##### JJA
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_JJA_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_JJA_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_JJA_case["TCE_JJA_c2"] = simulation(13.25,479,571,155278,46.07)

##### SON
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_SON_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_SON_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_SON_case["TCE_SON_c2"] = simulation(13.25,479,571,155278,46.07)

################# Plot of TCE distribution, scenario 2 (mean bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(rotterdam_DJF_case["TCE_DJF_c2"], label='DJF', ax=ax)
sns.kdeplot(rotterdam_MAM_case["TCE_MAM_c2"], label='MAM', ax=ax)
sns.kdeplot(rotterdam_JJA_case["TCE_JJA_c2"], label='JJA', ax=ax)
sns.kdeplot(rotterdam_SON_case["TCE_SON_c2"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 1 - Scenario 2", fontsize=12)
plt.savefig('TCE_r1_c2.png', dpi = 300)
sns.plt.show()

################# TCE calculation simulation, scenario 3 (high bunker price) ####################
##### DJF

# For this route, we used the total fuel consumption per voyage for both areas within the emisson control ares and outside
# We used both of these for calculating the TCE, using np.random.choice to retrieve one random variable for the TCE equation
# and then repeating this for each row of the dataframe to create a distribution of TCE from the fuel consumption distribution

def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_DJF_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_DJF_case['fuel_consumption_x'])    
    time_horizon = len(rotterdam_DJF_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_DJF_case["TCE_DJF_c3"] = simulation(21.25,549.21,636.18,155278,46.07)

##### MAM
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_MAM_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_MAM_case['fuel_consumption_x'])    
    time_horizon = len(rotterdam_MAM_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_MAM_case["TCE_MAM_c3"] = simulation(21.25,549.21,636.18,155278,46.07)

##### JJA
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_JJA_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_JJA_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_JJA_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_JJA_case["TCE_JJA_c3"] = simulation(21.25,549.21,636.18,155278,46.07)

##### SON
def simulation(freightrate,hfso,ulsfo,othercosts,duration):
    res1 = pd.DataFrame()
    fc_hsfo = (rotterdam_SON_case['fuel_consumption_y'])
    fc_ulsfo = (rotterdam_SON_case['fuel_consumption_x'])
    time_horizon = len(rotterdam_SON_case.index)
    TCE_result = []
    for dayss in range(time_horizon):
        TCE=((freightrate*50000)-(((np.random.choice(fc_hsfo))*hfso)+((np.random.choice(fc_ulsfo))*ulsfo))-othercosts)/duration
        TCE_result.append(TCE)
    res1[dayss]=TCE_result
    return res1
rotterdam_SON_case["TCE_SON_c3"] = simulation(21.25,549.21,636.18,155278,46.07)

################# Plot of TCE distribution, scenario 3 (high bunker price) ####################
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(rotterdam_DJF_case["TCE_DJF_c3"], label='DJF', ax=ax)
sns.kdeplot(rotterdam_MAM_case["TCE_MAM_c3"], label='MAM', ax=ax)
sns.kdeplot(rotterdam_JJA_case["TCE_JJA_c3"], label='JJA', ax=ax)
sns.kdeplot(rotterdam_SON_case["TCE_SON_c3"], label='SON', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - Route 1 - Scenario 3", fontsize=12)
plt.savefig('TCE_r1_c3.png', dpi = 300)
sns.plt.show()

# Retrieving statistics of mean, min and max fuel consumption and TCE values
route1_desc_DJF = rotterdam_DJF_case.describe()
route1_desc_MAM = rotterdam_MAM_case.describe()
route1_desc_JJA = rotterdam_JJA_case.describe()
route1_desc_SON = rotterdam_SON_case.describe()

# Using joblib to gather the dataframes in seperate scripts for comparing fuel consumption and TCE
import joblib
joblib.dump(rotterdam_DJF_case,"rotterdam_DJF_case.joblib")
joblib.dump(rotterdam_MAM_case,"rotterdam_MAM_case.joblib")
joblib.dump(rotterdam_JJA_case,"rotterdam_JJA_case.joblib")
joblib.dump(rotterdam_SON_case,"rotterdam_SON_case.joblib")










