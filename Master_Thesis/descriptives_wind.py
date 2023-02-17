#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:43:43 2022

@authors: Marie Log Staveland & Sia Benedikte Strømsnes
 
"""

#Importing necessary libraries 
import numpy as np
import pandas as pd
import glob as glob
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt 


#Create months and year list to call on folders from nc downloaded files
years=[year for year in range(2006,2020)]
months=["%.2d" % i for i in range(1,13)]


#Glob read all files inside a folder. Folders organized as year/month
file_names=[glob.glob("/Volumes/LaCie/Master Thesis/Codes/Wind/{}/{}/*.nc".format(x,y)) for x in years for y in months]
file_names = [item for sublist in file_names for item in sublist]


#Open the dataset 
ds = xr.open_mfdataset(file_names, engine='h5netcdf', phony_dims='sort')


#Choosing the variables I need for further use regarding the wind descriptives
variables = ['eastward_wind', 'northward_wind']

ds = ds[variables]

ds.head()


#Convert the the eastward wind data and the northward wind data to speed in knots
ds['wind_speed_kts'] = np.sqrt(ds['eastward_wind']**2+ds['northward_wind']**2)*1.94384449


#Here, we are slicing the coordinates to a fiiting area based on our pre-determined routes. 
#We have three different limits, one for the North Atlantic Ocean, one for 
#the South Atlantic Ocean, and the third for the Indian Ocean. We therefore set maximum
#latitude and longitude, as well as minimum latitude and longitude. This is 
#possible due to our xarray, meaning we have a dataset with three dimensions: latitude, longitude 
#time.

#First for the North Atlantic Ocean 
min_lon_n_atl = -100
min_lat_n_atl = 8
max_lon_n_atl = 10 
max_lat_n_atl = 58 

n_atlantic = ds.sel(lat=slice(min_lat_n_atl,max_lat_n_atl), lon=slice(min_lon_n_atl,max_lon_n_atl))


#Then for the South Atlantic Ocean
min_lon_s_atl = -70
min_lat_s_atl = -60
max_lon_s_atl = 27 
max_lat_s_atl = 8 

s_atlantic = ds.sel(lat=slice(min_lat_s_atl,max_lat_s_atl), lon=slice(min_lon_s_atl,max_lon_s_atl))


#Last, for the Indian Ocean
min_lon_ind = 14
min_lat_ind = -40
max_lon_ind = 121
max_lat_ind = 30 

ind_ocean = ds.sel(lat=slice(min_lat_ind,max_lat_ind), lon=slice(min_lon_ind,max_lon_ind))


#In order to split the datasets into seasons, we will have to select the which months
#we sould like to group into seasons. For the North Atlantic Ocean,
#we chose to divide it into winter (December, January, and February) which was indexed with 
#12, 1, and 2. The Spring (March, April, and May) was indexed 3, 4, and 5), 
#whereas, the summer (June, July, and August) was indexed 6, 7, and 8. The remaning months 
#(September, October, and November) was categorized as autumn, which was indexed 8, 9, and 10.

n_atlantic_win = n_atlantic.isel(time=n_atlantic.time.dt.month.isin([12,1,2]))
n_atlantic_spr = n_atlantic.isel(time=n_atlantic.time.dt.month.isin([3,4,5]))
n_atlantic_sum = n_atlantic.isel(time=n_atlantic.time.dt.month.isin([6,7,8]))
n_atlantic_aut = n_atlantic.isel(time=n_atlantic.time.dt.month.isin([9,10,11])) 


#Similarly, we did this for the South Atlantic Ocean. However the seasons here are somewhat 
#different as the winter are June, July, and August. The spring are September, October, and November, whereas 
#the Summer are December, January, and February. Last, but not least, the autumn accounts for the months 
#March, April and May. 

s_atlantic_sum = s_atlantic.isel(time=s_atlantic.time.dt.month.isin([12,1,2]))
s_atlantic_aut = s_atlantic.isel(time=s_atlantic.time.dt.month.isin([3,4,5]))
s_atlantic_win = s_atlantic.isel(time=s_atlantic.time.dt.month.isin([6,7,8]))
s_atlantic_spr = s_atlantic.isel(time=s_atlantic.time.dt.month.isin([9,10,11])) 


#Cocerning the Indian Ocea, they operate with different kinds of seasons compared to the Atlantic Ocean.
#This means that the winter are December, January, and February, and the summer are 
# March, April and May. From June, throughout September, we have the monsoon-season in the Indian Ocean,
# which leads us to the teo remaining months, October and November, which is categorized as post-monsoon.   

ind_ocean_win = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([12,1,2]))
ind_ocean_sum = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([3,4,5]))
ind_ocean_mon = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([6,7,8]))
ind_ocean_postmon = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([9,10,11])) 


#To make a simple plot showing the wind speed in knots, we used the following code. 

n_atlantic_win_WS = n_atlantic_win.mean("time")
plot_n_atlantic_win = n_atlantic_win_WS.wind_speed_kts
plot_n_atlantic_win.plot()
plt.savefig('wind_example.png')


#Afterwards, we made plots of the North Atlantic Ocean displaying the wind speed in knots based on seasons.
#We chose to average the data across time, meaning the illustration only depicts the averaged values. Hence, 
#it does not take extreme weather events into account. 

#Plot of mean wind speed in knots in the North Atlantic Ocean during the winter
n_atlantic_win_WS = n_atlantic_win.mean("time")
plot_n_atlantic_win = n_atlantic_win_WS.wind_speed_kts


#Plot of mean wind speed in knots in the North Atlantic Ocean during the spring
n_atlantic_spr_WS = n_atlantic_spr.mean("time")
plot_n_atlantic_spr = n_atlantic_spr_WS.wind_speed_kts


#Plot of mean wind speed in knots in the North Atlantic Ocean during the summer
n_atlantic_sum_WS = n_atlantic_sum.mean("time")
plot_n_atlantic_sum = n_atlantic_sum_WS.wind_speed_kts


#Plot of mean wind speed in knots in the North Atlantic Ocean during the autumn
n_atlantic_aut_WS = n_atlantic_aut.mean("time")
plot_n_atlantic_aut = n_atlantic_aut_WS.wind_speed_kts


#Combine all the plots together to get a better overview if the differences in the 
#North Atlantic Ocean based on the seasons. made levels to better structure the wind speed. 

levels = [0, 5, 10, 15, 20, 25, 30]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Wind Speed in the North Atlantic Ocean", fontsize=16)

plot_n_atlantic_win.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Knots"})
ax1.set_title("DJF")

plot_n_atlantic_spr.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Knots"})
ax2.set_title("MAM")

plot_n_atlantic_sum.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Knots"})
ax3.set_title("JJA")

plot_n_atlantic_aut.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Knots"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('NA_WS.png')


#Did the same for the South Atlantic Ocean 
#Plot of mean wind speed in knots in the South Atlantic Ocean during the winter
s_atlantic_win_WS = s_atlantic_win.mean("time")
plot_s_atlantic_win = s_atlantic_win_WS.wind_speed_kts


#Plot of mean wind speed in knots in the South Atlantic Ocean during the spring
s_atlantic_spr_WS = s_atlantic_spr.mean("time")
plot_s_atlantic_spr = s_atlantic_spr_WS.wind_speed_kts


#Plot of mean wind speed in knots in the South Atlantic Ocean during the summer
s_atlantic_sum_WS = s_atlantic_sum.mean("time")
plot_s_atlantic_sum = s_atlantic_sum_WS.wind_speed_kts


#Plot of mean wind speed in knots in the South Atlantic Ocean during the winter
s_atlantic_aut_WS = s_atlantic_aut.mean("time")
plot_s_atlantic_aut = s_atlantic_aut_WS.wind_speed_kts


#Combine all the plots together to get a better overview if the differences in the 
#South Atlantic Ocean based on the seasons. Made levels to better structure the wind speed. 

levels = [0, 5, 10, 15, 20, 25, 30]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Wind Speed in the South Atlantic Ocean", fontsize=16)

plot_s_atlantic_sum.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Knots"})
ax1.set_title("DJF")

plot_s_atlantic_aut.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Knots"})
ax2.set_title("MAM")

plot_s_atlantic_win.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Knots"})
ax3.set_title("JJA")

plot_s_atlantic_spr.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Knots"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('SA_WS.png')


#Did the same for the Indian Ocean.
#Plot of mean wind speed in knots in the Indian Ocean during the winter.
ind_ocean_win_WS = ind_ocean_win.mean("time")
plot_ind_ocean_win = ind_ocean_win_WS.wind_speed_kts


#Plot of mean wind speed in knots in the Indian Ocean during the summer.
ind_ocean_sum_WS = ind_ocean_sum.mean("time")
plot_ind_ocean_sum = ind_ocean_sum_WS.wind_speed_kts


#Plot of mean wind speed in knots in the Indian Ocean during the monsoon.
ind_ocean_mon_WS = ind_ocean_mon.mean("time")
plot_ind_ocean_mon = ind_ocean_mon_WS.wind_speed_kts


#Plot of mean wind speed in knots in the Indian Ocean during the post-monsoon.
ind_ocean_postmon_WS = ind_ocean_postmon.mean("time")
plot_ind_ocean_postmon = ind_ocean_postmon_WS.wind_speed_kts


#Combine all the plots together to get a better overview if the differences in the 
#Indian Ocean based on the seasons. Made levels to better structure the wind speed. 

levels = [0, 5, 10, 15, 20, 25, 30]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Wind Speed in the Indian Ocean", fontsize=16)

plot_ind_ocean_win.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Knots"})
ax1.set_title("DJF")

plot_ind_ocean_sum.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Knots"})
ax2.set_title("MAM")

plot_ind_ocean_mon.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Knots"})
ax3.set_title("JJA")

plot_ind_ocean_postmon.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Knots"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('IO_WS.png')


#For the summary statistics, we are only interested in wind speed in knots. We therefore
#chooses to select only this variable, and remove the rest. This is to avoid inecessary 
#data in our working frames, as it slows down the computer. 

#To do this, we made a new variable, called wind_speed_kts. We attached this variable 
#to alle our datasets.

#Started to this for the North Atlantic Ocean.

wind_speed = ['wind_speed_kts']

n_atlantic_win_arr = n_atlantic_win[wind_speed]
n_atlantic_spr_arr = n_atlantic_spr[wind_speed]
n_atlantic_sum_arr = n_atlantic_sum[wind_speed]
n_atlantic_aut_arr = n_atlantic_aut[wind_speed]


#To be able to extract summary statistics, it is easier to tranform the 
#datasets to either a 2d dataframe using Pandas, or making an array of the variable 
#we need with Numpy. As dataframes only are more spacious, and therefor takes more 
#time to use, when working with such large numbers, we decided to use numpy arryays. 
#Therefore, we made array for all the datasets over the North Atlantic Ocean.

n_atlantic_win_arr = n_atlantic_win_arr.to_array()
n_atlantic_spr_arr = n_atlantic_spr_arr.to_array()
n_atlantic_sum_arr = n_atlantic_sum_arr.to_array()
n_atlantic_aut_arr = n_atlantic_aut_arr.to_array()


#Afterwards, we retrieved the values from the arrays. 

n_atlantic_win_arr = n_atlantic_win_arr.values
n_atlantic_spr_arr = n_atlantic_spr_arr.values
n_atlantic_sum_arr = n_atlantic_sum_arr.values
n_atlantic_aut_arr = n_atlantic_aut_arr.values


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

n_atlantic_win_arr = n_atlantic_win_arr[~np.isnan(n_atlantic_win_arr)]
n_atlantic_spr_arr = n_atlantic_spr_arr[~np.isnan(n_atlantic_spr_arr)]
n_atlantic_sum_arr = n_atlantic_sum_arr[~np.isnan(n_atlantic_sum_arr)]
n_atlantic_aut_arr = n_atlantic_aut_arr[~np.isnan(n_atlantic_aut_arr)]


#Since numpy array do not have a specific describe() function, similar to pandas,
#we printed out seperate values. We found the mean, standard deviation, 
#maxmimum value, minimum value, and the 95% quantile. The reason for the 
#choice of measures, was that we wanted to look at the avreage values across time, but also 
#if there were any individual weather events worth noticing. Also, the normal range and 
#fluctuations is interesting to know more about. 

#Summary statistics of DJF in the North Atlantic Ocean

print(n_atlantic_win_arr.mean())
print(n_atlantic_win_arr.std())
print(n_atlantic_win_arr.max())
print(n_atlantic_win_arr.min())
print(np.quantile(n_atlantic_win_arr, 0.95))


#Summary statistics of MAM in the North Atlantic Ocean

print(n_atlantic_spr_arr.mean())
print(n_atlantic_spr_arr.std())
print(n_atlantic_spr_arr.max())
print(n_atlantic_spr_arr.min())
print(np.quantile(n_atlantic_spr_arr, 0.95))


#Summary statistics of JJA in the North Atlantic Ocean

print(n_atlantic_sum_arr.mean())
print(n_atlantic_sum_arr.std())
print(n_atlantic_sum_arr.max())
print(n_atlantic_sum_arr.min())
print(np.quantile(n_atlantic_sum_arr, 0.95))


#Summary statistics of SON in the North Atlantic Ocean

print(n_atlantic_aut_arr.mean())
print(n_atlantic_aut_arr.std())
print(n_atlantic_aut_arr.max())
print(n_atlantic_aut_arr.min())
print(np.quantile(n_atlantic_aut_arr, 0.95))


#Selected wind_speed_kts to all our datasets, and removed eastward and northward 
#wind as we do not need it anymore

s_atlantic_win_arr = s_atlantic_win[wind_speed]
s_atlantic_spr_arr = s_atlantic_spr[wind_speed]
s_atlantic_sum_arr = s_atlantic_sum[wind_speed]
s_atlantic_aut_arr = s_atlantic_aut[wind_speed]


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

s_atlantic_win_arr = s_atlantic_win_arr.to_array()
s_atlantic_spr_arr = s_atlantic_spr_arr.to_array()
s_atlantic_sum_arr = s_atlantic_sum_arr.to_array()
s_atlantic_aut_arr = s_atlantic_aut_arr.to_array()


#Afterwards, we retrieved the values from the arrays. 

s_atlantic_win_arr = s_atlantic_win_arr.values
s_atlantic_spr_arr = s_atlantic_spr_arr.values
s_atlantic_sum_arr = s_atlantic_sum_arr.values
s_atlantic_aut_arr = s_atlantic_aut_arr.values


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

s_atlantic_win_arr = s_atlantic_win_arr[~np.isnan(s_atlantic_win_arr)]
s_atlantic_spr_arr = s_atlantic_spr_arr[~np.isnan(s_atlantic_spr_arr)]
s_atlantic_sum_arr = s_atlantic_sum_arr[~np.isnan(s_atlantic_sum_arr)]
s_atlantic_aut_arr = s_atlantic_aut_arr[~np.isnan(s_atlantic_aut_arr)]

#Summary statistics of DJF in the South Atlantic Ocean

print(s_atlantic_sum_arr.mean())
print(s_atlantic_sum_arr.std())
print(s_atlantic_sum_arr.max())
print(s_atlantic_sum_arr.min())
print(np.quantile(s_atlantic_sum_arr, 0.95))


#Summary statistics of MAM in the South Atlantic Ocean

print(s_atlantic_aut_arr.mean())
print(s_atlantic_aut_arr.std())
print(s_atlantic_aut_arr.max())
print(s_atlantic_aut_arr.min())
print(np.quantile(s_atlantic_aut_arr, 0.95))


#Summary statistics of JJA in the South Atlantic Ocean

print(s_atlantic_win_arr.mean())
print(s_atlantic_win_arr.std())
print(s_atlantic_win_arr.max())
print(s_atlantic_win_arr.min())
print(np.quantile(s_atlantic_win_arr, 0.95))


#Summary statistics of SON in the South Atlantic Ocean

print(s_atlantic_spr_arr.mean())
print(s_atlantic_spr_arr.std())
print(s_atlantic_spr_arr.max())
print(s_atlantic_spr_arr.min())
print(np.quantile(s_atlantic_spr_arr, 0.95))


#Selected wind_speed_kts to all our datasets, and removed eastward and northward 
#wind as we do not need it anymore. 

ind_ocean_win_arr = ind_ocean_win[wind_speed]
ind_ocean_sum_arr = ind_ocean_sum[wind_speed]
ind_ocean_mon_arr = ind_ocean_mon[wind_speed]
ind_ocean_postmon_arr = ind_ocean_postmon[wind_speed]


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

ind_ocean_win_arr = ind_ocean_win_arr.to_array()
ind_ocean_sum_arr = ind_ocean_sum_arr.to_array()
ind_ocean_mon_arr = ind_ocean_mon_arr.to_array()
ind_ocean_postmon_arr = ind_ocean_postmon_arr.to_array()


#Afterwards, we retrieved the values from the arrays. 

ind_ocean_win_arr = ind_ocean_win_arr.values
ind_ocean_sum_arr = ind_ocean_sum_arr.values
ind_ocean_mon_arr = ind_ocean_mon_arr.values
ind_ocean_postmon_arr = ind_ocean_postmon_arr.values



#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

ind_ocean_win_arr = ind_ocean_win_arr[~np.isnan(ind_ocean_win_arr)]
ind_ocean_sum_arr = ind_ocean_sum_arr[~np.isnan(ind_ocean_sum_arr)]
ind_ocean_mon_arr = ind_ocean_mon_arr[~np.isnan(ind_ocean_mon_arr)]
ind_ocean_postmon_arr = ind_ocean_postmon_arr[~np.isnan(ind_ocean_postmon_arr)]

#Summary statistics of DJF in the Indian Ocean

print(ind_ocean_win_arr.mean())
print(ind_ocean_win_arr.std())
print(ind_ocean_win_arr.max())
print(ind_ocean_win_arr.min())
print(np.quantile(ind_ocean_win_arr, 0.95))


#Summary statistics of MAM in the Indian Ocean

print(ind_ocean_sum_arr.mean())
print(ind_ocean_sum_arr.std())
print(ind_ocean_sum_arr.max())
print(ind_ocean_sum_arr.min())
print(np.quantile(ind_ocean_sum_arr, 0.95))


#Summary statistics of JJA in the Indian Ocean

print(ind_ocean_mon_arr.mean())
print(ind_ocean_mon_arr.std())
print(ind_ocean_mon_arr.max())
print(ind_ocean_mon_arr.min())
print(np.quantile(ind_ocean_mon_arr, 0.95))


#Summary statistics of  in the Indian Ocean

print(ind_ocean_postmon_arr.mean())
print(ind_ocean_postmon_arr.std())
print(ind_ocean_postmon_arr.max())
print(ind_ocean_postmon_arr.min())
print(np.quantile(ind_ocean_postmon_arr, 0.95))
