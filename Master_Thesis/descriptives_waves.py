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
file_names=[glob.glob("/Volumes/LaCie/Master Thesis/Codes/Waves/{}/{}/*.nc".format(x,y)) for x in years for y in months]
file_names = [item for sublist in file_names for item in sublist]


#Open the dataset 
ds = xr.open_mfdataset(file_names, engine='h5netcdf', phony_dims='sort')


#Choosing the variables I need for further use regarding the wave descriptives
variables = ['VHM0', 'VHM0_SW1', 'VMDR', 'VMDR_SW1']

ds = ds[variables]

ds.head()

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


n_atlantic = ds.sel(latitude=slice(max_lat_n_atl, min_lat_n_atl), longitude=slice(min_lon_n_atl,max_lon_n_atl))


#Then for the South Atlantic Ocean
min_lon_s_atl = -70
min_lat_s_atl = -60
max_lon_s_atl = 27 
max_lat_s_atl = 8 


s_atlantic = ds.sel(latitude=slice(max_lat_s_atl, min_lat_s_atl), longitude=slice(min_lon_s_atl,max_lon_s_atl))


#Last, for the Indian Ocean
min_lon_ind = 14
min_lat_ind = -40
max_lon_ind = 121
max_lat_ind = 30 


ind_ocean = ds.sel(latitude=slice(max_lat_ind, min_lat_ind), longitude=slice(min_lon_ind,max_lon_ind))


#In order to split the datasets into periods, we will have to select the which months
#we would like to group. For the North Atlantic Ocean,
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


#Cocerning the Indian Ocean, they operate with different kinds of seasons compared to the Atlantic Ocean.
#This means that the winter are December, January, and February, and the summer are 
# March, April and May. From June, throughout August, we have the monsoon-season in the Indian Ocean,
# which leads us to the three remaining months, September, October and November, which is categorized as post-monsoon.   

ind_ocean_win = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([12,1,2]))
ind_ocean_sum = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([3,4,5]))
ind_ocean_mon = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([6,7,8]))
ind_ocean_postmon = ind_ocean.isel(time=ind_ocean.time.dt.month.isin([9,10,11])) 


#Afterwards, we made plots of the North Atlantic Ocean displaying the wave height based on seasons.
#We chose to average the data across time, meaning the illustration only depicts the averaged values. Hence, 
#it does not take extreme weather events into account. 

#Plot of mean wave height in the North Atlantic Ocean during DJF
n_atl_win_VHM0 = n_atlantic_win.mean("time")
plot_n_atl_VHM0_win = n_atl_win_VHM0.VHM0


#Plot of mean wave height in the North Atlantic Ocean during MAM
n_atl_spr_VHM0 = n_atlantic_spr.mean("time")
plot_n_atl_VHM0_spr = n_atl_spr_VHM0.VHM0


#Plot of mean wave height in the North Atlantic Ocean during JJA
n_atl_sum_VHM0 = n_atlantic_sum.mean("time")
plot_n_atl_VHM0_sum = n_atl_sum_VHM0.VHM0


#Plot of mean wave height in the North Atlantic Ocean during SON
n_atl_aut_VHM0 = n_atlantic_aut.mean("time")
plot_n_atl_VHM0_aut = n_atl_aut_VHM0.VHM0


#Combine all the plots together to get a better overview if the differences in the 
#North Atlantic Ocean based on the seasons. Made levels to better structure the wave height. 

levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Wave Height in the North Atlantic Ocean", fontsize=16)

plot_n_atl_VHM0_win.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Meters"})
ax1.set_title("DJF")

plot_n_atl_VHM0_spr.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Meters"})
ax2.set_title("MAM")

plot_n_atl_VHM0_sum.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Meters"})
ax3.set_title("JJA")

plot_n_atl_VHM0_aut.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Meters"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('NA_VHM0.png')


#Did the same for the South Atlantic Ocean 
#Plot of mean wave height in the South Atlantic Ocean during DJF
s_atl_win_VHM0 = s_atlantic_win.mean("time")
plot_s_atl_VHM0_win = s_atl_win_VHM0.VHM0


#Plot of mean wave height in the South Atlantic Ocean during MAM
s_atl_spr_VHM0 = s_atlantic_spr.mean("time")
plot_s_atl_VHM0_spr = s_atl_spr_VHM0.VHM0


#Plot of mean wave height in the South Atlantic Ocean during JJA
s_atl_sum_VHM0 = s_atlantic_sum.mean("time")
plot_s_atl_VHM0_sum = s_atl_sum_VHM0.VHM0


#Plot of mean wave height in the South Atlantic Ocean during SON
s_atl_aut_VHM0 = s_atlantic_aut.mean("time")
plot_s_atl_VHM0_aut = s_atl_aut_VHM0.VHM0


#Combine all the plots together to get a better overview if the differences in the 
#South Atlantic Ocean based on the seasons. Made levels to better structure the wave height. 

levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Wave Height in the South Atlantic Ocean", fontsize=16)

plot_s_atl_VHM0_sum.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Meters"})
ax1.set_title("DJF")

plot_s_atl_VHM0_aut.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Meters"})
ax2.set_title("MAM")

plot_s_atl_VHM0_win.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Meters"})
ax3.set_title("JJA")

plot_s_atl_VHM0_spr.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Meters"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('SA_VHM0.png')


#Did the same for the Indian Ocean.
#Plot of mean wave height in the Indian Ocean during DJF
ind_ocean_win_VHM0 = ind_ocean_win.mean("time")
plot_ind_VHM0_win = ind_ocean_win_VHM0.VHM0


#Plot of mean wave height in the Indian Ocean during MAM
ind_ocean_sum_VHM0 = ind_ocean_sum.mean("time")
plot_ind_VHM0_sum = ind_ocean_sum_VHM0.VHM0


#Plot of mean wave height in the Indian Ocean during JJA
ind_ocean_mon_VHM0 = ind_ocean_mon.mean("time")
plot_ind_VHM0_mon = ind_ocean_mon_VHM0.VHM0


#Plot of mean wave height in the Indian Ocean during SON
ind_ocean_postmon_VHM0 = ind_ocean_postmon.mean("time")
plot_ind_VHM0_postmon = ind_ocean_postmon_VHM0.VHM0


#Combine all the plots together to get a better overview if the differences in the 
#Indian Ocean based on the seasons. Made levels to better structure the wave height. 

levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Wave Height in the Indian Ocean", fontsize=16)

plot_ind_VHM0_win.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Meters"})
ax1.set_title("DJF")

plot_ind_VHM0_sum.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Meters"})
ax2.set_title("MAM")

plot_ind_VHM0_mon.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Meters"})
ax3.set_title("JJA")

plot_ind_VHM0_postmon.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Meters"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('IO_VHM0.png')


#Plot of swell height in the North Atlantic Ocean during DJF
n_atl_win_VHM0SW1 = n_atlantic_win.mean("time")
plot_n_atl_VHM0SW1_win = n_atl_win_VHM0SW1.VHM0_SW1


#Plot of swell height in the North Atlantic Ocean during MAM
n_atl_spr_VHM0SW1 = n_atlantic_spr.mean("time")
plot_n_atl_VHM0SW1_spr = n_atl_spr_VHM0SW1.VHM0_SW1


#Plot of swell height in the North Atlantic Ocean during JJA
n_atl_sum_VHM0SW1 = n_atlantic_sum.mean("time")
plot_n_atl_VHM0SW1_sum = n_atl_sum_VHM0SW1.VHM0_SW1


#Plot of swell height in the North Atlantic Ocean during SON
n_atl_aut_VHM0SW1 = n_atlantic_aut.mean("time")
plot_n_atl_VHM0SW1_aut = n_atl_aut_VHM0SW1.VHM0_SW1


#Combine all the plots together to get a better overview if the differences in the 
#North Atlantic Ocean based on the seasons. made levels to better structure swell height. 

levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Swell Height in the North Atlantic Ocean", fontsize=16)

plot_n_atl_VHM0SW1_win.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Meters"})
ax1.set_title("DJF")

plot_n_atl_VHM0SW1_spr.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Meters"})
ax2.set_title("MAM")

plot_n_atl_VHM0SW1_sum.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Meters"})
ax3.set_title("JJA")

plot_n_atl_VHM0SW1_aut.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Meters"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('NA_SW1.png')


#Did the same for the South Atlantic Ocean 
#Plot of mean swell height in the South Atlantic Ocean during DJF
s_atl_win_VHM0SW1 = s_atlantic_win.mean("time")
plot_s_atl_VHM0SW1_win = s_atl_win_VHM0SW1.VHM0_SW1


#Plot of mean swell height in the South Atlantic Ocean during MAM
s_atl_spr_VHM0SW1 = s_atlantic_spr.mean("time")
plot_s_atl_VHM0SW1_spr = s_atl_spr_VHM0SW1.VHM0_SW1


#Plot of mean swell height in the South Atlantic Ocean during JJA
s_atl_sum_VHM0SW1 = s_atlantic_sum.mean("time")
plot_s_atl_VHM0SW1_sum = s_atl_sum_VHM0SW1.VHM0_SW1


#Plot of mean swell height in the South Atlantic Ocean during SON
s_atl_aut_VHM0SW1 = s_atlantic_aut.mean("time")
plot_s_atl_VHM0SW1_aut = s_atl_aut_VHM0SW1.VHM0_SW1


#Combine all the plots together to get a better overview if the differences in the 
#South Atlantic Ocean based on the seasons. Made levels to better structure the swell height. 

levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Swell Height in the South Atlantic Ocean", fontsize=16)

plot_s_atl_VHM0SW1_sum.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Meters"})
ax1.set_title("DJF")

plot_s_atl_VHM0SW1_aut.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Meters"})
ax2.set_title("MAM")

plot_s_atl_VHM0SW1_win.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Meters"})
ax3.set_title("JJA")

plot_s_atl_VHM0SW1_spr.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Meters"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('SA_SW1.png')


#Did the same for the Indian Ocean.
#Plot of mean swell height in the Indian Ocean during DJF
ind_ocean_win_VHM0SW1 = ind_ocean_win.mean("time")
plot_ind_VHM0SW1_win = ind_ocean_win_VHM0SW1.VHM0_SW1


#Plot of mean swell height in the Indian Ocean during MAM
ind_ocean_sum_VHM0SW1 = ind_ocean_sum.mean("time")
plot_ind_VHM0SW1_sum = ind_ocean_sum_VHM0SW1.VHM0_SW1


#Plot of mean swell height in the Indian Ocean during JJA
ind_ocean_mon_VHM0SW1 = ind_ocean_mon.mean("time")
plot_ind_VHM0SW1_mon = ind_ocean_mon_VHM0SW1.VHM0_SW1


#Plot of mean swell height in the Indian Ocean during SON
ind_ocean_postmon_VHM0SW1 = ind_ocean_postmon.mean("time")
plot_ind_VHM0SW1_postmon = ind_ocean_postmon_VHM0SW1.VHM0_SW1


#Combine all the plots together to get a better overview if the differences in the 
#Indian Ocean based on the seasons. Made levels to better structure the swell height. 

levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
f.suptitle("Swell Height in the Indian Ocean", fontsize=16)

plot_ind_VHM0SW1_win.plot(ax=ax1, levels = levels, cbar_kwargs={"label": "Meters"})
ax1.set_title("DJF")

plot_ind_VHM0SW1_sum.plot(ax=ax2, levels = levels, cbar_kwargs={"label": "Meters"})
ax2.set_title("MAM")

plot_ind_VHM0SW1_mon.plot(ax=ax3, levels = levels, cbar_kwargs={"label": "Meters"})
ax3.set_title("JJA")

plot_ind_VHM0SW1_postmon.plot(ax=ax4, levels = levels, cbar_kwargs={"label": "Meters"})
ax4.set_title("SON")

f.subplots_adjust(top=0.88)
plt.tight_layout()
plt.savefig('IO_SW1.png')


#For the summary statistics, we are only interested in wave height. We therefore
#chooses to select only this variable, and remove the rest. This is to avoid inecessary 
#data in our working frames, as it slows down the computer. 

#To do this, we made a new variable, called wave_height. We attached this variable 
#to alle our datasets.

#Started to this for the North Atlantic Ocean.

wave_height = ['VHM0']

n_atl_win_WH = n_atlantic_win[wave_height]
n_atl_spr_WH = n_atlantic_spr[wave_height]
n_atl_sum_WH = n_atlantic_sum[wave_height]
n_atl_aut_WH = n_atlantic_aut[wave_height]


#To be able to extract summary statistics, it is easier to tranform the 
#datasets to either a 2d dataframe using Pandas, or making an array of the variable 
#we need with Numpy. As dataframes only are more spacious, and therefor takes more 
#time to use, when working with such large numbers, we decided to use numpy arryays. 
#Therefore, we made array for all the datasets over the North Atlantic Ocean.

n_atl_win_WH = n_atl_win_WH.to_array()
n_atl_spr_WH = n_atl_spr_WH.to_array()
n_atl_sum_WH = n_atl_sum_WH.to_array()
n_atl_aut_WH = n_atl_aut_WH.to_array()


#Afterwards, we retrieved the values from the arrays. 

n_atl_win_WH=n_atl_win_WH.values
n_atl_spr_WH = n_atl_spr_WH.values
n_atl_sum_WH = n_atl_sum_WH.values
n_atl_aut_WH = n_atl_aut_WH.values


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

n_atl_win_WH = n_atl_win_WH[~np.isnan(n_atl_win_WH)]
n_atl_spr_WH = n_atl_spr_WH[~np.isnan(n_atl_spr_WH)]
n_atl_sum_WH = n_atl_sum_WH[~np.isnan(n_atl_sum_WH)]
n_atl_aut_WH = n_atl_aut_WH[~np.isnan(n_atl_aut_WH)]


#Since numpy array do not have a specific describe() function, similar to pandas,
#we printed out seperate values. We found the mean, standard deviation, 
#maxmimum value, minimum value, and the 95% quantile. The reason for the 
#choice of measures, was that we wanted to look at the avreage values across time, but also 
#if there were any individual weather events worth noticing. Also, the normal range and 
#fluctuations is interesting to know more about. 

#Summary statistics of DJF in the North Atlantic Ocean

print(n_atl_win_WH.mean())
print(n_atl_win_WH.std())
print(n_atl_win_WH.max())
print(n_atl_win_WH.min())
print(np.quantile(n_atl_win_WH, 0.95))


#Summary statistics of MAM in the North Atlantic Ocean

print(n_atl_spr_WH.mean())
print(n_atl_spr_WH.std())
print(n_atl_spr_WH.max())
print(n_atl_spr_WH.min())
print(np.quantile(n_atl_spr_WH, 0.95))


#Summary statistics of JJA in the North Atlantic Ocean

print(n_atl_sum_WH.mean())
print(n_atl_sum_WH.std())
print(n_atl_sum_WH.max())
print(n_atl_sum_WH.min())
print(np.quantile(n_atl_sum_WH, 0.95))


#Summary statistics of SON in the North Atlantic Ocean

print(n_atl_aut_WH.mean())
print(n_atl_aut_WH.std())
print(n_atl_aut_WH.max())
print(n_atl_aut_WH.min())
print(np.quantile(n_atl_aut_WH, 0.95))


#Selected wave_height to all our datasets

s_atl_win_WH = s_atlantic_win[wave_height]
s_atl_spr_WH = s_atlantic_spr[wave_height]
s_atl_sum_WH = s_atlantic_sum[wave_height]
s_atl_aut_WH = s_atlantic_aut[wave_height]


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

s_atl_win_WH = s_atl_win_WH.to_array()
s_atl_spr_WH = s_atl_spr_WH.to_array()
s_atl_sum_WH = s_atl_sum_WH.to_array()
s_atl_aut_WH = s_atl_aut_WH.to_array()


#Afterwards, we retrieved the values from the arrays. 

s_atl_win_WH = s_atl_win_WH.values
s_atl_spr_WH = s_atl_spr_WH.values
s_atl_sum_WH = s_atl_sum_WH.values
s_atl_aut_WH = s_atl_aut_WH.values


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

s_atl_win_WH = s_atl_win_WH[~np.isnan(s_atl_win_WH)]
s_atl_spr_WH = s_atl_spr_WH[~np.isnan(s_atl_spr_WH)]
s_atl_sum_WH = s_atl_sum_WH[~np.isnan(s_atl_sum_WH)]
s_atl_aut_WH = s_atl_aut_WH[~np.isnan(s_atl_aut_WH)]

#Summary statistics of DJF in the South Atlantic Ocean

print(s_atl_sum_WH.mean())
print(s_atl_sum_WH.std())
print(s_atl_sum_WH.max())
print(s_atl_sum_WH.min())
print(np.quantile(s_atl_sum_WH, 0.95))


#Summary statistics of MAM in the South Atlantic Ocean

print(s_atl_aut_WH.mean())
print(s_atl_aut_WH.std())
print(s_atl_aut_WH.max())
print(s_atl_aut_WH.min())
print(np.quantile(s_atl_aut_WH, 0.95))

#Summary statistics of JJA in the South Atlantic Ocean

print(s_atl_win_WH.mean())
print(s_atl_win_WH.std())
print(s_atl_win_WH.max())
print(s_atl_win_WH.min())
print(np.quantile(s_atl_win_WH, 0.95))


#Summary statistics of SON in the South Atlantic Ocean

print(s_atl_spr_WH.mean())
print(s_atl_spr_WH.std())
print(s_atl_spr_WH.max())
print(s_atl_spr_WH.min())
print(np.quantile(s_atl_spr_WH, 0.95))


#Selected wave_height to all our datasets

ind_ocean_win_WH = ind_ocean_win[wave_height]
ind_ocean_sum_WH = ind_ocean_sum[wave_height]
ind_ocean_mon_WH = ind_ocean_mon[wave_height]
ind_ocean_postmon_WH = ind_ocean_postmon[wave_height]


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

ind_ocean_win_WH = ind_ocean_win_WH.to_array()
ind_ocean_sum_WH = ind_ocean_sum_WH.to_array()
ind_ocean_mon_WH = ind_ocean_mon_WH.to_array()
ind_ocean_postmon_WH = ind_ocean_postmon_WH.to_array()


#Afterwards, we retrieved the values from the arrays. 

ind_ocean_win_WH = ind_ocean_win_WH.values
ind_ocean_sum_WH = ind_ocean_sum_WH.values
ind_ocean_mon_WH = ind_ocean_mon_WH.values
ind_ocean_postmon_WH = ind_ocean_postmon_WH.values



#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

ind_ocean_win_WH = ind_ocean_win_WH[~np.isnan(ind_ocean_win_WH)]
ind_ocean_sum_WH = ind_ocean_sum_WH[~np.isnan(ind_ocean_sum_WH)]
ind_ocean_mon_WH = ind_ocean_mon_WH[~np.isnan(ind_ocean_mon_WH)]
ind_ocean_postmon_WH = ind_ocean_postmon_WH[~np.isnan(ind_ocean_postmon_WH)]


#Summary statistics of DJF in the Indian Ocean

print(ind_ocean_win_WH.mean())
print(ind_ocean_win_WH.std())
print(ind_ocean_win_WH.max())
print(ind_ocean_win_WH.min())
print(np.quantile(ind_ocean_win_WH, 0.95))


#Summary statistics of MAM in the Indian Ocean

print(ind_ocean_sum_WH.mean())
print(ind_ocean_sum_WH.std())
print(ind_ocean_sum_WH.max())
print(ind_ocean_sum_WH.min())
print(np.quantile(ind_ocean_sum_WH, 0.95))


#Summary statistics of JJA in the Indian Ocean

print(ind_ocean_mon_WH.mean())
print(ind_ocean_mon_WH.std())
print(ind_ocean_mon_WH.max())
print(ind_ocean_mon_WH.min())
print(np.quantile(ind_ocean_mon_WH, 0.95))


#Summary statistics of SON in the Indian Ocean

print(ind_ocean_postmon_WH.mean())
print(ind_ocean_postmon_WH.std())
print(ind_ocean_postmon_WH.max())
print(ind_ocean_postmon_WH.min())
print(np.quantile(ind_ocean_postmon_WH, 0.95))


#Started to the same for swell height in the North Atlantic Ocean.

swell_height = ['VHM0_SW1']

n_atl_win_SH = n_atlantic_win[swell_height]
n_atl_spr_SH = n_atlantic_spr[swell_height]
n_atl_sum_SH = n_atlantic_sum[swell_height]
n_atl_aut_SH = n_atlantic_aut[swell_height]


#To be able to extract summary statistics, it is easier to tranform the 
#datasets to either a 2d dataframe using Pandas, or making an array of the variable 
#we need with Numpy. As dataframes only are more spacious, and therefor takes more 
#time to use, when working with such large numbers, we decided to use numpy arryays. 
#Therefore, we made array for all the datasets over the North Atlantic Ocean.

n_atl_win_SH = n_atl_win_SH.to_array()
n_atl_spr_SH = n_atl_spr_SH.to_array()
n_atl_sum_SH = n_atl_sum_SH.to_array()
n_atl_aut_SH = n_atl_aut_SH.to_array()


#Afterwards, we retrieved the values from the arrays. 

n_atl_win_SH = n_atl_win_SH.values
n_atl_spr_SH = n_atl_spr_SH.values
n_atl_sum_SH = n_atl_sum_SH.values
n_atl_aut_SH = n_atl_aut_SH.values


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

n_atl_win_SH = n_atl_win_SH[~np.isnan(n_atl_win_SH)]
n_atl_spr_SH = n_atl_spr_SH[~np.isnan(n_atl_spr_SH)]
n_atl_sum_SH = n_atl_sum_SH[~np.isnan(n_atl_sum_SH)]
n_atl_aut_SH = n_atl_aut_SH[~np.isnan(n_atl_aut_SH)]


#Since numpy array do not have a specific describe() function, similar to pandas,
#we printed out seperate values. We found the mean, standard deviation, 
#maxmimum value, minimum value, and the 95% quantile. The reason for the 
#choice of measures, was that we wanted to look at the avreage values across time, but also 
#if there were any individual weather events worth noticing. Also, the normal range and 
#fluctuations is interesting to know more about. 

#Summary statistics of DJF in the North Atlantic Ocean

print(n_atl_win_SH.mean())
print(n_atl_win_SH.std())
print(n_atl_win_SH.max())
print(n_atl_win_SH.min())
print(np.quantile(n_atl_win_SH, 0.95))


#Summary statistics of MAM in the North Atlantic Ocean

print(n_atl_spr_SH.mean())
print(n_atl_spr_SH.std())
print(n_atl_spr_SH.max())
print(n_atl_spr_SH.min())
print(np.quantile(n_atl_spr_SH, 0.95))


#Summary statistics of JJA in the North Atlantic Ocean

print(n_atl_sum_SH.mean())
print(n_atl_sum_SH.std())
print(n_atl_sum_SH.max())
print(n_atl_sum_SH.min())
print(np.quantile(n_atl_sum_SH, 0.95))


#Summary statistics of SON in the North Atlantic Ocean

print(n_atl_aut_SH.mean())
print(n_atl_aut_SH.std())
print(n_atl_aut_SH.max())
print(n_atl_aut_SH.min())
print(np.quantile(n_atl_aut_SH, 0.95))


#Selected swell_height to all our datasets

s_atl_win_SH = s_atlantic_win[swell_height]
s_atl_spr_SH = s_atlantic_spr[swell_height]
s_atl_sum_SH = s_atlantic_sum[swell_height]
s_atl_aut_SH = s_atlantic_aut[swell_height]


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

s_atl_win_SH = s_atl_win_SH.to_array()
s_atl_spr_SH = s_atl_spr_SH.to_array()
s_atl_sum_SH = s_atl_sum_SH.to_array()
s_atl_aut_SH = s_atl_aut_SH.to_array()


#Afterwards, we retrieved the values from the arrays. 

s_atl_win_SH = s_atl_win_SH.values
s_atl_spr_SH = s_atl_spr_SH.values
s_atl_sum_SH = s_atl_sum_SH.values
s_atl_aut_SH = s_atl_aut_SH.values


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

s_atl_win_SH = s_atl_win_SH[~np.isnan(s_atl_win_SH)]
s_atl_spr_SH = s_atl_spr_SH[~np.isnan(s_atl_spr_SH)]
s_atl_sum_SH = s_atl_sum_SH[~np.isnan(s_atl_sum_SH)]
s_atl_aut_SH = s_atl_aut_SH[~np.isnan(s_atl_aut_SH)]

#Summary statistics of DJF in the South Atlantic Ocean

print(s_atl_sum_SH.mean())
print(s_atl_sum_SH.std())
print(s_atl_sum_SH.max())
print(s_atl_sum_SH.min())
print(np.quantile(s_atl_sum_SH, 0.95))


#Summary statistics of MAM in the South Atlantic Ocean

print(s_atl_aut_SH.mean())
print(s_atl_aut_SH.std())
print(s_atl_aut_SH.max())
print(s_atl_aut_SH.min())
print(np.quantile(s_atl_aut_SH, 0.95))


#Summary statistics of JJA in the South Atlantic Ocean

print(s_atl_win_SH.mean())
print(s_atl_win_SH.std())
print(s_atl_win_SH.max())
print(s_atl_win_SH.min())
print(np.quantile(s_atl_win_SH, 0.95))


#Summary statistics of SON in the South Atlantic Ocean

print(s_atl_spr_SH.mean())
print(s_atl_spr_SH.std())
print(s_atl_spr_SH.max())
print(s_atl_spr_SH.min())
print(np.quantile(s_atl_spr_SH, 0.95))





#Selected swell_height to all our datasets

ind_ocean_win_SH = ind_ocean_win[swell_height]
ind_ocean_sum_SH = ind_ocean_sum[swell_height]
ind_ocean_mon_SH = ind_ocean_mon[swell_height]
ind_ocean_postmon_SH = ind_ocean_postmon[swell_height]


#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

ind_ocean_win_SH = ind_ocean_win_SH.to_array()
ind_ocean_sum_SH = ind_ocean_sum_SH.to_array()
ind_ocean_mon_SH = ind_ocean_mon_SH.to_array()
ind_ocean_postmon_SH = ind_ocean_postmon_SH.to_array()


#Afterwards, we retrieved the values from the arrays. 

ind_ocean_win_SH = ind_ocean_win_SH.values
ind_ocean_sum_SH = ind_ocean_sum_SH.values
ind_ocean_mon_SH = ind_ocean_mon_SH.values
ind_ocean_postmon_SH = ind_ocean_postmon_SH.values



#Then, we had to remove all NaN as it is not possible to retrieve statistics 
#when there is missing values. 

ind_ocean_win_SH = ind_ocean_win_SH[~np.isnan(ind_ocean_win_SH)]
ind_ocean_sum_SH = ind_ocean_sum_SH[~np.isnan(ind_ocean_sum_SH)]
ind_ocean_mon_SH = ind_ocean_mon_SH[~np.isnan(ind_ocean_mon_SH)]
ind_ocean_postmon_SH = ind_ocean_postmon_SH[~np.isnan(ind_ocean_postmon_SH)]


#Summary statistics of DJF in the Indian Ocean

print(ind_ocean_win_SH.mean())
print(ind_ocean_win_SH.std())
print(ind_ocean_win_SH.max())
print(ind_ocean_win_SH.min())
print(np.quantile(ind_ocean_win_SH, 0.95))

#Summary statistics of MAM in the Indian Ocean

print(ind_ocean_sum_SH.mean())
print(ind_ocean_sum_SH.std())
print(ind_ocean_sum_SH.max())
print(ind_ocean_sum_SH.min())
print(np.quantile(ind_ocean_sum_SH, 0.95))


#Summary statistics of JJA in the Indian Ocean

print(ind_ocean_mon_SH.mean())
print(ind_ocean_mon_SH.std())
print(ind_ocean_mon_SH.max())
print(ind_ocean_mon_SH.min())
print(np.quantile(ind_ocean_mon_SH, 0.95))


#Summary statistics of SON in the Indian Ocean

print(ind_ocean_postmon_SH.mean())
print(ind_ocean_postmon_SH.std())
print(ind_ocean_postmon_SH.max())
print(ind_ocean_postmon_SH.min())
print(np.quantile(ind_ocean_postmon_SH, 0.95))


