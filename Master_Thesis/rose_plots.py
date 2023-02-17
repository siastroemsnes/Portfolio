# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:20:43 2022

@authors: Marie Log Staveland & Sia Benedikte Strømsnes

"""

## Generating weather routing distributions
import pandas as pd
import glob
import numpy as np
from matplotlib import pyplot as plt
from windrose import WindroseAxes
import matplotlib.font_manager as font_manager

# Uploading the parquet files for creating wave rose plots
files = glob.glob("/Volumes/LaCie/Master Thesis/merged_files")
df = pd.concat([pd.read_parquet(fp) for fp in files])

# Uploading the parquet files for creating wind rose plots
files_wind = glob.glob("C/Volumes/LaCie/Master Thesis/Codes/Wind_parquet/Wind_parquet-20221212T204127Z-001/Wind_parquet/")
df1 = pd.concat([pd.read_parquet(fp) for fp in files_wind])

#Adding Beaufort scale to the dataframe for wind for use in the rose plots
bf_scale = []
for knots in df1["wind_speed_kts"]:
    if knots < 1:
        bf_scale.append(0)
    elif knots >= 1 and knots < 4:
        bf_scale.append(1)
    elif knots >= 4 and knots < 7:
        bf_scale.append(2)
    elif knots >= 7 and knots < 11:
        bf_scale.append(3)
    elif knots >= 11 and knots < 17:
        bf_scale.append(4)
    elif knots >=17 and knots < 22:
        bf_scale.append(5)
    elif knots >=22 and knots < 28:
        bf_scale.append(6)
    elif knots >=28 and knots < 34:
        bf_scale.append(7)
    elif knots >=34 and knots < 41:
        bf_scale.append(8)
    elif knots >=41 and knots < 48:
        bf_scale.append(9)
    elif knots >=48 and knots < 56:
        bf_scale.append(10)
    elif knots >=56 and knots < 64:
        bf_scale.append(11)
    else:
        bf_scale.append(12)

# Creating a column for the Beaufort Scale
df1["bf_scale"] = bf_scale

####################### Waves #######################
# Splitting into the four periods for wave
df['month'] = df['time_xa'].dt.month
#Creating a column called season in order to get the mode of periods based on what day the voyage start
season=[]
for x in df["month"]:
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
df['season'] = season

### Splitting the dataset into three, one for each route
# to Houston to Rotterdam
df_rotterdam = df[df['full_route'] == "rendezvous-houston-rotterdam"]
# to Samarinda to Pipavav
df_pipapav = df[df['full_route'] == "rendezvous-samarinda-pipapav"]
# to Richards Bay to Mundra
df_mundra = df[df['full_route'] == "rendezvous-richards_bay-mundra"]


####################################################
# Splitting voyages of Route 1 into season routes 
# Creating a new column for mode in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df_rotterdam["voyage_season"] = (df_rotterdam.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Creating a new dataframe of route 1 of months December, January & February
rotterdam_DJF = df_rotterdam[df_rotterdam['voyage_season'] == 1]

# Creating a new dataframe of route 1 of months March, April & May
rotterdam_MAM = df_rotterdam[df_rotterdam['voyage_season'] == 2]

# Creating a new dataframe of route 1 of months June, July & August
rotterdam_JJA = df_rotterdam[df_rotterdam['voyage_season'] == 3]

# Creating a new dataframe of route 1 of months September, October & November
rotterdam_SON = df_rotterdam[df_rotterdam['voyage_season'] == 4]

####################################################
# Splitting voyages of Route B into season routes 
## Creating a new column for mode in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df_pipapav["voyage_season"] = (df_pipapav.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Creating a new dataframe of route 2 of months December, January & February
pipapav_DJF = df_pipapav[df_pipapav['voyage_season'] == 1]

# Creating a new dataframe of route 2 of months March, April & May
pipapav_MAM = df_pipapav[df_pipapav['voyage_season'] == 2]

# Creating a new dataframe of route 2 of months June, July, August & September
pipapav_JJA = df_pipapav[df_pipapav['voyage_season'] == 3]

# Creating a new dataframe of route 2 of months October & November
pipapav_SON = df_pipapav[df_pipapav['voyage_season'] == 4]

####################################################
# Splitting voyages of Route 3 into season routes
# Creating a new column for mode in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df_mundra["voyage_season"] = (df_mundra.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Creating a new dataframe of route 3 of months December, January & February
mundra_DJF = df_mundra[df_mundra['voyage_season'] == 1]

# Creating a new dataframe of route 3 of months March, April & May
mundra_MAM = df_mundra[df_mundra['voyage_season'] == 2]

# Creating a new dataframe of route 3 of months June, July, August & September
mundra_JJA = df_mundra[df_mundra['voyage_season'] == 3]

# Creating a new dataframe of route 3 of months October & November
mundra_SON = df_mundra[df_mundra['voyage_season'] == 4]

####################### Wind #######################
# Splitting into the four periods for wind
df1['month'] = df1['time_xa'].dt.month
#Creating a column called season in order to get the mode of periods based on what day the voyage start
season=[]
for x in df1["month"]:
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
df1['season'] = season

### Splitting the dataset into three, one for each route
# to Houston to Rotterdam
df1_rotterdam = df1[df1['full_route'] == "rendezvous-houston-rotterdam"]
# to Samarinda to Pipavav
df1_pipapav = df1[df1['full_route'] == "rendezvous-samarinda-pipapav"]
# to Richards Bay to Mundra
df1_mundra = df1[df1['full_route'] == "rendezvous-richards_bay-mundra"]

####################################################
# Splitting voyages of Route 1 into season routes
# Creating a new column for mode in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df1_rotterdam["voyage_season"] = (df1_rotterdam.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Creating a new dataframe of route 1 of months December, January & February
rotterdam_DJF_wind = df1_rotterdam[df1_rotterdam['voyage_season'] == 1]

# Creating a new dataframe of route 1 of months March, April & May
rotterdam_MAM_wind = df1_rotterdam[df1_rotterdam['voyage_season'] == 2]

# Creating a new dataframe of route 1 of months June, July & August
rotterdam_JJA_wind = df1_rotterdam[df1_rotterdam['voyage_season'] == 3]

# Creating a new dataframe of route 1 of months September, October & November
rotterdam_SON_wind = df1_rotterdam[df1_rotterdam['voyage_season'] == 4]

####################################################
# Splitting voyages of Route 2 into season routes 
## Creating a new column for mode in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df1_pipapav["voyage_season"] = (df1_pipapav.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Creating a new dataframe of route 2 of months December, January & February
pipapav_DJF_wind = df1_pipapav[df1_pipapav['voyage_season'] == 1]

# Creating a new dataframe of route 2 of months March, April & May
pipapav_MAM_wind = df1_pipapav[df1_pipapav['voyage_season'] == 2]

# Creating a new dataframe of route 2 of months June, July, August & September
pipapav_JJA_wind = df1_pipapav[df1_pipapav['voyage_season'] == 3]

# Creating a new dataframe of route 2 of months October & November
pipapav_SON_wind = df1_pipapav[df1_pipapav['voyage_season'] == 4]

####################################################
# Splitting voyages of Route 3 into season routes
# Creating a new column for mode in order to split into seasons depending on starting date, for which season the most part of the voyage is in
df1_mundra["voyage_season"] = (df1_mundra.groupby('starting_date')['season'].transform(lambda x: x.value_counts().index[0]))

# Creating a new dataframe of route 3 of months December, January & February
mundra_DJF_wind = df1_mundra[df1_mundra['voyage_season'] == 1]

# Creating a new dataframe of route 3 of months March, April & May
mundra_MAM_wind = df1_mundra[df1_mundra['voyage_season'] == 2]

# Creating a new dataframe of route 3 of months June, July, August & September
mundra_JJA_wind = df1_mundra[df1_mundra['voyage_season'] == 3]

# Creating a new dataframe of route 3 of months October & November
mundra_SON_wind = df1_mundra[df1_mundra['voyage_season'] == 4]


###################### Wave PLOTS Route 1 ##############################

# Making wave rose plots of waves for Route 1 for period DJF
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_DJF.vmdr,rotterdam_DJF.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 1, DJF',fontsize=20)
plt.savefig('wh_dir_rot_DJF', dpi=300)

# Making wave rose plots of waves for Route 1 for period MAM
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_MAM.vmdr,rotterdam_MAM.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 1, MAM',fontsize=20)
plt.savefig('wh_dir_rot_MAM', dpi=300)

# Making wave rose plots of waves for Route 1 for period JJA
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_JJA.vmdr,rotterdam_JJA.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 1, JJA',fontsize=20)
plt.savefig('wh_dir_rot_JJA', dpi=300)

# Making wave rose plots of waves for Route 1 for period SON
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_SON.vmdr,rotterdam_SON.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 1, SON',fontsize=20)
plt.savefig('wh_dir_rot_SON', dpi=300)


###################### Wind PLOTS Route 1 ##############################

# Making wind rose plots of wind for Route 1 for period DJF
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_DJF_wind.wind_dir,rotterdam_DJF_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 1, DJF, Beaufort scale', fontsize=20)
plt.savefig('windr_rot_DJF', dpi=300)

# Making wind rose plots of wind for Route 1 for period MAM
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_MAM_wind.wind_dir,rotterdam_MAM_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font, bbox_to_anchor=(0, 0))
plt.title('Wind speed & direction for Route 1, MAM, Beaufort scale', fontsize=20)
plt.savefig('windr_rot_MAM', dpi=300)

# Making wind rose plots of wind for Route 1 for period JJA
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_JJA_wind.wind_dir,rotterdam_JJA_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 1, JJA, Beaufort scale', fontsize=20)
plt.savefig('windr_rot_JJA', dpi=300)

# Making wind rose plots of wind for Route 1 for period SON
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(rotterdam_SON_wind.wind_dir,rotterdam_SON_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 1, SON, Beaufort scale', fontsize=20)
plt.savefig('windr_rot_SON', dpi=300)

###################### Wave PLOTS Route 2 ##############################

# Making wave rose plots of waves for Route 2 for period DJF
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_DJF.vmdr,pipapav_DJF.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 2, DJF',fontsize=20)
plt.savefig('wh_dir_pip_DJF', dpi=300)

# Making wave rose plots of waves for Route 2 for period MAM
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_MAM.vmdr,pipapav_MAM.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 2, MAM',fontsize=20)
plt.savefig('wh_dir_pip_MAM', dpi=300)

# Making wave rose plots of waves for Route 2 for period JJA
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_JJA.vmdr,pipapav_JJA.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 2, JJA',fontsize=20)
plt.savefig('wh_dir_pip_JJA', dpi=300)

# Making wave rose plots of waves for Route 2 for period SON
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_SON.vmdr,pipapav_SON.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 2, SON',fontsize=20)
plt.savefig('wh_dir_pip_SON', dpi=300)

###################### Wind PLOTS Route 2 ##############################

# Making wind rose plots of wind for Route 2 for period DJF
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_DJF_wind.wind_dir,pipapav_DJF_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 2, DJF, Beaufort scale',fontsize=20)
plt.savefig('windr_pip_DJF', dpi=300)

# Making wind rose plots of wind for Route 2 for period MAM
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_MAM_wind.wind_dir,pipapav_MAM_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 2, MAM, Beaufort scale',fontsize=20)
plt.savefig('windr_pip_MAM', dpi=300)

# Making wind rose plots of wind for Route 2 for period JJA
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_JJA_wind.wind_dir,pipapav_JJA_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 2, JJA, Beaufort scale',fontsize=20)
plt.savefig('windr_pip_JJA', dpi=300)

# Making wind rose plots of wind for Route 2 for period SON
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(pipapav_SON_wind.wind_dir,pipapav_SON_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 2, SON, Beaufort scale',fontsize=20)
plt.savefig('windr_pip_SON', dpi=300)

###################### Wave PLOTS Route 3 ##############################

# Making wave rose plots of waves for Route 3 for period DJF
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(mundra_DJF.vmdr,mundra_DJF.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 3, DJF',fontsize=20)
plt.savefig('wh_dir_mun_DJF', dpi=300)

# Making wave rose plots of waves for Route 3 for period MAM
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(mundra_MAM.vmdr,mundra_MAM.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 3, MAM',fontsize=20)
plt.savefig('wh_dir_mun_MAM', dpi=300)

# Making wave rose plots of waves for Route 3 for period JJA
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(mundra_JJA.vmdr,mundra_JJA.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font, loc = 'lower right')
plt.title('Wave height & direction for Route 3, JJA',fontsize=20)
plt.savefig('wh_dir_mun_JJA', dpi=300)

# Making wave rose plots of waves for Route 3 for period SON
bins_range = np.arange(0,6,1)
ax = WindroseAxes.from_ax()
ax.bar(mundra_SON.vmdr,mundra_SON.vhm0, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wave height & direction for Route 3, SON',fontsize=20)
plt.savefig('wh_dir_mun_SON', dpi=300)

###################### Wind PLOTS Route 3 ##############################
# Making wind rose plots of wind for Route 3 for period DJF
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(mundra_DJF_wind.wind_dir,mundra_DJF_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 3, DJF, Beaufort scale',fontsize=20)
plt.savefig('windr_mun_DJF', dpi=300)

# Making wind rose plots of wind for Route 3 for period MAM
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(mundra_MAM_wind.wind_dir,mundra_MAM_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 3, MAM, Beaufort scale',fontsize=20)
plt.savefig('windr_mun_MAM', dpi=300)

# Making wind rose plots of wind for Route 3 for period JJA
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(mundra_JJA_wind.wind_dir,mundra_JJA_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 3, JJA, Beaufort scale',fontsize=20)
plt.savefig('windr_mun_JJA', dpi=300)

# Making wind rose plots of wind for Route 3 for period SON
bins_range = np.arange(0,12,2)
ax = WindroseAxes.from_ax()
ax.bar(mundra_SON_wind.wind_dir,mundra_SON_wind.bf_scale, normed=True,bins=bins_range)
plt.xticks(size = 15)
plt.yticks(size = 15)
font = font_manager.FontProperties(weight='bold',
                                   size=18)
ax.set_legend(fontsize=20, prop=font)
plt.title('Wind speed & direction for Route 3, SON, Beaufort scale',fontsize=20)
plt.savefig('windr_mun_SON', dpi=300)

