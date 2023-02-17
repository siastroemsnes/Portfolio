# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:35:18 2022

@author: Marie Log Staveland & Sia Benedikte Strømsnes

"""
import joblib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Distributions_Route_1 import rotterdam_DJF_voy_fuel 
from Distributions_Route_1 import rotterdam_MAM_voy_fuel 
from Distributions_Route_1 import rotterdam_JJA_voy_fuel 
from Distributions_Route_1 import rotterdam_SON_voy_fuel 

from Distributions_Route_2 import pipapav_DJF_voy_fuel
from Distributions_Route_2 import pipapav_MAM_voy_fuel
from Distributions_Route_2 import pipapav_JJA_voy_fuel
from Distributions_Route_2 import pipapav_SON_voy_fuel

from Distributions_Route_3 import mundra_DJF_voy_fuel
from Distributions_Route_3 import mundra_MAM_voy_fuel
from Distributions_Route_3 import mundra_JJA_voy_fuel
from Distributions_Route_3 import mundra_SON_voy_fuel

# Using the total voyage days to get information on total fuel consumption per voyage, averaged per voyage day
days_route1=46.07
days_route2=35.28
days_route3=19.53

# Plot of fuel consumption comparing the three routes on average fuel consumption per day for the period DJF
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_DJF_voy_fuel["fuel_consumption"]/days_route1), label='Route 1', ax=ax)
sns.kdeplot((pipapav_DJF_voy_fuel["fuel_consumption"]/days_route2), label='Route 2', ax=ax)
sns.kdeplot((mundra_DJF_voy_fuel["fuel_consumption"]/days_route3), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage day",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - DJF", fontsize=12)
plt.savefig('fc_djf.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on average fuel consumption per day for the period MAM
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_MAM_voy_fuel["fuel_consumption"]/days_route1), label='Route 1', ax=ax)
sns.kdeplot((pipapav_MAM_voy_fuel["fuel_consumption"]/days_route2), label='Route 2', ax=ax)
sns.kdeplot((mundra_MAM_voy_fuel["fuel_consumption"]/days_route3), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage day",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - MAM", fontsize=12)
plt.savefig('fc_mam.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on average fuel consumption per day for the period JJA
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_JJA_voy_fuel["fuel_consumption"]/days_route1), label='Route 1', ax=ax)
sns.kdeplot((pipapav_JJA_voy_fuel["fuel_consumption"]/days_route2), label='Route 2', ax=ax)
sns.kdeplot((mundra_JJA_voy_fuel["fuel_consumption"]/days_route3), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage day",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - JJA", fontsize=12)
plt.savefig('fc_jja.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on average fuel consumption per day for the period SON
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_SON_voy_fuel["fuel_consumption"]/days_route1), label='Route 1', ax=ax)
sns.kdeplot((pipapav_SON_voy_fuel["fuel_consumption"]/days_route2), label='Route 2', ax=ax)
sns.kdeplot((mundra_SON_voy_fuel["fuel_consumption"]/days_route3), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons/voyage day",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - SON", fontsize=12)
plt.savefig('fc_son.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on total fuel consumption for the period DJF
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_DJF_voy_fuel["fuel_consumption"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_DJF_voy_fuel["fuel_consumption"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_DJF_voy_fuel["fuel_consumption"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - DJF", fontsize=12)
plt.savefig('fc_djf_total.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on total fuel consumption for the period MAM
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_MAM_voy_fuel["fuel_consumption"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_MAM_voy_fuel["fuel_consumption"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_MAM_voy_fuel["fuel_consumption"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - MAM", fontsize=12)
plt.savefig('fc_mam_total.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on total fuel consumption for the period JJA
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_JJA_voy_fuel["fuel_consumption"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_JJA_voy_fuel["fuel_consumption"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_JJA_voy_fuel["fuel_consumption"]), label='Route 3', ax=ax)
ax.legend(fontsize=10,loc = 'upper right')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - JJA", fontsize=12)
plt.savefig('fc_jja_total.png', dpi = 300)
sns.plt.show()

# Plot of fuel consumption comparing the three routes on total fuel consumption for the period SON
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_SON_voy_fuel["fuel_consumption"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_SON_voy_fuel["fuel_consumption"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_SON_voy_fuel["fuel_consumption"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("Fuel consumption in tons",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("Fuel Consumption - SON", fontsize=12)
plt.savefig('fc_son_total.png', dpi = 300)
sns.plt.show()
