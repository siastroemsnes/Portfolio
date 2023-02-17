# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 00:22:24 2022

@authors: Marie Log Staveland & Sia Benedikte Strømsnes

"""
# TCE
import joblib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

######################### Comparing of TCE #########################
from Distributions_Route_1 import rotterdam_DJF_case 
from Distributions_Route_1 import rotterdam_MAM_case 
from Distributions_Route_1 import rotterdam_JJA_case 
from Distributions_Route_1 import rotterdam_SON_case 

from Distributions_Route_2 import pipapav_DJF_case
from Distributions_Route_2 import pipapav_MAM_case
from Distributions_Route_2 import pipapav_JJA_case
from Distributions_Route_2 import pipapav_SON_case

from Distributions_Route_3 import mundra_DJF_case
from Distributions_Route_3 import mundra_MAM_case
from Distributions_Route_3 import mundra_JJA_case
from Distributions_Route_3 import mundra_SON_case

# Using joblib to get TCE case dataframes for comparing the routes
pipapav_DJF_case = joblib.load('pipapav_DJF_case.joblib')
pipapav_MAM_case = joblib.load('pipapav_MAM_case.joblib')
pipapav_JJA_case = joblib.load('pipapav_JJA_case.joblib')
pipapav_SON_case = joblib.load('pipapav_SON_case.joblib')

rotterdam_DJF_case = joblib.load('rotterdam_DJF_case.joblib')
rotterdam_MAM_case = joblib.load('rotterdam_MAM_case.joblib')
rotterdam_JJA_case = joblib.load('rotterdam_JJA_case.joblib')
rotterdam_SON_case = joblib.load('rotterdam_SON_case.joblib')

mundra_DJF_case = joblib.load('mundra_DJF_case.joblib')
mundra_MAM_case = joblib.load('mundra_MAM_case.joblib')
mundra_JJA_case = joblib.load('mundra_JJA_case.joblib')
mundra_SON_case = joblib.load('mundra_SON_case.joblib')

################# Plot of TCE distributions to compare the three rotues ####################

# Plot of TCE scenario 1 for the period DJF to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(rotterdam_DJF_case["TCE_DJF_c1"], label='Route 1', ax=ax)
sns.kdeplot(pipapav_DJF_case["TCE_DJF_c1"], label='Route 2', ax=ax)
sns.kdeplot(mundra_DJF_case["TCE_DJF_c1"], label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - DJF - Scenario 1", fontsize=12)
plt.savefig('TCE_c1_djf_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 2 for the period DJF to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_DJF_case["TCE_DJF_c2"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_DJF_case["TCE_DJF_c2"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_DJF_case["TCE_DJF_c2"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - DJF - Scenario 2", fontsize=12)
plt.savefig('TCE_c2_djf_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 3 for the period DJF to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_DJF_case["TCE_DJF_c3"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_DJF_case["TCE_DJF_c3"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_DJF_case["TCE_DJF_c3"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - DJF - Scenario 3", fontsize=12)
plt.savefig('TCE_c3_djf_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 1 for the period MAM to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_MAM_case["TCE_MAM_c1"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_MAM_case["TCE_MAM_c1"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_MAM_case["TCE_MAM_c1"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - MAM - Scenario 1", fontsize=12)
plt.savefig('TCE_c1_mam_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 2 for the period MAM to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_MAM_case["TCE_MAM_c2"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_MAM_case["TCE_MAM_c2"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_MAM_case["TCE_MAM_c2"]), label='Route 3', ax=ax)
ax.legend(fontsize=10, loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - MAM - Scenario 2", fontsize=12)
plt.savefig('TCE_c2_mam_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 3 for the period MAM to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_MAM_case["TCE_MAM_c3"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_MAM_case["TCE_MAM_c3"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_MAM_case["TCE_MAM_c3"]), label='Route 3', ax=ax)
ax.legend(fontsize=10, loc= 'upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - MAM - Scenario 3", fontsize=12)
plt.savefig('TCE_c3_mam_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 1 for the period JJA to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_JJA_case["TCE_JJA_c1"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_JJA_case["TCE_JJA_c1"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_JJA_case["TCE_JJA_c1"]), label='Route 3', ax=ax)
ax.legend(fontsize=10, loc='upper right')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - JJA - Scenario 1", fontsize=12)
plt.savefig('TCE_c1_jja_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 2 for the period JJA to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_JJA_case["TCE_JJA_c2"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_JJA_case["TCE_JJA_c2"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_JJA_case["TCE_JJA_c2"]), label='Route 3', ax=ax)
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - JJA - Scenario 2", fontsize=12)
plt.savefig('TCE_c2_jja_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 3 for the period JJA to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_JJA_case["TCE_JJA_c3"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_JJA_case["TCE_JJA_c3"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_JJA_case["TCE_JJA_c3"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - JJA - Scenario 3", fontsize=12)
plt.savefig('TCE_c3_jja_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 1 for the period SON to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_SON_case["TCE_SON_c1"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_SON_case["TCE_SON_c1"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_SON_case["TCE_SON_c1"]), label='Route 3', ax=ax)
ax.legend(fontsize=10, loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - SON - Scenario 1", fontsize=12)
plt.savefig('TCE_c1_son_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 2 for the period SON to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_SON_case["TCE_SON_c2"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_SON_case["TCE_SON_c2"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_SON_case["TCE_SON_c2"]), label='Route 3', ax=ax)
ax.legend(fontsize=10)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - SON - Scenario 2", fontsize=12)
plt.savefig('TCE_c2_son_50.png', dpi = 300)
sns.plt.show()

# Plot of TCE scenario 3 for the period SON to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot((rotterdam_SON_case["TCE_SON_c3"]), label='Route 1', ax=ax)
sns.kdeplot((pipapav_SON_case["TCE_SON_c3"]), label='Route 2', ax=ax)
sns.kdeplot((mundra_SON_case["TCE_SON_c3"]), label='Route 3', ax=ax)
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Density",fontsize=12)
plt.title("TCE - SON - Scenario 3", fontsize=12)
plt.savefig('TCE_c3_son_50.png', dpi = 300)
sns.plt.show()

################# Cumulative plot of TCE distributions to compare the three rotues ####################

# Cumulative plot of TCE scenario 1 for the period DJF to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_DJF_case, x="TCE_DJF_c1", label='Route 1')
sns.ecdfplot(data=pipapav_DJF_case, x="TCE_DJF_c1", label='Route 2')
sns.ecdfplot(data=mundra_DJF_case, x="TCE_DJF_c1", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - DJF - Scenario 1", fontsize=12)
plt.savefig('cumul_c1_djf_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 2 for the period DJF to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_DJF_case, x="TCE_DJF_c2", label='Route 1')
sns.ecdfplot(data=pipapav_DJF_case, x="TCE_DJF_c2", label='Route 2')
sns.ecdfplot(data=mundra_DJF_case, x="TCE_DJF_c2", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - DJF - Scenario 2", fontsize=12)
plt.savefig('cumul_c2_djf_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 3 for the period DJF to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_DJF_case, x="TCE_DJF_c3", label='Route 1')
sns.ecdfplot(data=pipapav_DJF_case, x="TCE_DJF_c3", label='Route 2')
sns.ecdfplot(data=mundra_DJF_case, x="TCE_DJF_c3", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - DJF - Scenario 3", fontsize=12)
plt.savefig('cumul_c3_djf_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 1 for the period MAM to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_MAM_case, x="TCE_MAM_c1", label='Route 1')
sns.ecdfplot(data=pipapav_MAM_case, x="TCE_MAM_c1", label='Route 2')
sns.ecdfplot(data=mundra_MAM_case, x="TCE_MAM_c1", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - MAM - Scenario 1", fontsize=12)
plt.savefig('cumul_c1_mam_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 2 for the period MAM to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_MAM_case, x="TCE_MAM_c2", label='Route 1')
sns.ecdfplot(data=pipapav_MAM_case, x="TCE_MAM_c2", label='Route 2')
sns.ecdfplot(data=mundra_MAM_case, x="TCE_MAM_c2", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - MAM - Scenario 2", fontsize=12)
plt.savefig('cumul_c2_mam_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 3 for the period MAM to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_MAM_case, x="TCE_MAM_c3", label='Route 1')
sns.ecdfplot(data=pipapav_MAM_case, x="TCE_MAM_c3", label='Route 2')
sns.ecdfplot(data=mundra_MAM_case, x="TCE_MAM_c3", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - MAM - Scenario 3", fontsize=12)
plt.savefig('cumul_c3_mam_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 1 for the period JJA to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_JJA_case, x="TCE_JJA_c1", label='Route 1')
sns.ecdfplot(data=pipapav_JJA_case, x="TCE_JJA_c1", label='Route 2')
sns.ecdfplot(data=mundra_JJA_case, x="TCE_JJA_c1", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - JJA - Scenario 1", fontsize=12)
plt.savefig('cumul_c1_jja_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 2 for the period JJA to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_JJA_case, x="TCE_JJA_c2", label='Route 1')
sns.ecdfplot(data=pipapav_JJA_case, x="TCE_JJA_c2", label='Route 2')
sns.ecdfplot(data=mundra_JJA_case, x="TCE_JJA_c2", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - JJA - Scenario 2", fontsize=12)
plt.savefig('cumul_c2_jja_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 3 for the period JJA to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_JJA_case, x="TCE_JJA_c3", label='Route 1')
sns.ecdfplot(data=pipapav_JJA_case, x="TCE_JJA_c3", label='Route 2')
sns.ecdfplot(data=mundra_JJA_case, x="TCE_JJA_c3", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - JJA - Scenario 3", fontsize=12)
plt.savefig('cumul_c3_jja_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 1 for the period SON to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_SON_case, x="TCE_SON_c1", label='Route 1')
sns.ecdfplot(data=pipapav_SON_case, x="TCE_SON_c1", label='Route 2')
sns.ecdfplot(data=mundra_SON_case, x="TCE_SON_c1", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - SON - Scenario 1", fontsize=12)
plt.savefig('cumul_c1_son_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 2 for the period SON to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_SON_case, x="TCE_SON_c2", label='Route 1')
sns.ecdfplot(data=pipapav_SON_case, x="TCE_SON_c2", label='Route 2')
sns.ecdfplot(data=mundra_SON_case, x="TCE_SON_c2", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - SON - Scenario 2", fontsize=12)
plt.savefig('cumul_c2_son_50.png', dpi = 300)
sns.plt.show()

# Cumulative plot of TCE scenario 3 for the period SON to compare the three routes
fig, ax = plt.subplots(figsize=(10, 4))
sns.ecdfplot(data=rotterdam_SON_case, x="TCE_SON_c3", label='Route 1')
sns.ecdfplot(data=pipapav_SON_case, x="TCE_SON_c3", label='Route 2')
sns.ecdfplot(data=mundra_SON_case, x="TCE_SON_c3", label='Route 3')
ax.legend(fontsize=10,loc='upper center')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.xlabel("TCE ($/day)",fontsize=10)
plt.ylabel("Probability",fontsize=12)
plt.title("TCE - SON - Scenario 3", fontsize=12)
plt.savefig('cumul_c3_son_50.png', dpi = 300)
sns.plt.show()


