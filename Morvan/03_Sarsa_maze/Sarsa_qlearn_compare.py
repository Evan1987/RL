# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:09:37 2019

@author: Cigar
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _utils import u_constant
path = u_constant.PATH_ROOT + "Code projects/Python/RL_Learn/Morvan/"

sarsa = pd.read_excel(path + "03_Sarsa_maze/sarsa_log.xlsx")
qlearn = pd.read_excel(path + "02_Q_Learning_maze/qlearn_log.xlsx")

fig, axe = plt.subplots(figsize=(10, 10))
sns.distplot(sarsa["total_step"].values[20:], ax=axe, color="red")
sns.distplot(qlearn["total_step"].values[20:], ax=axe, color="blue")
fig.savefig(path + "qlearn_sarsa_compare.png", dpi=100)
