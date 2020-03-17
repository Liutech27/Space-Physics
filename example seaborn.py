# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:29:28 2020

@author: markl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
sns.relplot(x = "total_bill", y ="tip", data =tips)

print(tips)