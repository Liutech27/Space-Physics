# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:57:55 2020

@author: markl
"""
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd 
import math 
from scipy import signal
import seaborn as sns


data3d=np.zeros((200,2,1000))
data2d=np.zeros((200,2))

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
picture_filename = 'p2.png'
textfile_name_input = "*2010-11*dawn*.txt"

os.chdir((path_string + database_name))

filelist1=glob.glob(textfile_name_input)
filelist1 = np.array(filelist1)
filelist1size=filelist1.size
for i in range(filelist1size):
    data0 = np.genfromtxt(filelist1[i])

    height = data0[:,0]
    height = height[:] - 6371 
    on2_den = data0[:,1]
    
sns.lineplot(x=on2_den,y=height)