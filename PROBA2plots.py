# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:24:41 2019

@author: Liu
"""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd 
import math 
from scipy import signal 


data3d=np.zeros((200,2,1000))
data2d=np.zeros((200,2))

os.chdir(r'C:/Users/markl/Desktop/PROBA2LYRA/')
#for file in glob.glob("*.txt"):
#    print(file)
    
filelist1=glob.glob("*2011-01*dawn*.txt")
filelist1 = np.array(filelist1)
#print(a[1])
filelist1size=filelist1.size
maxh=np.zeros(filelist1size)
minh=np.zeros(filelist1size)

print(maxh)

for i in range(filelist1size):
#    print(i)
#    print(filelist1[i])
    data2d[:,:]=0.0
    data0 = np.genfromtxt(filelist1[i])

    height = data0[:,0]
    height = height[:] - 6371 
    ma = np.max(height[:])
    mi = np.min(height[:])
    maxh[i] = ma
    minh[i] = mi
    
    #Heightideal = range(long(math.ceil(mi)), long(math.floor(ma)), 3)
    
    on2_den = data0[:,1]
    sys_error = data0[:,2]
    rand_error = data0[:,3]
    lon = data0[:,4]
    lat = data0[:,5]
    size1=height.size
    data2d[0:size1,0]=height
    data2d[0:size1,1]=on2_den
    #data2d[data2d == 0] = 'nan'
    data3d[:,:,i]=data2d
    data3d[data3d == 0] = 'nan' 
    #if i==1:break
    height = data3d[:,0]
    
Heightideal = range(int(math.ceil(np.max(minh))), int(math.floor(np.min(maxh))), 3)
row = np.size(Heightideal,0)
InterpON2=np.zeros((row,1000))

plt.figure(figsize=(7,7))
plt.title('Height vs Density', fontsize =22) 
plt.xscale('log')
plt.xlabel('O-N2 Density(cm^-3)', fontsize =18)
plt.ylabel('Height (km)', fontsize =18)   

x_sum = np.zeros(size1)
y_sum = np.zeros(size1)
for j in range(filelist1size):
    for i in range(size1):
        x_sum[i] += data3d[i,0,j]
        y_sum[i] += data3d[i,1,j]
    ax1 = plt.plot(data3d[:,1,j], data3d[:,0,j], 'k--', linewidth = 2)

x_avg = [x/filelist1size for x in x_sum]
y_avg = [y/filelist1size for y in y_sum]
z=np.interp(Heightideal,x_avg,y_avg)
InterpON2[:,j]=z
    
#height = height[:]  
#on2_den = data3d[:,1]
#Heightideal = np.float64(Heightideal)
#row = np.size(Heightideal,0)
#col = filelist1size
#h = np.zeros((row,col))
#for i in range(col):
#    x = on2_den[:,i][0:row]
#    y = height[:,i][0:row]
#    inter = np.interp(Heightideal,y,x)
#    h[:,i] = inter
#heightmean = np.zeros((row,1))
#heightmedian = np.zeros((row,1))
#densitymean = np.zeros((row,1))
#densitymedian = np.zeros((row,1))
#for i in range(row):
#    heightmean[i] = np.mean(height[i][0:filelist1size])
#    heightmedian[i] = np.median(height[i][0:filelist1size])
#    densitymean[i] = np.mean(on2_den[i][0:filelist1size])
#    densitymedian[i] = np.median(on2_den[i][0:filelist1size])

    #ax1 = plt.plot(on2_den,height, 'k--', linewidth =2)
ax2 = plt.plot(z, Heightideal,'r--', linewidth =2)
#
#ax2 = plt.plot(densitymean,heightmean, 'r--', linewidth = 5)
#ax3 = plt.plot(densitymedian,heightmedian, 'b--', linewidth = 5)

#os.chdir(r'C:/Users/Liu/Desktop/') 
#plt.savefig('2011dawn.png')

