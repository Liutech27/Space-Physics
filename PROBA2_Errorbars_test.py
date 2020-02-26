# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:43:36 2019

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

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
picture_filename = 'p2.png'
textfile_name_input = "*2014-11*dawn*.txt"

os.chdir((path_string + database_name))
#for file in glob.glob("*.txt"):
#    print(file)
    
filelist1=glob.glob(textfile_name_input)
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
plt.xlabel('O+N2 Density(cm^-3)', fontsize =18)
plt.ylabel('Height (km)', fontsize =18)   

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)
    ax1 = plt.plot(yb, xb, 'k--', linewidth = 2)
    
    z.append(zb)

Z= []
Z1 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z1.append(np.std(buff2)*1)
    
    
ax2 = plt.plot(Z, Heightideal,'r', linewidth =4)
X = np.array(Z,dtype=float)
Y = np.array(Heightideal,dtype=float)
x = np.asanyarray(X)
y = np.asanyarray(Y)
xerror = Z1
yerror = 0
ax3 = plt.errorbar(x,y, xerr=xerror, yerr=yerror, errorevery=1,markeredgewidth=4,linewidth=3)
plt.draw()
os.chdir(path_string) 
plt.savefig(picture_filename)