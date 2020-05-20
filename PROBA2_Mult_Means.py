# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:36:38 2020

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
picture_filename = 'exp'
textfile_name_input = "*2010-11*dawn*.txt"
path_save = "C:/Users/markl/Desktop/"

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

plt.figure(figsize=(10,10))
plt.title('Height vs Density', fontsize =22) 
plt.xscale('log')
plt.xlabel('O+N2 Density(cm^-3)', fontsize =18)
plt.ylabel('Height (km)', fontsize =18)   

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe1 = Z1
    
ax1 = plt.plot(Z, Heightideal,'r', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2010-12*dawn*.txt"

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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe = Z1
    
ax2 = plt.plot(Z, Heightideal,'b', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2011-01*dawn*.txt"

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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe = Z1
    
ax3 = plt.plot(Z, Heightideal,'g', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2011-11*dawn*.txt"

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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe = Z1
    
ax4 = plt.plot(Z, Heightideal,'c', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2011-12*dawn*.txt"

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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe = Z1
    
ax5 = plt.plot(Z, Heightideal,'m', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2012-01*dawn*.txt"

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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe = Z1
    
ax6 = plt.plot(Z, Heightideal,'y', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2012-02*dawn*.txt"

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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
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
    Z1.append(np.std(buff2)*1.17)
    Xe = Z1
    
ax7 = plt.plot(Z, Heightideal,'tab:blue', linewidth =2)

plt.draw()
os.chdir(path_save) 
plt.savefig(picture_filename)