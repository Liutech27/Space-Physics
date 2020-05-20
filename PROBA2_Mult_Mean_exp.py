# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:30:32 2020

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
    
ax1 = plt.plot(Z, Heightideal,'--', linewidth =2)

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
Z2 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z2.append(np.std(buff2)*1.17)
    Xe2 = Z2
    
ax2 = plt.plot(Z, Heightideal,'--', linewidth =2)

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
Z3 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z3.append(np.std(buff2)*1.17)
    Xe3 = Z3
    
ax3 = plt.plot(Z, Heightideal,'--', linewidth =2)

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
Z4 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z4.append(np.std(buff2)*1.17)
    Xe4 = Z4
    
ax4 = plt.plot(Z, Heightideal,'--', linewidth =2)

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
Z5 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z5.append(np.std(buff2)*1.17)
    Xe5 = Z5
    
ax5 = plt.plot(Z, Heightideal,'--', linewidth =2)

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
Z6 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z6.append(np.std(buff2)*1.17)
    Xe6 = Z6
    
ax6 = plt.plot(Z, Heightideal,'--', linewidth =2)

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
Z7 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z7.append(np.std(buff2)*1.17)
    Xe7 = Z7
    
ax7 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2012-12*dawn*.txt"

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
Z8 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z8.append(np.std(buff2)*1.17)
    Xe8 = Z8
    
ax8 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2013-01*dawn*.txt"

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
Z9 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z9.append(np.std(buff2)*1.17)
    Xe9 = Z9
    
ax9 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2013-02*dawn*.txt"

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
Z10 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z10.append(np.std(buff2)*1.17)
    Xe10 = Z10
    
ax10 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2013-11*dawn*.txt"

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
Z11 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z11.append(np.std(buff2)*1.17)
    Xe11 = Z11
    
ax11 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2013-12*dawn*.txt"

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
Z12 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z12.append(np.std(buff2)*1.17)
    Xe12 = Z12
    
ax12 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2014-01*dawn*.txt"

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
Z13 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z13.append(np.std(buff2)*1.17)
    Xe13 = Z13
    
ax13 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2014-02*dawn*.txt"

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
Z14 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z14.append(np.std(buff2)*1.17)
    Xe14 = Z14
    
ax14 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
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

z = []
for j in range(filelist1size):
    yb = data3d[:,1,j]
    xb = data3d[:,0,j]
    zb = np.interp(Heightideal,xb,yb)   
    z.append(zb)

Z= []
Z15 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z15.append(np.std(buff2)*1.17)
    Xe15 = Z15
    
ax15 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2014-12*dawn*.txt"

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
Z16 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z16.append(np.std(buff2)*1.17)
    Xe16 = Z16
    
ax16 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2015-01*dawn*.txt"

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
Z17 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z17.append(np.std(buff2)*1.17)
    Xe17 = Z17
    
ax17 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2015-11*dawn*.txt"

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
Z18 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z18.append(np.std(buff2)*1.17)
    Xe18 = Z18
    
ax18 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2015-12*dawn*.txt"

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
Z19 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z19.append(np.std(buff2)*1.17)
    Xe19 = Z19
    
ax19 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2016-01*dawn*.txt"

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
Z20 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z20.append(np.std(buff2)*1.17)
    Xe20 = Z20
    
ax20 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2016-12*dawn*.txt"

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
Z21 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z21.append(np.std(buff2)*1.17)
    Xe21 = Z21
    
ax21 = plt.plot(Z, Heightideal,'--', linewidth =2)

path_string = "C:/Users/markl/Desktop/"
database_name = 'PROBA2LYRA/'
textfile_name_input = "*2017-01*dawn*.txt"

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
Z22 = []
for j in range(len(z[0])):
    buff = 0
    buff2 = np.empty(len(z))
    for i in range(len(z)):
        buff += z[i][j]
        buff2[i]=z[i][j]
    Z.append(buff/len(z))
    Z22.append(np.std(buff2)*1.17)
    Xe22 = Z22
    
ax22 = plt.plot(Z, Heightideal,'--', linewidth =2)

Xetotal = [Xe1, Xe2, Xe3, Xe4, Xe5, Xe6, Xe7, Xe8, Xe9, Xe10, Xe11, Xe12, Xe13, Xe14, Xe15, Xe16, Xe17, Xe18, Xe19, Xe20, Xe21, Xe22] 


#axerror = plt.errorbar(Z,Heightideal, xerr=Xetotal, color='red', ecolor = 'red', elinewidth=2, capsize =3);

plt.draw()
os.chdir(path_save) 
plt.savefig(picture_filename)