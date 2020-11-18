#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:45:54 2019
@author: Nico
"""


import csv
import numpy as np

#import matplotlib.pyplot as plt
#import shapely
from shapely.geometry import LineString, Point
import math 
import copy
from func_Metrology import GC,GC2,angle, length, dotproduct, Tria, GCenter

from numpy import genfromtxt
import pandas as pd

def rotate1(v1,ang):
    #theta = np.radians(ang)
    r = np.array(( (np.cos(ang), -np.sin(ang)),
               (np.sin(ang),  np.cos(ang)) ))
    return r.dot(v1)

def cm(x):                                                                                              #Calculation of coordinates of center of mass
    return np.array([np.average(x[:,0]),np.average(x[:,1])])

def fit(fit_array,origin_array,delta,angles):
    com_r = np.sqrt((fit_array[:,0]-cm(origin_array)[0])**2+(fit_array[:,1]-cm(origin_array)[1])**2)    #Distance of needles i.e. sensor corners to the array's center of mass
    com_xy = fit_array-cm(origin_array)                                                                 #Position of needles i.e. sensor corners in center of mass coordinates
    com_angles=np.arccos(com_xy[:,0]/com_r)                                                             #Angles of needles i.e. sensor corners in center of mass coordinates
    
    for i in range(len(com_angles)):
        if   com_xy[i,0] >= 0 and com_xy[i,1] >= 0: com_angles[i] = com_angles[i]- angles               #Applying rotational shift from fit to angles; since angles are measured from x-axis, values need to be added or subtracted depending on the quartile of coordinate system 
        elif com_xy[i,0] < 0  and com_xy[i,1] >= 0: com_angles[i] = com_angles[i]- angles
        elif com_xy[i,0] < 0  and com_xy[i,1] < 0:  com_angles[i] = - abs(com_angles[i])- angles
        elif com_xy[i,0] >= 0 and com_xy[i,1] < 0:  com_angles[i] = - abs(com_angles[i])- angles
        else: print ('Problem in angle transformation')
    
    cm_rot = np.zeros(np.shape(fit_array))
    cm_rot[:,0],cm_rot[:,1]= com_r*np.cos(com_angles), com_r*np.sin(com_angles)                         #Needles i.e. sensor coordinates with rotational shift in center of mass coordinates
    rot=cm_rot+cm(origin_array)                                                                         #Needles i.e. sensor coordinates with rotational
    rottrans=rot-delta                                                                                  #Needles i.e. sensor coordinates with rotational and translational shift
    return rottrans


#df = pd.read_excel("data2.xlsx", "2S_102819_1")
#df = pd.read_excel("data2.xlsx", "2S_102819_2")
#df = pd.read_excel("data2.xlsx", "2S_102819_3")
#df = pd.read_excel("data2.xlsx", "2S_102819_4")
#df = pd.read_excel("data2.xlsx", "2S_102819_5")
#df = pd.read_excel("data2.xlsx", "NCP_M4")
#df = pd.read_excel("data2.xlsx", "NCP_Dummy_01")
#df = pd.read_excel("data2.xlsx", "NCP_Dummy_02")
df = pd.read_excel("data2.xlsx", "NCP_Dummy_02_Fmark")
#df = pd.read_excel("data2.xlsx", "testing")


df1 = pd.DataFrame(df, columns= ['x'])
df2 = pd.DataFrame(df, columns= ['y'])

px = df1['x']; py = df2['y']


nt = []; nb = []; mt = []; mb = []
for x in range(0,4):
    nt.append(np.array([px[x],py[x]],dtype=np.float))
for x in range(4,8):
    nb.append(np.array([-px[x],py[x]],dtype=np.float))
for x in range(8,12):
    mt.append(np.array([px[x],py[x]],dtype=np.float))
for x in range(12,16):
    mb.append(np.array([-px[x],py[x]],dtype=np.float))   



bot_copy, bot_needle_copy = copy.deepcopy(mb), copy.deepcopy(nb)
for i,j in enumerate([1,0,3,2]):
    mb[i],nb[i] = bot_copy[j],bot_needle_copy[j]   
    

#sprint('array nt = ',nt, '\n\narray nb = ',nb, '\n\narray mt = ',mt, '\n\narray mb = ',mb)


nt_GC = GC2(nt)
#nt_GC = GCenter(nt)
#print('\ncm top == ', nt_GC[0],nt_GC[1])

# Digonal vectors of Bottom needle frame  
#nb_GC = GCenter(nb)
nb_GC = GC2(nb)
#print('cm bot == ', nb_GC[0],nb_GC[1])
#translation = np.array([(nb_GC[0] - nt_GC[0]), (nb_GC[1] - nt_GC[1])])
translation = np.array([(nb_GC[0] - nt_GC[0]), (nb_GC[1] - nt_GC[1])])

#print('translation needle == ', translation)

trans_nt=[]
for x in nt:
    trans_nt.append(x + translation)
#print('\n top needle frame after translation = ', trans_nt)

trans_nt_GC = GCenter(trans_nt)
#print('\n trans_nt needle frame GC = ', trans_nt_GC[0],trans_nt_GC[1])


diff_nb_trans_nt = [(nb[0][0]-trans_nt[0][0],nb[0][1]-trans_nt[0][1]),(nb[1][0]-trans_nt[1][0],nb[1][1]-trans_nt[1][1]),(nb[2][0]-trans_nt[2][0],nb[2][1]-trans_nt[2][1]),(nb[3][0]-trans_nt[3][0],nb[3][1]-trans_nt[3][1])]
diff_nt_nb = [(nb[0][0]-nt[0][0],nb[0][1]-nt[0][1]),(nb[1][0]-nt[1][0],nb[1][1]-nt[1][1]),(nb[2][0]-nt[2][0],nb[2][1]-nt[2][1]),(nb[3][0]-nt[3][0],nb[3][1]-nt[3][1])]

#print('\nDifference in reference bottom and translated top bridge coordinates = \n',np.round(diff_nt_nb,2))
#print('\nDifference in bottom and Translated top bridge coordinates = \n', np.round(diff_nb_trans_nt,2))

o_b = trans_nt_GC #GC of top needle frame

nb_dig=[]
for x in nb:
    nb_dig.append(x - o_b)
trans_nt_dig=[]
for x in trans_nt:
    trans_nt_dig.append(x - o_b)

A=[0,0,0,0]
for i in range(4):
    A[i]= angle(nb_dig[i],trans_nt_dig[i])
    #A[i]= angle(trans_nt_dig[i],nb_dig[i])
print('Angle 1 of rotation in radian= ', A[0])
print('Angle 2 of rotation in radian= ', A[1])
print('Angle 3 of rotation in radian= ', A[2])
print('Angle 4 of rotation in radian= ', A[3])
A_avg = (A[0]+A[1]+A[2]+A[3])/4
#print('Avarage 1 - 4 in radian= ', A_avg)

print(r'Rotation (μrad):',A_avg*10**6)                              #Explicit calculation of rotational shift  of needles 
print( r'Translation x,y (μ):',translation[0],',',translation[1])

#bot_needle_rottrans=[]
#bot_needle_rottrans = np.array(fit(nb,nb,translation,A_avg)) 




#b_o1= nb[0] - o_b
#t_o1= trans_nt[0] - o_b
#Angle1 = angle(b_o1,t_o1)
#b_o2= nb[1] - o_b
#t_o2= trans_nt[1] - o_b
#Angle2 = angle(b_o2,t_o2)
#print('\n angle btw first digonals Angle1 == ',Angle1 )
#print('\n angle btw first digonals Angle2 == ',Angle2 )

trans_rotated_nt=[]
for x in trans_nt_dig:
    trans_rotated_nt.append((rotate1(x,A_avg)) + o_b)
    #trans_rotated_nt.append(rotate(x,A_avg) + o_b)
   
print('\n top needle after trans_rotation = ', trans_rotated_nt)

diff_nt_trans_rot_nb = [(trans_rotated_nt[0][0]-nb[0][0],trans_rotated_nt[0][1]-nb[0][1]),(trans_rotated_nt[1][0]-nb[1][0],trans_rotated_nt[1][1]-nb[1][1]),(trans_rotated_nt[2][0]-nb[2][0],trans_rotated_nt[2][1]-nb[2][1]),(trans_rotated_nt[3][0]-nb[3][0],trans_rotated_nt[3][1]-nb[3][1])]
#print('\nDifference in top and rotated bottom bridge coordinates = \n', np.round(diff_nt_trans_rot_nb,2))

trans_mt=[]
for x in mt:
    trans_mt.append(x + translation)
    
#print(' \n top sensor after translation = ', trans_mt)

mb_GC = GC2(mb)
#print('\n GC of mb == ', GCenter(mb))
#print('\n GC of mt == ', GCenter(mt))
#print(' GC of trans_mt == ', GCenter(trans_mt))

diff_nb_trans_mt = [(mb[0][0]-trans_mt[0][0],mb[0][1]-trans_mt[0][1]),(mb[1][0]-trans_mt[1][0],mb[1][1]-trans_mt[1][1]),(mb[2][0]-trans_mt[2][0],mb[2][1]-trans_mt[2][1]),(mb[3][0]-trans_mt[3][0],mb[3][1]-trans_mt[3][1])]
#print('\nDifference in top and rotated bottom bridge coordinates = \n', np.round(diff_nb_trans_mt,2))

trans_mt_dig=[]
for x in trans_mt:
    trans_mt_dig.append(x - o_b)
mb_dig=[]
for x in mb:
    mb_dig.append(x - o_b)
    
trans_rotated_mt=[]
for x in trans_mt_dig:
    trans_rotated_mt.append(rotate1(x,A_avg) +o_b)
    #trans_rotated_mt.append(rotate1(x,A[0])+o_b)

#print(' \n top sensor after translation and rotation= ', trans_rotated_mt)

mt_GC = GC2(trans_rotated_mt)
#print(' GC of trans_rotated_mt == ', GCenter(trans_rotated_mt))


translation_sensors_1 = np.array([(mb_GC[0] - mt_GC[0]), (mb_GC[1] - mt_GC[1])])

print('\n\n Translation vector between sensors == ', translation_sensors_1*1000)

o_s = mt_GC 

mb_dig=[]
for x in mb:
    mb_dig.append(x - o_s)
trans_mt_dig=[]
for x in trans_rotated_mt:
    trans_mt_dig.append(x - o_s)
 
M=[0,0,0,0]
for i in range(4):
    M[i]= angle(mb_dig[i],trans_mt_dig[i])
    #A[i]= angle(trans_nt_dig[i],nb_dig[i])
#print('Sensor Angle 1 of rotation in radian= ', M[0])
#print('Sensor Angle 2 of rotation in radian= ', M[1])
#print('Sensor Angle 3 of rotation in radian= ', M[2])
#print('Sensor Angle 4 of rotation in radian= ', M[3])
M_avg = (M[0]+M[1]+M[2]+M[3])/4
print('Sensors Angle Avarage 1 - 4 in radian= ', M_avg*1000000)

#x1=1663.1
#y1=107.5
#x2=104072.7
#y2=2441.7
#distance = math.sqrt(((x2-x1)**2)+((y2-y1)**2))
#print ('distance is === ',distance)

#x12=1582.6
#y12=1187.3
#x22=104014.7
#y22=2322.8
#distance2 = math.sqrt(((x22-x12)**2)+((y22-y12)**2))
#print ('distance is === ',distance2)
#trans_rotated_mt=[]
#for x in mt:
#    trans_rotated_mt.append((rotate1(x,-A_avg))+o_b)
    #trans_rotated_mb.append(rotate(x,(A[0]+A[1])/2) + o_b)

#print(' \n top sensor after translation and rotation= ', trans_rotated_mt)

#diff_nb_trans_rotated_mt = [(mb[0][0]-trans_rotated_mt[0][0],mb[0][1]-trans_rotated_mt[0][1]),(mb[1][0]-trans_rotated_mt[1][0],mb[1][1]-trans_rotated_mt[1][1]),(mb[2][0]-trans_rotated_mt[2][0],mb[2][1]-trans_rotated_mt[2][1]),(mb[3][0]-trans_rotated_mt[3][0],mb[3][1]-trans_rotated_mt[3][1])]
#print('\nDifference in top and rotated bottom bridge coordinates = \n', np.round(diff_nb_trans_rotated_mt,2))

