#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:47:49 2020

@author: apple
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 03:54:57 2020

@author: apple
"""

#Applying Translation and Rotation to pins
import csv
import numpy as np

#import matplotlib.pyplot as plt
#import shapely
from shapely.geometry import LineString, Point
import math 

from func_Metrology import GC,GC2,angle,rotate, length, Tria, GCenter, dotproduct
from numpy import genfromtxt
import pandas as pd

df = pd.read_excel("data2.xlsx", "Module4_brown")

#df = pd.read_excel("data1.xlsx", "Module4_R2_Brown_standard")
#df = pd.read_excel("data2.xlsx", "Module4_R2_with_standard")
#df = pd.read_excel("data2.xlsx", "Module4_R2_with_standard_bot_re")
#df = pd.read_excel("data1.xlsx", "Module4_R2")
#df = pd.read_excel("data1.xlsx", "Module3_R2")
#df = pd.read_excel("data1.xlsx", "Module4_R2_reacces")

df1 = pd.DataFrame(df, columns= ['x'])
df2 = pd.DataFrame(df, columns= ['y'])

px = df1['x']; py = df2['y']


nt = []; nb = []; mt = []; mb = []
for x in range(0,4):
    nt.append(np.array([px[x],py[x]],dtype=np.float))
for x in range(4,8):
    nb.append(np.array([px[x],py[x]],dtype=np.float))
for x in range(8,12):
    mt.append(np.array([px[x],py[x]],dtype=np.float))
for x in range(12,16):
    mb.append(np.array([px[x],py[x]],dtype=np.float))    

print('array nt = ',nt, '\n\narray nb = ',nb, '\n\narray mt = ',mt, '\n\narray mb = ',mb)
################################################################
## Geometrical Centres of Needle frames and Translation Vector #######
################################################################
# Digonal vectors of Top needle frame  


#nt_GC = GC2(nt)
nt_GC = GCenter(nt)
print('\n top needle frame GC = ', nt_GC[0],nt_GC[1])

# Digonal vectors of Bottom needle frame  
#nb_GC = GC2(nb)
nb_GC = GCenter(nb)
print('\n bottom needle frame GC = ', nb_GC[0],nb_GC[1])
translation = np.array([(nb_GC[0] - nt_GC[0]), (nb_GC[1] - nt_GC[1])])
print('\n Translation vector of needle frames == ', translation)

#######################################################
### Applying Translation on Bottom Needle frame ##
#######################################################
trans_nb=[]
for x in nb:
    trans_nb.append(x - translation)

print('\n bottom needle frame after translation = ', trans_nb)
#####################################################################
### Making Digonal vectors from geometrical centre: For Needle frames  ####
############# And Angles between the Digonal vectors ################
#####################################################################

nt12 = nt[1] - nt[0]
nb12 = nb[1] - nb[0]
Ang = angle(nt12,nb12)



o_b = nt_GC #GC of top needle frame

nt_dig=[]
for x in nt:
    nt_dig.append(x - o_b)
trans_nb_dig=[]
for x in trans_nb:
    trans_nb_dig.append(x - o_b)

A=[0,0,0,0]
for i in range(4):
    A[i]= angle(nt_dig[i],trans_nb_dig[i])
print('Angle 1 of rotation in radian= ', A[0])
print('Angle 2 of rotation in radian= ', A[1])
print('Avarage 1 & 2 in radian= ', (A[0]+A[1])/2)

trans_rotated_nb=[]
#for x in trans_nb_dig:
for x in nt_dig:
#for x in trans_nb:
    #trans_rotated_nb.append(rotate(x,(A[0]+A[1])/2) +o_b)
    #trans_rotated_nb.append(rotate(x,(1000000)/2) +o_b)
    #trans_rotated_nb.append(rotate(x,A[1]) +o_b)
    trans_rotated_nb.append(rotate(x,Ang)+o_b)
print('\n bottom fram after Rotation = ', trans_rotated_nb)

diff_nt_trans_rot_nb = [(trans_rotated_nb[0][0]-nt[0][0],trans_rotated_nb[0][1]-nt[0][1]),(trans_rotated_nb[1][0]-nt[1][0],trans_rotated_nb[1][1]-nt[1][1]),(trans_rotated_nb[2][0]-nt[2][0],trans_rotated_nb[2][1]-nt[2][1]),(trans_rotated_nb[3][0]-nt[3][0],trans_rotated_nb[3][1]-nt[3][1])]
diff_nt_nb = [(nt[0][0]-nb[0][0],nt[0][1]-nb[0][1]),(nt[1][0]-nb[1][0],nt[1][1]-nb[1][1]),(nt[2][0]-nb[2][0],nt[2][1]-nb[2][1]),(nt[3][0]-nb[3][0],nt[3][1]-nb[3][1])]

print('\nDifference in initial top and bottom bridge coordinates = \n',np.round(diff_nt_nb,2))
print('\nDifference in top and rotated bottom bridge coordinates = \n', np.round(diff_nt_trans_rot_nb,2))

"""

########################################################
### Translate and Rotate Bottom Sensor by Angles A[0] ##
########################################################
trans_mb=[]
for x in mb:
    trans_mb.append(x - translation)

#print(' \n bottom sensor after translation = ', trans_mb)
mt_dig=[]
for x in mt:
    mt_dig.append(x - o_b)
trans_mb_dig=[]
for x in trans_mb:
    trans_mb_dig.append(x - o_b)
    
trans_rotated_mb=[]
#for x in trans_mb_dig:
for x in mt_dig:
    trans_rotated_mb.append(rotate(x,Ang)+o_b)
    #trans_rotated_mb.append(rotate(x,(A[0]+A[1])/2) + o_b)

#print(' \n bottom sensor after translation and rotation= ', trans_rotated_mb)
#print(trans_rotated_mb)
################################
### Sesnsor's Operations statrt
################################

arr_mb_new = trans_rotated_mb

#mt_GC = GC(mt)
mt_GC = GCenter(mt)
print('top sensor GC = %.3f'% mt_GC[0],mt_GC[1])
#mb_GC = GC(arr_mb_new)
mb_GC = GCenter(arr_mb_new)
print('bottom sensor GC = %.3f'% mb_GC[0],mb_GC[1])
translation_sensors = np.array([(mb_GC[0] - mt_GC[0]), (mb_GC[1] - mt_GC[1])])
#print('Translation vector of Sensors == ', translation_sensors)
trans_mb_new = []
for x in arr_mb_new:
    trans_mb_new.append(x - translation_sensors)

o_s = mt_GC

mt_dig=[]
for x in mt:
    mt_dig.append(x - o_s)
trans_mb_new_dig=[]
for x in trans_mb_new:
    trans_mb_new_dig.append(x - o_s)

M=[0,0,0,0]
for i in range(4):
    M[i]= angle(mt_dig[i],trans_mb_new_dig[i])
#print('Angle 1 A[0] of rotation in radian= ', M[0])
#print('Angle 2 A[1] of rotation in radian= ', M[1])
#print('Angle A[0] of rotation in degree = ', M[0]*180/np.pi)

trans_rotated_mb_new=[]
for x in trans_mb_new_dig:
    trans_rotated_mb_new.append(rotate(x,M[0])+o_s)

diff_mt_trans_rot_mb_new = [(trans_rotated_mb[0][0]-mt[0][0],trans_rotated_mb[0][1]-mt[0][1]),(trans_rotated_mb[1][0]-mt[1][0],trans_rotated_mb[1][1]-mt[1][1]),(trans_rotated_mb[2][0]-mt[2][0],trans_rotated_mb[2][1]-mt[2][1]),(trans_rotated_mb[3][0]-mt[3][0],trans_rotated_mb[3][1]-mt[3][1])]

######################################

print('\n-------------- After Bottom Bridge Translation and Rotation ----------------')
print(' top side = ', nt)
print(' bottom side = ', nb)
print(' \nTranslated bottom side = ',trans_nb)
print(' \nTranslated and Rotated bottom side = ',trans_rotated_nb)
print('\nDifference in initial top and bottom bridge coordinates = \n',np.round(diff_nt_nb,2))
print('\nDifference in top and rotated bottom bridge coordinates = \n', np.round(diff_nt_trans_rot_nb,2))
print('\nDifference in top and rotated bottom sensor coordinates = \n',np.round(diff_mt_trans_rot_mb_new,2))

print( '######### Metrology Results############## ')
print('Translation vector between sensors = ', translation_sensors )
print('Angle 1 A[0] of rotation in radian= ', M[0])
print('Angle 2 A[1] of rotation in radian= ', M[1])
print('Average angle in radian = ', (M[0] + M[1])/2)

"""