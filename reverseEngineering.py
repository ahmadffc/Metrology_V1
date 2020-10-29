#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:47:49 2020
@author: apple
"""
import csv
import numpy as np

#import matplotlib.pyplot as plt
#import shapely
from shapely.geometry import LineString, Point
import math 

from func_Metrology import GC,GC2,angle,rotate, length, dotproduct, Tria, GCenter
from numpy import genfromtxt
import pandas as pd

#xls = pd.read_excel('data1.xlsx')
#df = pd.read_excel(xls, 'Sheet1')
#df = pd.read_excel("data2.xlsx", "Module4_R2_with_standard_bot_re")
#df = pd.read_excel("data2.xlsx", "Module4_R2_with_standard")
df = pd.read_excel("data2.xlsx", "Module4_brown")

#df = pd.read_excel("data2.xlsx", "Module4_brown")

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

#print('array nt = ',nt, '\n\narray nb = ',nb, '\n\narray mt = ',mt, '\n\narray mb = ',mb)
print('\narray mt = ',mt, '\narray mb = ',mb)




trans_rotated_mt_2 =  [np.array([2314.01, 2364.18]), np.array([104393.25,   -209.71]), np.array([106746.27,  93352.62]), np.array([ 4671.45, 95931.25])]
top_sensor_after_translation = [np.array([6860.15320168, -687.608018  ]), np.array([108948.99320168,   2250.991982  ]), np.array([106257.35320168,  95817.821982  ]), np.array([ 4172.63320168, 92879.781982  ])]


o_b = GCenter(trans_rotated_mt_2) #GC of top needle frame

nb_dig=[]
for x in top_sensor_after_translation:
    nb_dig.append(x - o_b)
trans_nt_dig=[]
for x in trans_rotated_mt_2:
    trans_nt_dig.append(x - o_b)

A=[0,0,0,0]
for i in range(4):
    A[i]= angle(nb_dig[i],trans_nt_dig[i])
print('Angle 1 of rotation in radian= ', A[0])
#print('Angle 2 of rotation in radian= ', A[1])
#print('Avarage 1 & 2 in radian= ', (A[0]+A[1])/2)



translation = (-4.8, -146)

trans_mb=[]
for x in mb:
    trans_mb.append(x - translation)
print('\n Translated top sensor coordinated == ', trans_mb)

mb_GC = GCenter(mb)
print('\n bottom needle frame GC = ', mb_GC[0],mb_GC[1])
trans_mb_GC = GCenter(trans_mb)
print('\n trans_mt_GC needle frame GC = ', trans_mb_GC[0],trans_mb_GC[1])

mt_GC = GCenter(mt)
print('\n top needle frame GC = ', mt_GC[0],mt_GC[1])

translation_sensors = np.array([(mb_GC[0] - trans_mb_GC[0]), (mb_GC[1] - trans_mb_GC[1])])
print('Translation vector of Sensors == ', translation_sensors)

"""
#nt_GC = GC(nt)
nt_GC = GCenter(nt)
print('\n top needle frame GC = ', nt_GC[0],nt_GC[1])

# Digonal vectors of Bottom needle frame  
nb_GC = GCenter(nb)
#nb_GC = GCenter(nb)
print('\n bottom needle frame GC = ', nb_GC[0],nb_GC[1])
#translation = np.array([(nb_GC[0] - nt_GC[0]), (nb_GC[1] - nt_GC[1])])
translation = np.array([(nb_GC[0] - nt_GC[0]), (nb_GC[1] - nt_GC[1])])

print('\n Translation vector of needle frames == ', translation)

trans_nt=[]
for x in nt:
    trans_nt.append(x + translation)
print('\n top needle frame after translation = ', trans_nt)

trans_nt_GC = GCenter(trans_nt)
print('\n trans_nt needle frame GC = ', trans_nt_GC[0],trans_nt_GC[1])


diff_nb_trans_nt = [(trans_nt[0][0]-nb[0][0],trans_nt[0][1]-nb[0][1]),(trans_nt[1][0]-nb[1][0],trans_nt[1][1]-nb[1][1]),(trans_nt[2][0]-nb[2][0],trans_nt[2][1]-nb[2][1]),(trans_nt[3][0]-nb[3][0],trans_nt[3][1]-nb[3][1])]
diff_nt_nb = [(nt[0][0]-nb[0][0],nt[0][1]-nb[0][1]),(nt[1][0]-nb[1][0],nt[1][1]-nb[1][1]),(nt[2][0]-nb[2][0],nt[2][1]-nb[2][1]),(nt[3][0]-nb[3][0],nt[3][1]-nb[3][1])]

print('\nDifference in initial top and bottom bridge coordinates = \n',np.round(diff_nt_nb,2))
print('\nDifference in bottom and Translated top bridge coordinates = \n', np.round(diff_nb_trans_nt,2))

o_b = nb_GC #GC of top needle frame
nb_dig=[]
for x in nb:
    nb_dig.append(x - o_b)
trans_nt_dig=[]
for x in trans_nt:
    trans_nt_dig.append(x - o_b)

A=[0,0,0,0]
for i in range(4):
    A[i]= angle(nb_dig[i],trans_nt_dig[i])
print('Angle 1 of rotation in radian= ', A[0])
#print('Angle 2 of rotation in radian= ', A[1])
#print('Avarage 1 & 2 in radian= ', (A[0]+A[1])/2)

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
    #trans_rotated_nt.append(rotate(x,(A[0]+A[1])/2) +o_b)
    #trans_rotated_nt.append(rotate(x,(1000000)/2) +o_b)
    #trans_rotated_nt.append(rotate(x,A[1]) +o_b)
    trans_rotated_nt.append(rotate(x,-A[0])+o_b)
print('\n bottom fram after Rotation = ', trans_rotated_nt)

diff_nt_trans_rot_nb = [(trans_rotated_nt[0][0]-nb[0][0],trans_rotated_nt[0][1]-nb[0][1]),(trans_rotated_nt[1][0]-nb[1][0],trans_rotated_nt[1][1]-nb[1][1]),(trans_rotated_nt[2][0]-nb[2][0],trans_rotated_nt[2][1]-nb[2][1]),(trans_rotated_nt[3][0]-nb[3][0],trans_rotated_nt[3][1]-nb[3][1])]
print('\nDifference in top and rotated bottom bridge coordinates = \n', np.round(diff_nt_trans_rot_nb,2))

trans_mt=[]
for x in mt:
    trans_mt.append(x + translation)

print(' \n top sensor after translation = ', trans_mt)

mb_dig=[]
for x in mb:
    mb_dig.append(x - o_b)
trans_mt_dig=[]
for x in trans_mt:
    trans_mt_dig.append(x - o_b)
    
trans_rotated_mt=[]
#for x in trans_mb_dig:
for x in trans_mt_dig:
    trans_rotated_mt.append(rotate(x,-A[0])+o_b)
    #trans_rotated_mb.append(rotate(x,(A[0]+A[1])/2) + o_b)

print(' \n top sensor after translation and rotation= ', trans_rotated_mt)
#print(trans_rotated_mb)

#arr_mb_new = trans_rotated_mb

#mt_GC = GC(mt)
mt_GC = GCenter(trans_rotated_mt)
print('top sensor GC = %.3f'% mt_GC[0],mt_GC[1])
#mb_GC = GC(arr_mb_new)
mb_GC = GCenter(mb)
print('bottom sensor GC = %.3f'% mb_GC[0],mb_GC[1])
translation_sensors = np.array([(mb_GC[0] - mt_GC[0]), (mb_GC[1] - mt_GC[1])])
#translation_sensors = np.array([(mt_GC[0] - mb_GC[0]), (mt_GC[1] - mb_GC[1])])

print('Translation vector of Sensors == ', translation_sensors)

"""


