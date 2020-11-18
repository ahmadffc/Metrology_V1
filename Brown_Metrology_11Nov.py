#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:45:54 2019

@author: Nico
"""


import numpy as np
import numpy.linalg as npl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import time
import glob
from numpy import genfromtxt
import pandas as pd

show_plots = False

np.set_printoptions(suppress=True)
matplotlib.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["figure.figsize"] = (10,10)

def raise_window(figname=None):
    if figname: plt.figure(figname)
    cfm = plt.get_current_fig_manager()
    cfm.window.raise_()


############################################################################### RigidTransform class provides functions to 'fit' two clouds of points to each other. In the following, it is used to calculate the rotational shift of the top and bottom coordinates of the needles and the rotaional misalignment of the sensors
class RigidTransform(object):
    def __init__(self, translation=[0,0], rotation=0):
        self.t = translation
        self.R = np.array(((np.cos(rotation),-np.sin(rotation)),(np.sin(rotation), np.cos(rotation))))
        
    def estimate(self, d1, d2):
        p = d1
        q = d2
        p_ = np.mean(p, axis=0)
        q_ = np.mean(q, axis=0)
        x = p - p_
        y = q - q_
        S = np.dot(x.T, y)
        U, _, V = npl.svd(S)
        d = npl.det(np.dot(V, U.T))
        D = np.identity(2)
        D[1,1] = d
        R = np.dot(np.dot(V, D), U.T)
        t = q_ - np.dot(R, p_)
        self.t = t
        self.R = R
        
    def residuals(self, d1, d2):
        res = np.dot(d1, self.R) + self.t - d2
        return np.sqrt(np.power(res[:,0], 2)+np.power(res[:,1], 2))
        
    def __call__(self, d):
        return np.dot(d, self.R)+self.t
    @property
    
    def rotation(self):
        return np.arctan2(self.R[1,0], self.R[0,0])
    @property
    
    def translation(self):
        return self.t

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


# In[4]:

#df = pd.read_excel("data2.xlsx", "2S_102819_1")
#df = pd.read_excel("data2.xlsx", "2S_102819_2")
#df = pd.read_excel("data2.xlsx", "2S_102819_3")
#df = pd.read_excel("data2.xlsx", "2S_102819_4")
#df = pd.read_excel("data2.xlsx", "2S_102819_5")
#df = pd.read_excel("data2.xlsx", "NCP_M4")
#df = pd.read_excel("data2.xlsx", "NCP_Dummy_01")
#df = pd.read_excel("data2.xlsx", "NCP_Dummy_02")
df = pd.read_excel("data2.xlsx", "NCP_Dummy_02_Fmark")
#df = pd.read_excel("data2.xlsx", "Module_D1_arrow_BUP")
#df = pd.read_excel("data2.xlsx", "Module_D1_F1_BUP")

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

#file_list = glob.glob('/Users/apple/TrackerUpgrade/BrownCode/*.TXT')
#file_list.sort()
#for f in file_list:
#    with open(f, 'r') as file:
#        data = file.read()
#    data = data.split(",")
#    data = [float(entry) for entry in data[:-3]] # Don't grab the time and date
    
#Below is for reading data for SA1902    
    #bot = np.array([[data[12],data[13]], [data[15],data[16]], [data[18],data[19]] ,[data[21],data[22]]])
    #top = np.array([[data[36],data[37]], [data[39],data[40]], [data[42],data[43]] ,[data[45],data[46]]])
    #bot_needle =  np.array([[data[0],data[1]], [data[3],data[4]], [data[6],data[7]] ,[data[9],data[10]]])
    #top_needle = np.array([[data[24],data[25]], [data[27],data[28]], [data[30],data[31]] ,[data[33],data[34]]])
  
    
    top = np.array([[px[8],py[8]], [px[9],py[9]], [px[10],py[10]] ,[px[11],py[11]]])
    bot = np.array([[px[12],py[12]], [px[13],py[13]], [px[14],py[14]] ,[px[15],py[15]]])
    top_needle = np.array([[px[0],py[0]], [px[1],py[1]], [px[2],py[2]] ,[px[3],py[3]]])
    bot_needle = np.array([[px[4],py[4]], [px[5],py[5]], [px[6],py[6]] ,[px[7],py[7]]])
    
    #### Below is some testing conditions, keep them commented unless you are testing the code 
    #rot_val = -.05
    #rot_val2 = -.05
    #rot_mat = np.array((np.cos(rot_val), -np.sin(rot_val)),(np.sin(rot_val), np.cos(rot_val)))
    #rot_mat2 = np.array((np.cos(rot_val2), -np.sin(rot_val2)),(np.sin(rot_val2), np.cos(rot_val2)))
    #top = np.matmul(top,rot_mat)
    #bot = np.matmul(bot,rot_mat2)
    
    #print(top[:,1])
                                                          

    ###############################################################################Start of calculation
    top_copy = copy.deepcopy(top)
    bot[:,0],bot_needle[:,0] = np.negative(bot[:,0]), np.negative(bot_needle[:,0]   )        #'Flipping' bottom needle and bottom sensor x-coordinates (measured with flipped carrier, coorinates need to be 'mirrored') 
    bot_copy, bot_needle_copy = copy.deepcopy(bot), copy.deepcopy(bot_needle)
            
#    print('bot copy sensor == ',bot_copy)
#    print('top copy sensor == ',top_copy)
    
    for i,j in enumerate([1,0,3,2]):
        bot[i],bot_needle[i] = bot_copy[j],bot_needle_copy[j]                               #Adaption of coordinate 'order' (otherwise fit does not work properly)

    #print('top_needle == ',top_needle) 
    #print('bot_needle == ',bot_needle)
    #print('top sensor == ',top)
    #print('bot sensor == ',bot)
    ################################################################################Calculation of top-bottom needle alignment                     
    r = RigidTransform(top_needle,bot_needle)
    r.estimate(top_needle,bot_needle)                                                       
    r.residuals(top_needle,bot_needle)                                                      #Implicit calculation of rotational shift of needles 
    delta_cm = cm(bot_needle)-cm(top_needle)                                                #Difference of center of mass to calculate translational shift of needles
    bot_needle_rottrans = np.array(fit(bot_needle,bot_needle,delta_cm,r.rotation)) 
    top_needle_rottrans = np.array(fit(top_needle,top_needle,delta_cm,r.rotation))          #Bottom needles with rotation and translation from needle fit
    #print ('cm top ==', cm(top_needle))
    #print ('cm bot ==', cm(bot_needle))
    #print ('translation needle ==', delta_cm)
    print()
    print('Shift and Rotation of bottom needle pair with respect to top needle pair')
    #print(r'Rotation (μrad):',np.round(-r.rotation*10**6,1))                              #Explicit calculation of rotational shift  of needles 
    #print( r'Translation x,y (μ):',np.round(delta_cm[0],3)*10**3,',',np.round(delta_cm[1],3)*10**3)
    print(r'Rotation (μrad):',-r.rotation*10**6)                              #Explicit calculation of rotational shift  of needles 

    print( r'Translation x,y (μ):',delta_cm[0]*10**3,',',delta_cm[1]*10**3)

    print('top needle after trans_rotation = ', bot_needle_rottrans)

    ################################################################################Adaption of sensors to needle fit

    bot_rottrans = np.array(fit(bot,bot_needle,delta_cm,r.rotation))                        #Bottom sensor coordinates with rotation and translation from needle fit

    ################################################################################Calculation of top-bottom sensor alignment with fit of sensors
    r2= RigidTransform(top,bot_rottrans) 
    r2.estimate(top,bot_rottrans)
    r2.residuals(top,bot_rottrans)                                                           #Implicit calculation of rotational shift of sensor 
    delta_cm2 = cm(bot_rottrans)-cm(top)                                                     #Difference of center of mass to calculate translational shift of sensors

    bot_rottrans_rottrans = np.array(fit(bot_rottrans,bot_rottrans,delta_cm2,r2.rotation))   #Bottom sensor coordinates with rotation and translation from sensor fit

    #print(f.split('/')[-1])
    print( 'Shift and Rotation of top sensor with respect to bottom sensor, CCW positive')
    print(r'Rotation (μrad):',np.round(-r2.rotation*10**6,1)) ### Negative is CCW
    #print( r'Translation x,y (μm):',np.round(delta_cm2[0],4)*1000,',',np.round(delta_cm2[1],4)*1000) 
    print( r'Translation x,y (μm):',np.round(delta_cm2[0]*1000,4),',',np.round(delta_cm2[1]*1000,4)) 

    
    ################################################################################Calculation of top-bottom needle distance to asses measurement quality
    for i in range(len(top_needle[:,0])):
        print ('Distance top-bottom needle pair ',i,r' (μm) : ', np.round(np.sqrt((top_needle[i,0]-bot_needle_rottrans[i,0])**2+(top_needle[i,1]-bot_needle_rottrans[i,1])**2)*10**3,2))
    print( 'Avg. distance top-bottom needle pairs ( ) : ', np.round(np.mean(np.sqrt((top_needle[:,0]-bot_needle_rottrans[:,0])**2+(top_needle[:,1]-bot_needle_rottrans[:,1])**2))*10**3,2))
    print ('Std. distance top-bottom needle pairs (μm) : ', np.round(np.std(np.sqrt((top_needle[:,0]-bot_needle_rottrans[:,0])**2+(top_needle[:,1]-bot_needle_rottrans[:,1])**2),ddof=1)*10**3,2))
    if show_plots == False:
        print('                   ')
    for i in (0,1):                                                                         
        top[:,i]-= top_copy[0,i]                                                            #Shift of objects to lower left corner of top sensor
        top_needle[:,i]-= top_copy[0,i]
        bot_needle_rottrans[:,i]-= top_copy[0,i]
        bot_rottrans[:,i]-= top_copy[0,i]
        bot_rottrans_rottrans[:,i]-= top_copy[0,i]


    ###############################################################################
    plt.close('all')
    if show_plots == True:
        fig0 = plt.figure(0)
        plt.title('Overview of needles, sensors and fit')
        l1 = fig0.add_subplot(111).add_patch(patches.Polygon(top,color='b',alpha=0.35,label='top'))
        l2 = fig0.add_subplot(111).add_patch(patches.Polygon(bot,color='r',alpha=0.35,label='bottom'))
        l3 = fig0.add_subplot(111).add_patch(patches.Polygon(bot_rottrans,color='m',alpha = 0.35, label='bottom rottrans'))
        l4 = fig0.add_subplot(111).add_patch(patches.Polygon(bot_rottrans_rottrans,color='g',alpha=0.35,label='bottom rottrans rottrans'))
        space_holder = matplotlib.lines.Line2D(range(10), range(10), marker='', color="white")
        l5, = plt.plot(top_needle[:,0],top_needle[:,1],'bD', label = 'top_needle')
        l6, = plt.plot(bot_needle[:,0],bot_needle[:,1],'rD', label = 'bot_needle')
        l7, = plt.plot(bot_needle_rottrans[:,0],bot_needle_rottrans[:,1],'gD', label ='bot_needle_rottrans', alpha=0.5)
        plt.axis('equal')
        plt.grid()
        plt.legend((l5,l6,l7,space_holder,l1,l2,l3,l4),('top_needle','bot_needle','bot_needle_rotttans','','top','bottom','bottom_rottrans','bottom_rottrans_rottrans'),ncol=2)
        time.sleep(0.01)




        z1=3  #####Plot range
        fig1 = plt.figure(1)
        plt.suptitle('Zoomed view of top, bottom and fitted bottom sensor',y=0.995)
        for i,e in enumerate([3,2,0,1]):
            plt.subplot(2,2,i+1)
            plt.axis('equal')
            l1=fig1.add_subplot(2,2,i+1).add_patch(patches.Polygon(top,color='b',alpha=0.35,label='Top'))
            l2=fig1.add_subplot(2,2,i+1).add_patch(patches.Polygon(bot_rottrans,color='r',alpha=0.35,label='Bottom rottrans'))
            #l3=fig1.add_subplot(2,2,i+1).add_patch(patches.Polygon(bot_rottrans_rottrans,color='g',alpha=0.35,label='Bottom rottrans2'))
            plt.xlim(top[e,0]-z1,top[e,0]+z1)
            plt.ylim(top[e,1]-z1,top[e,1]+z1)
            plt.grid()
            if i == 0:
                plt.ylabel('(mm)')
            if i == 2:
                plt.xlabel('(mm)')
                plt.ylabel('(mm)')
                plt.ticklabel_format(style='plain')
            if i == 3:
                plt.xlabel('(mm)')
                plt.legend((l1,l2,l3),['Top sensor','Bottom sensor','Bottom sensor fit'],numpoints=1,loc='lower right')
                #plt.legend((l1,l2,l3),['Bottom sensor','Top sensor'],numpoints=1,loc='lower right') ## I CHANGED THIS TO REFLECT SA1902, CHANGE BACK!

            else: None
        plt.tight_layout()
        time.sleep(0.01)



        z2 = 0.015
        fig2 = plt.figure(2)
        plt.suptitle('Zoomed view of top and fitted bottom needles',y=0.995)
        for i,e in enumerate([3,2,0,1]):
            plt.subplot(2,2,i+1)
            plt.axis('equal')
            l1,=plt.plot(top_needle[:,0],top_needle[:,1],'b.', alpha=0.5)
            l2,=plt.plot(bot_needle_rottrans[e,0],bot_needle_rottrans[e,1],'r.',alpha=0.5)
            plt.xlim(top_needle[e,0]-z2,top_needle[e,0]+z2)
            plt.ylim(top_needle[e,1]-z2,top_needle[e,1]+z2)
            plt.grid()
            if i == 0:
                plt.ylabel('(mm)')
            if i == 2:
                plt.xlabel('(mm)')
                plt.ylabel('(mm)')
                plt.ticklabel_format(style='plain')
            if i == 3:
                plt.xlabel('(mm)')
                plt.legend((l1,l2),['Top needles','Bottom needles fit'],numpoints=1,loc='lower right')
            else: None

        plt.show()
        plt.tight_layout()
        time.sleep(0.01)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




