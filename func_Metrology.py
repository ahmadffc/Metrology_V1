import numpy as np

import matplotlib.pyplot as plt
import shapely
from shapely.geometry import LineString, Point
import math


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
#    if (dotproduct(v1, v2) / (length(v1) * length(v2))) < 1:
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
#    else:
#        return 0

def rotate(v1,ang):
    #theta = np.radians(ang)
    r = np.array(( (np.cos(ang), -np.sin(ang)),
               (np.sin(ang),  np.cos(ang)) ))
    return r.dot(v1)

def Tria(array):
    xx = ([array[0][0]+array[1][0]+array[2][0]])/3
    yy = ([array[0][1]+array[1][1]+array[2][1]])/3
    cent = (xx,yy)
    return cent



def GC(array):
    l1 = LineString([array[0], array[2]])
    l2 = LineString([array[1], array[3]])
    B1int_pt = l1.intersection(l2)
    g_centre = (B1int_pt.x, B1int_pt.y)
    return g_centre

def GC2(array):
    g_centre2 = ((array[0][0]+array[1][0]+array[2][0]+array[3][0])/4, (array[0][1]+array[1][1]+array[2][1]+array[3][1])/4)
    return g_centre2

#def GC3(array):
#    g_centre3 = centroiod.coords(array[0], array[1],array[2])
#    return g_centre3

def GCenter(arr):
    Cabc = np.array([(arr[0][0]+arr[1][0]+arr[2][0])/3,(arr[0][1]+arr[1][1]+arr[2][1])/3])
    Cacd = np.array([(arr[0][0]+arr[2][0]+arr[3][0])/3,(arr[0][1]+arr[2][1]+arr[3][1])/3])
    Cabd = np.array([(arr[0][0]+arr[1][0]+arr[3][0])/3,(arr[0][1]+arr[1][1]+arr[3][1])/3])
    Cdbc = np.array([(arr[3][0]+arr[1][0]+arr[2][0])/3,(arr[3][1]+arr[1][1]+arr[2][1])/3])
    l1 = LineString([Cabc, Cacd])
    l2 = LineString([Cabd, Cdbc])
    B1int_pt = l1.intersection(l2)
    g_cent = (B1int_pt.x, B1int_pt.y)
    return g_cent

def read_cell(x, y):
    with open('data_Metrology.csv', 'r') as f:
        reader = csv.reader(f)
        y_count = 0
        for n in reader:
            if y_count == y:
                cell = n[x]
                return cell
            y_count += 1