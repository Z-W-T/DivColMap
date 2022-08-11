from cv2 import sqrt
from math import pi, sin, pow
import numpy as np

white = np.array([255,255,255])
CharacterMetrix = np.array([[0.4124,0.2126,0.0193],
                            [0.3576,0.7152,0.1192],
                            [0.1805,0.0722,0.9505]])
CharacterMetrixReverse = np.array([[3.2406,-0.9689,0.0557],
                                   [-1.5372,1.8758,-0.2040],
                                   [-0.4986,0.0415,1.0570]])
WHITE = np.dot(white,CharacterMetrix)

def AdjustHue(Msh1, M2):
    M1, s1, h1 = Msh1
    if(M2<M1):
        h2 = h1
    else:
        Mdiff = M2**2-M1**2
        hSpin = s1*sqrt(Mdiff)/(M1*sin(s1))
        if h1 > -pi/3.0:
            h2 = h1 + hSpin
        else:
            h2 = h1 - hSpin
    return h2

def RGB2XYZ(RGB):
    RGB = np.array(RGB)
    XYZ = np.dot(RGB,CharacterMetrix)
    
    return XYZ
    
def f(x):
    if x>0.008856:
        return pow(x,1/3)
    else:
        return 7.787*x+16/116

def g(y):
    if y>pow(0.008856,1/3):
        return y**3
    else:
        return (y-16/116)/7.787

def XYZ2Lab(XYZ):
    L = 116*(f(XYZ[1]/WHITE[1])-16/116)
    a = 500*(f(XYZ[0]/WHITE[0])-f(XYZ[1]/WHITE[1]))
    b = 200*(f(XYZ[1]/WHITE[1])-f(XYZ[2]/WHITE[2]))
    return np.array([L,a,b])

def Lab2MSH(Lab):
    M = Lab[0]**2+Lab[1]**2+Lab[2]**2
    M = sqrt(M)
    s = np.arccos(Lab[0]/M)
    h = np.arctan(Lab[2]/Lab[1])
    return np.array([M,s,h])

def RGB2MSH(RGB):
    XYZ = RGB2XYZ(RGB)
    Lab = XYZ2Lab(XYZ)
    Msh = Lab2MSH(Lab)
    return Msh

def MSH2Lab(Msh):
    M = Msh[0]
    s = Msh[1]
    h = Msh[2]
    L = np.cos(s)*M
    a = sqrt((M**2-L**2)/(pow(np.tan(h),2)+1))
    b = M**2 - L**2 - a**2
    return np.array([L,a,b])

def Lab2XYZ(Lab):
    Y = g((Lab[0]+16)/116)*WHITE[1]
    X = g(a/500+f(Y/WHITE[1]))*WHITE[0]
    Z = g(f(Y/WHITE[1])-Lab[2]/200)*WHITE[2]
    return np.array([X,Y,Z])
    
def XYZ2RGB(XYZ):
    RGB = np.dot(XYZ,CharacterMetrixReverse)
    return RGB

def MSH2RGB(Msh):
    Lab = MSH2Lab(Msh)
    XYZ = Lab2XYZ(Lab)
    RGB = XYZ2RGB(XYZ)
    return RGB

def InterpolateColor(RGB1,RGB2,rate):
    Msh1 = RGB2MSH(RGB1)
    Msh2 = RGB2MSH(RGB2)
    if Msh1[1]>0.05 and Msh2[1]>0.05 and abs(Msh2[2]-Msh1[2])>pi/3:
        Mmid = max(Msh1[0],Msh2[0],88)
        if rate<0.5:
            Msh2 = np.array([Mmid,0,0])
            rate = rate*2
        else:
            Msh1 = np.array([Mmid,0,0])
            rate = rate*2-1
    if Msh1[1]>0.05 and Msh2[1]<0.05:
        h2 = AdjustHue(Msh1,Msh2[0])
        Msh2 = np.array([Msh2[0],Msh2[1],h2])
    if Msh1[1]<0.05 and Msh2[1]>0.05:
        h1 = AdjustHue(Msh2,Msh1[0])
        Msh1 = np.array([Msh1[0],Msh1[1],h1])
    Mmid = Msh2*rate+Msh1*(1-rate)
    return Mmid

