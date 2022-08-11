from cv2 import sqrt
from math import pi, sin
import numpy as np

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
    CharacterMetrix = np.array([[0.4124,0.2126,0.0193],
                                [0.3576,0.7152,0.1192],
                                [0.1805,0.0722,0.9505]])
    XYZ = np.dot(RGB,CharacterMetrix)
    return XYZ
    

def RGB2MSH(RGB):


def InterpolateColor(RGB1,RGB2,rate):

