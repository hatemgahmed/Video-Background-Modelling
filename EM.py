import numpy as np
import cv2
import math
import shelve
import random

# reads image 'opencv-logo.png' as grayscale
img = cv2.imread('Dist2.jpeg', 0)
u1 = 150
s1 = 64
w1 = 0.5
u2 = 220
s2 = 64
w2 = 0.5

# u1 = 64
# s1 = 128
# w1 = 0.5
# u2 = 192
# s2 = 128
# w2 = 0.5
threashold = 0.5
# adding some noise to not have standard deviation=0
# not noticable to the naked eye
# for i in range(len(img)):
#     for j in range(len(img[0])):
#         img[i][j] += random.randint(-3, 3)


def gauss(x, mean, deviation):
    # special case when deviation=0, gaussian destribution becomes dirac delta function S(x-u)D
    if(deviation == 0):
        if(x == mean):
            return 1
        else:
            return 0
    return math.e**(- 0.5*(1/deviation**2)*(x-mean)**2)/(deviation*math.sqrt(2*math.pi))


def mean(data):
    result = 0
    for i in data:
        result += i
    return result/len(data)


def standardDeviation(data, u):
    result = 0
    for i in data:
        result += (i-u)**2
    return math.sqrt(result/len(data))


def getGauss(data):
    u = mean(data)
    s = standardDeviation(data, u)
    return u, s


stop = False
background = np.zeros_like(img)
frame = 0
while(True):
    L1 = []
    L2 = []
    background = np.zeros_like(img)
    # Expectation
    print(frame)
    frame += 1
    for i in range(len(img)):
        for j in range(len(img[0])):
            x = img[i][j]  # pixel
            r1 = gauss(x, u1, s1)
            r2 = gauss(x, u2, s2)
            r1, r2 = r1*w1/(r1+r2), r2*w2/(r1+r2)
            back = 0  # it is not background
            if(r1 > r2):
                L1.append(x)
                back = 255  # it is background
            else:
                L2.append(x)
            background[i][j] = back
        # print(L1)
        # print(L2)
    if(stop):
        break
    u1_new, s1_new = getGauss(L1)
    u2_new, s2_new = getGauss(L2)
    error = max(abs(u1_new-u1), abs(u2_new-u2), abs(s1_new-s1), abs(s2_new-s2))
    print(error)
    print((u1_new, s1_new))
    print((u2_new, s2_new))
    if(error < threashold):
        stop = True
    w1 = len(L1)/(len(L1)+len(L2))
    w2 = 1-w1
    u1, u2, s1, s2 = u1_new, u2_new, s1_new, s2_new
f = shelve.open('Task3')
f['image'] = background
cv2.imshow('output', background)
cv2.imwrite('back_for.jpg', background)
cv2.waitKey(0)
f.close()
