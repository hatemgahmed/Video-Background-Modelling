from __future__ import division

import numpy as np
import cv2
import math
import shelve


# def gauss(x, mean, deviation):
#     # special case when deviation=0, gaussian destribution becomes dirac delta function S(x-u)D
#     if(deviation == 0):
#         if(x == mean):
#             return 1
#         else:
#             return 0
#     return math.e**(- 0.5*(1/deviation**2)*(x-mean)**2)/(deviation*math.sqrt(2*math.pi))

def gauss(x, mean, deviation):
    # special case when deviation=0, gaussian destribution becomes dirac delta function S(x-u)D
    # if(deviation == 0):
    #     if(x == mean):
    #         return 1
    #     else:
    #         return 0
    return (1/(np.sqrt(2*np.pi)*deviation)) * (np.exp((-1/(2*(deviation**2))) * ((x-mean)**2)))
    # return np.exp(- 0.5*(1/deviation**2)*(x-mean)**2)/(deviation*np.sqrt(2*np.pi))


# vectorGauss = np.vectorize(gauss)

cap = cv2.VideoCapture('rouen_video.mp4.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
u1 = 150
s1 = 64
w1 = 0.5
u2 = 220
s2 = 64
w2 = 0.5
u1 = np.full((frame_height, frame_width), u1)
s1 = np.full((frame_height, frame_width), s1)
w1 = np.full((frame_height, frame_width), w1)
u2 = np.full((frame_height, frame_width), u2)
s2 = np.full((frame_height, frame_width), s2)
w2 = np.full((frame_height, frame_width), w2)

o1 = np.zeros((frame_height, frame_width))
o2 = np.zeros((frame_height, frame_width))
background = np.zeros((frame_height, frame_width))
# outVid = cv2.VideoWriter('GMM.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
#                          (frame_width, frame_height), 0)
outVid = cv2.VideoWriter('GMM.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                         (frame_width, frame_height), 0)

L = 100
alpha = (1/L)
count = 0
while(cap.isOpened()):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    if ret:
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break

        print(count)
        count += 1
        # Expectation
        r1 = gauss(img, u1, np.sqrt(s1))
        r2 = gauss(img, u2, np.sqrt(s2))

        tempa = r1*w1
        tempb = r2*w2

        background[tempa > tempb] = 255
        background[tempa <= tempb] = 0

        o1[tempa > tempb] = 1
        o1[tempa <= tempb] = 0

        o2[tempa > tempb] = 1
        o2[tempa <= tempb] = 0

        delta1 = img-u1
        delta2 = img-u2

        # o1 = o1.astype(int)
        # background = o1*255
        outVid.write(background.astype("uint8"))
        # Maximization

        w1 = w1+(o1-w1)*alpha
        u1 = u1+o1*(alpha / (w1+0.001))*(delta1*o1)
        s1 = s1+o1*(alpha / (w1+0.001))*((delta1*o1)**2-s1)

        w2 = w2+(o2-w2)*alpha
        u2 = u2+o2*(alpha / (w2+0.001))*(delta2*o2)
        s2 = s2+o2*(alpha / (w2+0.001))*((delta2*o2)**2-s2)
    else:
        break
# outVid.release()
# cap.release()
cap.release()
outVid.release()
cv2.destroyAllWindows()

# for i in range(len(img)):
#     for j in range(len(img[0])):
#         x = img[i][j]  # pixel
#         r1 = gauss(x, u1[i][j], math.sqrt(s1[i][j]))
#         r2 = gauss(x, u2[i][j], math.sqrt(s2[i][j]))
#         back = 0  # it is not background
#         if(r1 > r2):
#             background[i][j] = 255  # it is background
#             o1[i][j] = 1
#         else:
#             o2[i][j] = 1
