import numpy as np
import cv2
# import shelve


class AlphaFilter:
    def __init__(self, estimate):
        self.x = float(estimate)
        self.n = 1

    def apply(self, Zn):
        self.n += 1
        self.x = self.x-(self.x-Zn)/self.n
        return self.x

    def getEstimate(self):
        return self.x


# f = shelve.open('Task2')
cap = cv2.VideoCapture('rouen_video.mp4.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outVid = cv2.VideoWriter('Task2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                         (frame_width, frame_height), 0)
initialized = False
alpha = 0
L = 0
while(cap.isOpened()):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    print(L)
    L += 1
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        break
    if(not initialized):
        initialized = True
        out = []
        outVid.write(gray)
        for i in gray:
            row = []
            for j in i:
                row.append(AlphaFilter(j))
            out.append(row)
    else:
        result = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.uint8)
        for i in range(len(gray)):
            for j in range(len(gray[0])):
                out[i][j].apply(gray[i][j])
                result[i][j] = out[i][j].getEstimate()
        outVid.write(result)


# for i in range(len(out)):
#     for j in range(len(out[0])):
#         out[i][j] = out[i][j].getEstimate()

outVid.release()
cap.release()
finalResult = np.array(out, dtype=np.uint8)
cv2.imshow('frame', finalResult)
cv2.imwrite('background.jpg', np.array(out, dtype=np.uint8))
# f['result'] = np.array(f['result'], dtype=np.uint8)
# cv2.imshow('frame', f['result'])
cv2.waitKey(0)
# f.close()
