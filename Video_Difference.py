import numpy as np
import cv2

cap = cv2.VideoCapture('rouen_video.mp4.avi')
# Define the codec and create VideoWriter object
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('Task1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                      (frame_width, frame_height), 0)
initialized = False
prev = 0


def imgDiff(frame1, frame2):
    result = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.uint8)
    for i in range(len(frame1)):
        for j in range(len(frame1[0])):
            if(frame2[i][j] > frame1[i][j]):
                result[i][j] = frame2[i][j]-frame1[i][j]
            else:
                result[i][j] = frame1[i][j]-frame2[i][j]
    return result


L = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    print(L)
    L += 1
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        break
    if(not initialized):
        prev = gray
        initialized = True
        continue

    # write the flipped frame
    cur = imgDiff(prev, gray)
    # cv2.imshow('frame', cur)
    out.write(cur)
    prev = gray

    # cv2.imshow('frame', gray)

cap.release()
out.release()
cv2.destroyAllWindows()
