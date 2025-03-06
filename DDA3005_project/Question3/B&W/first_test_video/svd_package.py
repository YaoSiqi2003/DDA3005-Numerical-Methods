import numpy as np
import cv2
import time
from scipy.sparse.linalg import svds

# form matrix A
A = np.array([])
cap = cv2.VideoCapture('walking.mp4')
if not cap.isOpened():
    print("Error opening video file")
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (240, 135))
        # use count to rescale the frame to accelerate
        # the construction of A
        ### we pick part of the frames only
        if count % 1 == 0:
            #use gray scale to save storage
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m = gray_frame.ravel(order="F")
            if A.size == 0:
                A = m
            else:
                A = np.column_stack((A, m))  
        count += 1
    else:
        break
A = A.astype(float)

start_time = time.time()
U, s, Vt = svds(A)

end_time = time.time()
t = end_time - start_time
v = Vt[-1, :]

u = U[:,-1]

sigma = s[-1]
b = sigma * v[0] * u
B = b.reshape((135, 240), order = 'F')
B = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
B = cv2.resize(B, (640,360))
print(B)
print('runtime:', t)
cv2.imwrite("output.png", B)