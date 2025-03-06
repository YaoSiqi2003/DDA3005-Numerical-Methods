import numpy as np
from numpy.linalg import norm
import cv2
import time

# form matrix A
A = np.array([])
cap = cv2.VideoCapture('walking.mp4')
if not cap.isOpened():
    print("Error opening video file")
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # use count to rescale the frame to accelerate
        # the construction of A
        ### we pick part of the frames only
        if count % 1 == 0:
            frame = cv2.resize(frame, (240, 135))
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

def power_method(A):
    m = A.shape[0]
    n = A.shape[1]
    # construct the symmetric matrix A_0
    A_0 = np.zeros((m + n, m + n))
    A_0[0: n, n: n + m] = A.T
    A_0[n: n + m, 0: n] = A
    # start the power iteration on A_0 -------

    #construct x_0 
    x = np.random.rand(m + n)
    x = x / norm(x)
    # this tolerance here is also crucial
    tol = 0.000001
    error = 100
    sigma = 0
    prev_sigma = 0
    count_iter = 0
    while error > tol:
        x = A_0 @ x
        x = x / norm(x)
        sigma = x.T @ A_0 @ x
        
        if count_iter == 0:
            prev_sigma = sigma
        else:
            error = abs(prev_sigma - sigma)
            prev_sigma = sigma
        
        count_iter += 1
    
    return sigma, x[n:n+m], x[0:n], count_iter

start_time = time.time()
sigma, u, v, iter= power_method(A)
end_time = time.time()
t = end_time - start_time
b = sigma * v[0] * u
B = b.reshape((135, 240), order = 'F')
# we need to normalize this matrix B
B = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
B = cv2.resize(B, (640,360))
print(iter)
print('runtime:', t)
print(B)
cv2.imwrite('output.png', B)
