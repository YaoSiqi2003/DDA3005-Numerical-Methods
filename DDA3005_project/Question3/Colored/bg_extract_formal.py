from numpy.linalg import norm
import numpy as np
import cv2
import time

def read_video_frames(file_path, is_colored = False, frame_portion=0.5, resize_constant=None):
    
    # Initialize of getting video content
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = int(1/frame_portion)
    print("step_size:", step_size)
    if resize_constant is None:
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        frame_size = resize_constant
    count = 0
    print("Resolution:", frame_size[0], "x", frame_size[1])
    
    A_red, A_green, A_blue = np.array([]), np.array([]), np.array([])

    # Read a portion of the video frames into matrix A/ three matrices A_red, A_green, A_blue
    while count<=total_frames:
        if count % step_size == 0:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, frame_size)
        
            
            red, green, blue = frame[:,:,0], frame[:,:,1], frame[:,:,2]
            channels = [red, green, blue]

            for idx, channel in enumerate(channels):
                m = channel.ravel(order="F")
                if idx == 0:  # Red channel
                    A_red = np.column_stack([A_red, m]) if A_red.size else m
                elif idx == 1:  # Green channel
                    A_green = np.column_stack([A_green, m]) if A_green.size else m
                elif idx == 2:  # Blue channel
                    A_blue = np.column_stack([A_blue, m]) if A_blue.size else m
        count += 1
    cap.release()

    return (A_red, A_green, A_blue) 




# METHOD 1
def power_iteration(A):
    m = A.shape[0]
    n = A.shape[1]
    # construct the symmetric matrix A_0
    A_0 = np.zeros((m + n, m + n))
    A_0[0: n, n: n + m] = A.T
    A_0[n: n + m, 0: n] = A
    # start the power iteration on A_0

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
    
    u = x[n:n+m]
    v = x[0:n]
    B = sigma * v[0] * u
    print("Convergence Step:", count_iter)

    return B








# METHOD 2
def svd_package(A):
    # Perform SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Extract the largest singular value and corresponding singular vectors
    sigma = s[0]
    u = U[:, 0]
    v = Vt[0, :]
    # Reconstruct the channel
    B = sigma * v[0] * u
    return B







def main(main_folder, file_name, complete_path, method, resize_constant):
    # Init
    cap = cv2.VideoCapture(complete_path)

    # Start
    print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    A_red, A_green, A_blue = read_video_frames(complete_path, frame_portion=1, resize_constant=resize_constant)
    print("Frames taken:", A_red.shape[1])



    if method == 1: # power iteration
        start = time.time()
        B_red = power_iteration(A_red).reshape((resize_constant[1], resize_constant[0]), order = 'F')
        B_green = power_iteration(A_green).reshape((resize_constant[1], resize_constant[0]), order = 'F')
        B_blue = power_iteration(A_blue).reshape((resize_constant[1], resize_constant[0]), order = 'F')
        print(f"Time taken: {time.time() - start}s")
        B = cv2.merge([B_red, B_green, B_blue])
        B_normalized = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
        B_uint8 = B_normalized.astype(np.uint8)
        cv2.imwrite(f"{main_folder}output/method1/{file_name}.jpg", B_uint8)
    
    elif method == 2: # SVD package
        start = time.time()
        B_red = svd_package(A_red).reshape((resize_constant[1], resize_constant[0]), order = 'F')
        B_green = svd_package(A_green).reshape((resize_constant[1], resize_constant[0]), order = 'F')
        B_blue = svd_package(A_blue).reshape((resize_constant[1], resize_constant[0]), order = 'F')
        print(f"Time taken: {time.time() - start}s")
        B = cv2.merge([B_red, B_green, B_blue])
        B_normalized = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
        B_uint8 = B_normalized.astype(np.uint8)
        cv2.imwrite(f"{main_folder}output/method2/{file_name}.jpg", B_uint8)
        return


    








main_folder = "/Users/archerliu/Desktop/College Stuff/Y2T1/DDA3005/Course Project/"
sub_folder = "test_videos/2560_1440/"
file_name = "street"
file_extension = ".mp4"
complete_path = main_folder + sub_folder + file_name + file_extension

main(main_folder, file_name, complete_path, method=2, resize_constant=(960, 540))
