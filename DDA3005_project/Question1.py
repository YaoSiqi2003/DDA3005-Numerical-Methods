import numpy as np

def householder_reflection(v): #This function performs Householder reflections.
    u = v / (v[0] + np.copysign(np.linalg.norm(v), v[0]))
    u[0] = 1
    H = np.eye(len(v)) - (2 / np.dot(u, u)) * np.outer(u, u)
    return H

def bidiagonalize(A): #This function bidiagonalizes input matrix A.
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    
    for i in range(min(m, n)):
        
        # Left Householder transformation
        x = A[i:, i]
        H = np.eye(m)
        H[i:, i:] = householder_reflection(x)
        A = np.dot(H, A)
        U = np.dot(U, H)
        
        if i < n - 2:
            # Right Householder transformation
            x = A[i, i+1:]
            H = np.eye(n)
            H[i+1:, i+1:] = householder_reflection(x).T
            A = np.dot(A, H)
            V = np.dot(V, H)
    
    B = A

    return U, B, V.T #A=U*B*V^T where B is bidiagonal matrix.


A_example = np.array([[1,2,4], [4,5,6],[7,8,9]])

U, B, Vt = bidiagonalize(A_example)
print('U=',U)
print('B=',B)
print('V=',Vt)
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')




def phase_II_A(B, tol=1e-12):#This function performs QR algorithm with Wilkinson shift for a bidiagonal matrix B.
    if B.shape[0] > B.shape[1]:
        B = B[:B.shape[1], :]

    elif B.shape[1] > B.shape[0]:
        B = B[:, :B.shape[0]]

    
    #Performs QR algorithm with Wilkinson shift for a bidiagonal matrix B.
    n = B.shape[1]
    V = np.eye(n)
    T = B.T @ B  # Convert B to a symmetric tridiagonal matrix B^T B
    for m in range(n-1, 0, -1):
        while np.abs(T[m, m-1]) > tol:
            # Apply Wilkinson's shift
            d = (T[m-1, m-1] - T[m, m]) / 2
            t = T[m, m] - (T[m, m-1]**2) / (d + np.sign(d) * np.sqrt(d**2 + T[m, m-1]**2))
            Q, R = np.linalg.qr(T[:m+1, :m+1] - t * np.eye(m+1))
            T[:m+1, :m+1] = R @ Q + t * np.eye(m+1)
            V[:, :m+1] = V[:, :m+1] @ Q
        if m < n-1:
            T[m+1, m] = T[m, m+1] = 0

    # Extract eigenvalues
    eigenvalues = np.diag(T)
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]


    # Form the singular values from the square roots of the eigenvalues

    S = np.sqrt(eigenvalues)

    # Compute U using the formula U = B * V * diag(1/S)
    U1 = B @ V @ np.diag(1/S)
    
    # Make sure U is orthogonal
    U1, _ = np.linalg.qr(U)

    epsilon = 1e-10

    S_svd = np.diag(S)

    S_inv = np.diag(np.diag(1 / S_svd))

    a = A_example.shape[0]
    b = A_example.shape[1]
    if a < b:

        newMatrix = Vt.T[b-a:, b-a:]
 
  
        U_svd = A_example @ (newMatrix @ V) @S_inv
        
    else:
        U_svd = A_example @ (Vt.T @ V) @S_inv

    return U_svd, S_svd, V.T @ Vt




# Perform QR algorithm to compute the SVD
U_svd, S_svd, V_svd = phase_II_A(B)


# Print the results
print("SVD of original matrix A_example through phase II-A:\nU_svd:")
print(U_svd)
print("\nS_svd:")
print(S_svd)
print("\nVt_svd:")
print(V_svd.T)  # Transpose V_svd to get Vt_svd
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')



def phase_II_B_fixed(B, U_init, Vt_init, A): #Perform Phase II-B of the SVD computation which uses QR factorization and
    #Cholesky decomposition for bidiagonal matrix B.
    m, n = B.shape
    U = U_init.copy()
    Vt = Vt_init.copy()
    X = B.copy()
    tol = 1e-12  # Tolerance for deflation

    # Iterate until convergence
    while not np.isclose(X[0, 1], 0, atol=tol) and not np.isclose(X[-2, -1], 0, atol=tol):
        # QR factorization of X^T
        Q, R = np.linalg.qr(X.T)

        # Cholesky decomposition of R * R^T
        try:
            L = np.linalg.cholesky(R @ R.T).T
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, B is not full rank
            return None, None, None

        # Update X to the next bidiagonal matrix
        X = L.T

        # Update U and V
        U = U @ Q.T
        Vt = Vt @ Q

    # Once X is diagonal, the singular values are on the diagonal of X
    S = np.diag(np.abs(np.diag(X)))

    # Normalize U and Vt so that the singular values are non-negative
    for i in range(min(m, n)):
        if np.diag(X)[i] < 0:
            U[:, i] = -U[:, i]
            Vt[i, :] = -Vt[i, :]

    # Compute the final U and Vt for the original matrix A
    Sigma_inv = np.diag(1/np.diag(S))
    U_final = A @ Vt.T @ Sigma_inv

    # Ensure orthogonality of U_final
    U_final, _ = np.linalg.qr(U_final)

    return U_final, S, Vt

# Apply the fixed phase II-B function to obtain the SVD
U_phase_II_B_fixed, S_phase_II_B_fixed, Vt_phase_II_B_fixed = phase_II_B_fixed(B, U, Vt, A_example)

if U_phase_II_B_fixed is not None and S_phase_II_B_fixed is not None and Vt_phase_II_B_fixed is not None:
    print("SVD of original matrix A_example through phase II-B:\nU_phase_II_B:")
    print(U_phase_II_B_fixed)
    print("\nS_phase_II_B:")
    print(S_phase_II_B_fixed)
    print("\nVt_phase_II_B:")
    print(Vt_phase_II_B_fixed)  # Transpose V_phase_II_B to get Vt_phase_II_B
else:
    print("The matrix B is not full rank, Cholesky decomposition failed.")