import numpy as np
import numpy.linalg as LA
from scipy.linalg import lstsq

def GMRES(A, b, x0, m):
    eps = 1e-10
    r0 = b - A@x0
    r0norm = LA.norm(r0)
    if r0norm == 0:
        exit()
    V = np.array([r0 / r0norm])
    H = np.zeros((1, 0))
    for k in range(m):
        H = np.vstack((H, np.zeros(H.shape[-1]))).T
        H = np.vstack((H, np.zeros(H.shape[-1]))).T

        w = A@V[k]
        for i in range(k+1):
            H[i][k] = V[i]@w
            w -= H[i][k] * V[i]
        H[k+1][k] = LA.norm(w)

        e1 = np.zeros(k+2)
        e1[0] = r0norm
        y = lstsq(H, e1)[0]

        V = np.vstack((V, w / H[k+1][k]))
        if abs(H[k+1][k]) < eps:
            break
    V = V[:-1].T

    return V@y + x0

