import numpy as np
import numpy.linalg as LA
from scipy.linalg import lstsq
import time
import matplotlib.pyplot as plt


def rawGMRES(A, b, x0, m):
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


if __name__ == '__main__':
    np.random.seed(0)

    n = 300
    m = 300
    A = np.random.random((n, n))
    ans = np.random.random(n)
    b = A@ans

    # 実行 & 時間計測
    def run(func):
        t0 = time.time()
        ret = func(A, b, np.zeros(n), m)
        tf = time.time()
        return (tf-t0), ret

    T, X = run(rawGMRES)
    t, x = run(GMRES)
    print("raw:", T, LA.norm(X-ans) / LA.norm(ans))
    print("new:", t, LA.norm(x-ans) / LA.norm(ans))
