import numpy as np
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt

np.random.seed(0)

n = 300
m = 10
A = np.random.random((n,n))
ans = np.random.random(n)
b = A@ans

eps = 1e-10
x = np.zeros(n)
r0 = b
r0norm = LA.norm(r0)
if r0norm == 0:
    exit()
V = np.array([r0 / r0norm])
H = np.zeros((1,0))
for k in range(m):
    H = np.vstack((H, np.zeros(H.shape[-1]))).T
    H = np.vstack((H, np.zeros(H.shape[-1]))).T

    w = A@V[k]
    for i in range(k+1):
        H[i][k] = V[i]@w
        w -= H[i][k] * V[i]
    H[k+1][k] = LA.norm(w)

    # TODO これを解く
    e1 = np.zeros(k+2)
    e1[0] = r0norm

    if abs(H[k+1][k]) < eps:
        break
    V = np.vstack((V, w / H[k+1][k]))
V = V.T

