import numpy as np
import numpy.linalg as LA
import time

def TRinv(A):
    B = np.zeros(A.shape)
    for j in range(n):
        B[j][j] = 1 / A[j][j]
        for i in range(j-1,-1,-1):
            a = A[i,i+1:j+1]
            b = B[i+1:j+1, j]
            B[i][j] = -B[i][i] * (a@b)
    return B


if __name__ == '__main__':
    np.random.seed(2)
    n = 200
    A = 3 * np.random.random((n,n))
    for i in range(1,n):
        A[i,:i] = np.zeros(i)
    
    def run(func):
        t0 = time.time()
        B = func(A)
        tf = time.time()
        return (tf-t0),B

    T,correct = run(LA.inv)
    t,ans = run(TRinv)

    print("LA.inv:", T)
    print("TRinv: ", t)
    print(LA.norm(correct-ans) / LA.norm(correct))


