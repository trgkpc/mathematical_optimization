import numpy as np
import numpy.linalg as LA
from scipy.linalg import lstsq
import time

def use_scipy(A, b):
    return lstsq(A, b)[0]

def mylib(A, b):
    A = np.array(A)
    b = np.array(b)
    m,n = A.shape
    print(m,n)
    for i in range(m):
        r = np.sqrt(A[i][i]**2 + A[i+1][i]**2)
        cos = A[i][i] / r
        sin = A[i+1][i] / r
        R = np.array([[cos, sin], [-sin, cos]])
        print(A[:3,:3])
        A[i:i+2] = R@A[i:i+2]
        print(A[:3,:3])

        b = R.T@b
        
    print(A)
    return lstsq(A, b)[0]

if __name__ == '__main__':
    np.random.seed(0)
    n = 300
    A = 10*np.random.random((n+1, n))
    for i in range(2,n+1):
        A[i, :i-1] = np.zeros(i-1)
    
    b = np.random.random(n+1)

    def run(func):
        t0 = time.time()
        ret = func(A, b)
        tf = time.time()
        return (tf-t0),ret
    
    def use_numpy(A, b):
        return LA.pinv(A)@b

    T,correct = run(use_scipy)
    t,ans = run(mylib)

    print("scipy:", T, LA.norm(b-A@correct))
    print("mylib:", t, LA.norm(b-A@ans))
    print(LA.norm(correct-ans) / LA.norm(correct))

