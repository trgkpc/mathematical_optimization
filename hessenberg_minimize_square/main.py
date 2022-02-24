import numpy as np
import numpy.linalg as LA
from scipy.linalg import lstsq
import time

# scipyによって||Ax-b||の最小解を求める
def use_scipy(A, b):
    return lstsq(A, b)[0]

# 縦長ヘッセンベルグ行列について最小二乗解を求める
def mylib(A, b):
    A = np.array(A)
    b = np.array(b)
    m,n = A.shape
    for i in range(n):
        r = np.sqrt(A[i][i]**2 + A[i+1][i]**2)
        cos = A[i,i] / r
        sin = A[i+1,i] / r
        R = np.array([[cos, sin], [-sin, cos]])
        A[i:i+2] = R@A[i:i+2]
        b[i:i+2] = R@b[i:i+2]
        
    return lstsq(A, b)[0]

if __name__ == '__main__':
    # ランダムな縦長ヘッセンベルグ行列を作る
    np.random.seed(0)
    n = 300
    A = 10*np.random.random((n+1, n))
    for i in range(2,n+1):
        A[i, :i-1] = np.zeros(i-1)
    # ランダムなベクトルを作る    
    b = np.random.random(n+1)

    # 実行 & 時間計測
    def run(func):
        t0 = time.time()
        ret = func(A, b)
        tf = time.time()
        return (tf-t0),ret
    
    T,correct = run(use_scipy)
    t,ans = run(mylib)

    print("scipy:", T, LA.norm(b-A@correct))
    print("mylib:", t, LA.norm(b-A@ans))
    print("error:", LA.norm(correct-ans) / LA.norm(correct))

