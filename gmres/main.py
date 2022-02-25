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

def TRsolve(A, b):
    x = np.zeros(A.shape[1])
    for i in range(n-1,-1,-1):
        x[i] = (b[i] - A[i]@x) / A[i][i]
    return x

def hessenberg_lstsq(A, b):
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
    return TRsolve(A, b)

def GMRES(A, b, x0, m):
    eps = 1e-10

    # 初期解について処理
    r0 = b - A@x0
    r0norm = LA.norm(r0)
    if r0norm == 0:
        exit()

    # V,Hを構成
    V = np.array([r0 / r0norm])
    H = np.zeros((1, 0))
    e1 = np.array([r0norm])
    for k in range(m):
        # H,eを拡大
        H = np.vstack((H, np.zeros(H.shape[-1]))).T
        H = np.vstack((H, np.zeros(H.shape[-1]))).T
        e1 = np.append(e1,0.)

        # 次の基底を生成
        w = A@V[k]
        for i in range(k+1):
            H[i][k] = V[i]@w
            w -= H[i][k] * V[i]
        h = LA.norm(w)
        H[k+1][k] = h
        V = np.vstack((V, w / h))
        
        # ギブンス変換により残差を見積もる
        p = H[k][k]
        q = h
        r = np.sqrt(p**2 + q**2)
        cos = p / r
        sin = q / r
        R = np.array([[cos, sin], [-sin, cos]])
        e1[-2:] = R@e1[-2:]

        # Hがランク落ちしている場合もしくはabs(e1[-1])が0の場合、誤差0を達成出来るので打ち切って終了
        if abs(h) < eps or abs(e1[-1]) < eps:
            break
        
    V = V[:-1].T

    e1 = np.zeros(e1.shape)
    e1[0] = r0norm
    y = hessenberg_lstsq(H, e1)

    return V@y + x0


if __name__ == '__main__':
    np.random.seed(0)

    n = 100
    m = n
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
    print("raw:", T, LA.norm(b-A@X), LA.norm(X-ans) / LA.norm(ans))
    print("new:", t, LA.norm(b-A@x), LA.norm(x-ans) / LA.norm(ans))
