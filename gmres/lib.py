import numpy as np
import numpy.linalg as LA

def TRsolve(A, b):
    x = np.zeros(A.shape[1])
    n = A.shape[1]
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
    eps1 = 1e-10
    eps2 = 1e-13

    # 初期解について処理
    r0 = b - A@x0
    r0norm = LA.norm(r0)
    if r0norm == 0:
        exit()

    # V,Hを構成
    V = np.array([r0 / r0norm])
    H = np.zeros((1, 0))
    error = r0norm
    for k in range(m):
        # H,eを拡大
        H = np.vstack((H, np.zeros(H.shape[-1]))).T
        H = np.vstack((H, np.zeros(H.shape[-1]))).T

        # 次の基底を生成
        w = A@V[k]
        for i in range(k+1):
            H[i][k] = V[i]@w
            w -= H[i][k] * V[i]
        h = LA.norm(w)
        H[k+1][k] = h
        V = np.vstack((V, w / h))
        
        # Hがランク落ちしている場合誤差0を達成出来るので打ち切って終了
        if abs(h) < eps1:
            break

        # 誤差を計算し、十分小さい場合は打ち切って終了
        sin = h / np.sqrt(H[k][k]**2 + h**2)
        error *= abs(sin)
        if error < eps2:
            break
        
    V = V[:-1].T

    e1 = np.zeros(H.shape[0])
    e1[0] = r0norm
    y = hessenberg_lstsq(H, e1)

    return V@y + x0


