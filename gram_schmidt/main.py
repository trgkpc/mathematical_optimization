import numpy as np
import numpy.linalg as LA

def normalize(x):
    l = LA.norm(x)
    if l == 0:
        return np.zeros(x.shape)
    return x / l


def QRdecomposition(A):
    B = np.array(A).T
    n,m = A.shape
    Q = np.zeros((n,m))
    R = np.zeros((m,m))
    for i in range(m):
        e = normalize(B[i])
        Q[:,i] = e
        
        r = B[i:]@e
        R[i,i:] = r
        B[i+1:] -= np.c_[r[1:]] * e

    return Q,R

if __name__ == '__main__':
    A = np.random.random((5, 4))
    A /= LA.norm(A)

    Q,R = LA.qr(A)
    q,r = QRdecomposition(A)
    
    QTq = Q.T@q
    error = QTq - np.diag([np.sign(QTq[i][i]) for i in range(QTq.shape[0])])
    print(LA.norm(error))
    print(LA.norm(A - q@r))

