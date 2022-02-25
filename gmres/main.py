import numpy as np
import numpy.linalg as LA
import time

from old import GMRES as old
from lib import GMRES as lib

if __name__ == '__main__':
    np.random.seed(0)

    n = 100
    m = 80
    A = np.random.random((n, n))
    A += 10*np.identity(n)
    ans = np.random.random(n)
    b = A@ans

    # 実行 & 時間計測
    def run(func):
        t0 = time.time()
        ret = func(A, b, np.zeros(n), m)
        tf = time.time()
        return (tf-t0), ret

    T, X = run(old)
    t, x = run(lib)
    print("raw:", T, LA.norm(b-A@X), LA.norm(X-ans) / LA.norm(ans))
    print("new:", t, LA.norm(b-A@x), LA.norm(x-ans) / LA.norm(ans))
