import numpy as np
from scipy import optimize

inf = float('inf')
eps = 1e-6

# 単体法
## debug出力
def debug_print(string):
    print(f'\033[37m\033[41m[[WARNING]] {string}\033[0m')

## listから与えられた値を探す
def find(lis, target):
    for i in range(len(lis)):
        if lis[i] == target:
            return i
    debug_print("find(lis, target): Target not found.")
    return None

## 実行可能解が与えられた上での単体法
def pivoting(A, b, c, I, J, xlist, ylist):
    n = len(c)
    m = len(b)
    notI = [i for i in range(I)] + [i for i in range(I+1,n)]
    notJ = [j for j in range(J)] + [j for j in range(J+1,m)]

    # x[I]とy[j]の入れ替え
    AJI_inv = 1.0 / A[J][I]
    for i in notI:
        A[J][i] *= AJI_inv
    b[J] *= AJI_inv
    A[J][I] = AJI_inv

    # Aの各行に入れ替えを反映
    minusAJI = -AJI_inv
    for j in notJ:
        AjI = A[j][I]
        for i in notI:
            A[j][i] -= AjI * A[J][i]
        b[j] -= AjI * b[J]
        A[j][I] *= minusAJI

    # cに入れ替えを反映
    cI = c[I]
    for i in notI:
        c[i] -= cI * A[J][i]
    c[I] *= minusAJI

    # 変数名を保持
    xlist[I],ylist[J] = ylist[J],xlist[I]
    return cI * b[J]
    
def basic_simplex_method_impl(A, b, c, xlist=None, ylist=None):
    z = 0.0
    n = len(c)
    m = len(b)

    # 変数名を保持
    if xlist == None:
        xlist = [i for i in range(n)]
    if ylist == None:
        ylist = [n+i for i in range(m)]
    def get_xopt():
        xopt = np.zeros(n)
        for j in range(m):
            if ylist[j] < n:
                xopt[ylist[j]] = b[j]
        return xopt

    itenum = 0
    eps = 1e-3
    while True:
        itenum += 1
        xmax = 0.0
        for i in range(n):
            if c[i] > eps:
                xmax = inf
                J = m
                for j in range(m):
                    a = A[j][i]
                    if a > eps:
                        xmax_ = b[j] / a
                        if xmax_ < xmax:
                            xmax = xmax_
                            J = j
                if xmax > eps:
                    break
                else:
                    xmax = 0.0
        if xmax <= eps:
            break
        elif xmax == inf:
            z = inf
            break
        
        I = i

        # 添字入れ替え
        z += pivoting(A, b, c, I, J, xlist, ylist)

    return [get_xopt(), z]

def basic_simplex_method(A, b, c):
    return basic_simplex_method_impl(np.array(A), np.array(b), np.array(c))[-1]

def two_step_simplex_method(A_, b_, c_):
    n = len(c_)
    m = len(b_)
    # 実行可能な辞書を作る
    ## 補助問題の辞書を作る
    A = np.hstack([A_, [[-1] for i in range(m)]])
    b = np.array(b_)
    c = np.zeros(n+1)
    xlist = [i for i in range(n+1)]
    ylist = [n+i for i in range(m)]
    c[-1] = -1.0
    ## pivotingにより実行可能にする
    J = np.argmin(b)
    z = pivoting(A, b, c, n, J, xlist, ylist)

    ## 人工変数を最小化する
    x0, dz = basic_simplex_method_impl(A, b, c, xlist, ylist)
    z += dz

    ## 人工変数が負だったら終わり
    if z < -eps or np.min(b) < -eps:
        debug_print("Not executable problem")
        return -inf

    # 補助問題の辞書を参考に主問題の辞書を作る
    ## 人工変数の削除
    a = find(xlist+ylist, n) # 人工変数を見つける
    if a > n: # 基底変数だったら非基底変数に取り直す
        a_ = a-(n+1)
        i = np.argmax(np.abs(A[a_]))
        pivoting(A, b, c, i, a_, xlist, ylist)
        a = i
    A = np.hstack([A[:,:a], A[:,a+1:]])
    lis = xlist + ylist
    del lis[a]

    ## 主問題の目的関数を召喚する
    z0 = 0
    newc = np.zeros(n)
    for i in range(n): # 非基底変数のサーチ
        l = lis[i]
        if l < n: # x_0~x_{n-1}だったら
            newc[i] = c_[l]
    for j in range(m): # 基底変数のサーチ
        l = lis[n+j]
        if l < n: # x_0~x_{n-1}だったら
            z0 += c_[l] * b[j]
            newc -= c_[l] * A[j]
    return z0 + basic_simplex_method_impl(A, b, newc)[-1]

def simplex_method(A, b, c):
    return two_step_simplex_method(A, b, c)

