import numpy as np
import time
from solve import simplex_method
from scipy import optimize

inf = float('inf')
eps = 1e-6

# debug出力
def debug_print(string):
    print(f'\033[37m\033[41m[[WARNING]] {string}\033[0m')

# 実行可能かつ有界
def get_LP(m, n):
    x = np.random.rand(n)
    y = np.random.rand(m)
    A = np.random.randn(m, n) 
    while np.min(A@x) >= -eps:
        A = np.random.randn(m, n)
    b = A @ x
    c = A.T @ y
    return A, b, c

# 実行不可能
def get_infeasible(m, n):
    y = np.random.rand(m)
    b = np.random.randn(m)
    while y@b > -eps:
        b = np.random.randn(m)
    
    A = np.random.randn(n, m)
    A *= np.sign(A@y)[:, None]
    c = np.random.randn(n)
    return A.T, b, c

# 非有界(実行可能)
def get_unbounded(m, n):
    A, b, c = get_infeasible(n, m)
    return -A.T, -c, -b

LP_types = [get_LP, get_infeasible, get_unbounded]

def main(prob): # prob:0,1,2
    m =  20 # 制約式の数
    n = 100 # 問題の次元数
    
    A, b, c = LP_types[prob](m, n)
    
    start_time = time.time()
    ans = simplex_method(A, b, c)
    end_time = time.time()

    # ライブラリを用いて解いてみる
    print(f'result(calc):{ans}')
    result = optimize.linprog(-c, A, b, bounds=(0, None))
    true_result = -result["fun"]
    if not result["success"]:
        if "unbounded" in result["message"]:
            true_result = inf
        elif "infeasible" in result["message"]:
            true_result = -inf
        else:
            debug_print("unknown error for scipy.optimize.linprog")
            print(result)
    print(f'result(true):{true_result}')
    if ans != true_result and abs(ans-true_result) > eps:
        debug_print("Different result!")
        print(result)
        np.savez('debug', A, b, c)

    print(f'calculation time:{end_time - start_time}[sec]')

if __name__ == '__main__':
    for i in range(3):
        main(i)

