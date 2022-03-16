from sympy import Rational
import numpy as np
import numpy.linalg as LA
import time

##### 設定 #####
# 関数を定める
def f_impl(x):
    x1, x2 = x
    return Rational(1, 2) * (x1**4) - 2 * (x1**2) * x2 + 4 * (x2**2) + 8 * x1 + 8 * x2

##### 最適化を行う関数 #####
def optimize(F,
             x0=np.array([3, 1]),  # 初期解
             eps=1e-7,             # 停止条件
             max_iter_num=1000,    # イテレーション回数
             ):
    x = np.array(x0, dtype=np.float64)
    n = len(x)
    # パラメータ
    alpha = 1.0
    gamma = 1.0 + 2 / n
    rho = 0.75 - 1 / n
    sigma = 1.0 - 1 / n
    
    simplex = _initialize_simplex(n, x0)
    obj_list = np.array(list(map(F, simplex)))
    simplex, obj_list = _sort_by_objective_value(simplex, obj_list)
    x_g = np.mean(simplex[:-1], axis=0) # 最悪点を除いた値の重心
    
    best_obj = min(obj_list)
    prev_obj = best_obj
    
    for iter_num in range(max_iter_num):
        f_best = obj_list[0]
        f_second_worst = obj_list[-2]
        f_worst = obj_list[-1]
        x_worst = simplex[-1]
        
        # Reflection
        x_r = x_g + alpha * (x_g - x_worst)
        f_r = F(x_r)
        
        if f_best <= f_r < f_second_worst:
            obj_list, simplex, x_g = _update(obj_list, simplex, x_r, f_r)
        elif f_r < f_best:
            # Expansion
            x_e = x_g + gamma * (x_r - x_g)
            f_e = F(x_e)
            if f_e < f_r:
                obj_list, simplex, x_g = _update(obj_list, simplex, x_e, f_e)
            else:
                obj_list, simplex, x_g = _update(obj_list, simplex, x_r, f_r)
        else:
            # Contraction
            if f_r <= f_worst:
                x_c = x_g + rho * (x_r - x_g)
            else:
                x_c = x_g + rho * (x_worst - x_g)
            f_c = F(x_c)
            if f_c < f_worst:
                obj_list, simplex, x_g = _update(obj_list, simplex, x_c, f_c)
            else:
                # Shrink
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                obj_list[1:] = np.array(list(map(F, simplex[1:])))
                simplex, obj_list = _sort_by_objective_value(simplex, obj_list)
                x_g = np.mean(simplex[:-1], axis=0)
        
        best_obj = min(obj_list)
        best_x = simplex[np.argmin(obj_list)]

        if best_obj < prev_obj:
            if abs(prev_obj - best_obj) <= eps:
              break
            prev_obj = best_obj
    
    return {"iter_num": iter_num, "x": best_x, "f": best_obj}

def _initialize_simplex(n, x0):
    # 単体の初期化
    # https://stackoverflow.com/questions/17928010/choosing-the-initial-simplex-in-the-nelder-mead-optimization-algorithm/19282873#19282873
    h = lambda x: 0.00025 if x == 0.0 else 0.05
    simplex = np.array(
        [x0] + [x0 + h(x0[i])*e for i,e in enumerate(np.identity(n))])
    return simplex

def _sort_by_objective_value(x, ordering):
    # 目的関数の値を比較関数としてソート
    indices = np.argsort(ordering)
    return x[indices], ordering[indices]

def _update(obj_list, simplex, x_updated, f_updated):
    # 最悪点を削除して更新
    simplex[-1] = x_updated
    obj_list[-1] = f_updated
    simplex, obj_list = _sort_by_objective_value(simplex, obj_list)
    x_g = np.mean(simplex[:-1], axis=0)
    return obj_list, simplex, x_g

if __name__ == '__main__':
    # 関数を作成
    def F(x): return f_impl(x)

    t0 = time.time()
    result = optimize(F)
    t = time.time() - t0
    print("===== result =====")
    print("iter_num :", result["iter_num"])
    print("x        :", result["x"].tolist())
    print("f        :", result["f"])
    print("t        :", t)

