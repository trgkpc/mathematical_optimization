from sympy import Symbol,diff,Rational,lambdify
import numpy as np
import time

##### 設定 #####
## 関数を定める
def f_impl(x):
    x1, x2 = x
    return Rational(1, 2) * (x1**4) - 2 * (x1**2) * x2 + 4 * (x2**2) + 8 * x1 + 8 * x2

## パラメタ
# 初期解
x0 = np.array([3, 1])
# 停止条件
eps = 1e-7
# イテレーション回数
max_iter_num = 1000
inner_max_iter = 20
# アルミホ条件のパラメタ
armijo = 0.01
# 直線探索におけるバックトラックの戻り率
alpha = 0.6

##### sympyを利用して導関数を計算 #####
def get_f(n):
    X = np.array([Symbol(f"x{i+1}") for i in range(2)])

    F = f_impl(X)
    ans = [F]

    if n >= 1:
        GRAD = np.array([diff(F, x) for x in X])
        ans.append(GRAD)

    if n >= 2:
        HESS = np.array([np.array([diff(func, x) for x in X]) for func in GRAD])
        ans.append(HESS)

    return [lambdify(X, f, "numpy") for f in ans]

## 関数を定義
F0, G0 = get_f(1)
F = lambda x:F0(x[0], x[1])
G = lambda x:np.array(G0(x[0], x[1]), dtype=np.float64)

##### 最適化を実施 #####
x = np.array(x0, dtype=np.float64)
for iter_num in range(max_iter_num):
    # 探索方向を決定
    f = F(x)
    grad = G(x)
    d = -grad
    
    # 停止条件を満たす場合終了する
    if d@d < eps*eps:
        break
    
    # 直線探索を行う
    for _ in range(inner_max_iter):
        # アルミホ条件を満たしたら直線探索を終了
        if F(x + d) <= f + armijo * (grad@d):
            break
        else:
            d *= alpha
    else:
        # 更新量が小さいため終了する
        break
    
    # 更新し次のステップに進む
    x += d

print("===== result =====")
print("iter_num: ", iter_num)
print("x: ", x.tolist())
print("f: ", F(x))

