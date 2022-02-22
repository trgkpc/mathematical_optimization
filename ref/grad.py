from sympy import Symbol,diff,Rational,lambdify
import numpy as np
import time

def f_impl(x):
    c = Rational(1,2)
    A = np.array([[1,0,0],[0,1,c],[0,c,1]])
    return x@A@x

def get_f(n):
    X = np.array([Symbol(f"x{i+1}") for i in range(3)])

    F = f_impl(X)
    ans = [F]

    if n >= 1:
        GRAD = np.array([diff(F, x) for x in X])
        ans.append(GRAD)

    if n >= 2:
        HESS = np.array([np.array([diff(func, x) for x in X]) for func in GRAD])
        ans.append(HESS)
    
    return ans

F, G, H = get_f(2)

print(F)
print(G.tolist())
print(H.tolist())
