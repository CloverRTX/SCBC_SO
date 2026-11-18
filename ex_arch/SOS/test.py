import sympy as sp

if __name__ == "__main__":
    x1 = sp.symbols('x1')
    x2 = sp.symbols('x2')
    P = x1 + x2
    f1 = 2*x2
    f2 = 3*x1

    print(P.subs(x1, f1).subs(x2, f2))
    print(P.xreplace({x1:f1, x2:f2}))
    print(P.subs([(x1, f1), (x2, f2)]))

    pass
