import sympy as sp
from itertools import product
from functools import reduce
import numpy as np




class Polynomial():

    def __init__(self, var_dim, deg, para_index):


        self.var_dim = var_dim


        self.var_sym = sp.symbols(['x{}'.format(i + 1) for i in range(var_dim)])


        self.deg = deg


        self.para_index = para_index


        self.poly, self.parameters, self.terms \
            = self.polynomial(self.para_index, self.deg)

        self.RealExpr = None


    def substitute(self, points):

        funcs = [sp.lambdify(self.var_sym, e, 'numpy')
                 for e in self.terms]

        lst = [points[:, i] for i in range(self.var_dim)]

        results = [f(*lst) for f in funcs]
        results[0] = np.ones(points.shape[0])

        matrix = np.column_stack(results)

        # print(matrix.shape)  # (100, 6)
        return matrix, self.parameters

    def show(self):
        print(self.poly)

    def polynomial(self, para_index=0, deg=2):



        parameters = []

        terms = []


        x_sym = self.var_sym

        exponents = list(product(range(deg + 1), repeat=self.var_dim))  # Generate all possible combinations of indices.
        exponents = [e for e in exponents if sum(e) <= deg]  # Remove items with a count greater than deg.
        poly = 0
        for e in exponents:  # Generate all items.
            parameters.append(sp.symbols('parameter' + str(para_index)))
            para_index += 1
            terms.append(reduce(lambda a, b: a * b, [x_sym[i] ** exp for i, exp in enumerate(e)]))
            poly += parameters[-1] * terms[-1]

        self.para_index = para_index
        return poly, parameters, terms

    def RealExpression(self, param_value):

        subs_dict = dict(zip(self.parameters, param_value))
        expr_sub = self.poly.subs(subs_dict)
        self.RealExpr = sp.expand(expr_sub)

    def compute_lip_weights_method1(self, area):

        x_symbols = self.var_sym

        # area = np.array(area)
        low = area.low  # [low1, low2]
        high = area.up  # [upper1, upper2]
        terms = self.terms

        #low = np.asarray(X_low, dtype=float)
        #high = np.asarray(X_high, dtype=float)

        if len(low) != len(x_symbols) or len(high) != len(x_symbols):
            raise ValueError("wrong")

        if np.any(low >= high):
            raise ValueError("wrong")

        R = np.maximum(np.abs(low), np.abs(high))
        weights = []

        for term in terms:

            poly = sp.Poly(term, *x_symbols)

            if len(poly.monoms()) != 1 or poly.coeffs()[0] != 1:
                raise ValueError("wrong")

            alpha = poly.monoms()[0]

            grad_bounds = []

            for i, exponent in enumerate(alpha):
                if exponent == 0:
                    grad_bounds.append(0.0)
                    continue

                bound = float(exponent)

                for j, power in enumerate(alpha):
                    if i == j:
                        bound *= R[j] ** (power - 1)
                    else:
                        bound *= R[j] ** power

                grad_bounds.append(bound)

            weights.append(np.linalg.norm(grad_bounds, ord=2))

        return np.asarray(weights)

    def compute_lip_weights_method2(self, area):

        x_symbols = self.var_sym
        #area = np.array(area)
        low = area.low  # [low1, low2]
        high = area.up  # [upper1, upper2]
        terms = self.terms

        n = len(x_symbols)
        Q = len(terms)

        if len(low) != len(x_symbols) or len(high) != len(x_symbols):
            raise ValueError("wrong")


        if np.any(low >= high):
            raise ValueError("wrong")


        R = np.maximum(np.abs(low), np.abs(high))


        powers = np.array([
            sp.Poly(term, *x_symbols).monoms()[0]
            for term in terms
        ], dtype=int)


        D = np.zeros((n, Q), dtype=float)

        for i in range(n):
            for j in range(Q):
                alpha = powers[j]

                if alpha[i] == 0:
                    continue

                value = float(alpha[i])

                for k in range(n):
                    exponent = alpha[k] - (1 if k == i else 0)
                    value *= R[k] ** exponent

                D[i, j] = value

        return D

    def compute_gradient_matrix(self, points):

        M, n = points.shape
        x_symbols = self.var_sym
        terms = self.terms

        Q = len(terms)

        if n != len(x_symbols):
            raise ValueError("wrong")


        grad_expr = sp.Matrix([
            [sp.diff(term, x) for term in terms]
            for x in x_symbols
        ])


        grad_func = sp.lambdify(
            x_symbols,
            grad_expr,
            modules="numpy",
            cse=True
        )


        A = np.empty((M, n, Q))

        for i in range(M):
            A[i] = np.asarray(
                grad_func(*points[i])
            ).reshape(n, Q)

        return A

if __name__ == '__main__':
    '''
    deg = 2
    x_dim = 2
    pol = Polynomial(2, 2, 0)
    pol.show()

    A, par = pol.substitute(np.array([[1, 2], [0.1, 4]]))
    print(f"matrix = {A}")
    print(f"par = {par}")
    '''


    pol = Polynomial(2, 2, 0)
    print(pol.terms)
    weights = pol.compute_lip_weights_method2([[-2, 2], [-2, 2]])
    print(weights)
