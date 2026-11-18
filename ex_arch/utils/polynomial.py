import sympy as sp
from itertools import product
from functools import reduce
import numpy as np

# self.n  x的维度
# self.var_count 变量计数

class Polynomial():
    '''
    x: 多项式的变量; x1, x2, x3,...
    parameter: 多项式的系数;
    deg: 多项式的次数

    对于x1,x2, deg = 2，多项式的一般形式:
        parameter0 + parameter1*x1 + parameter2*x2 + parameter3*x1^{2}
        + parameter4*x2^{2} + parameter5*x1*x2
    '''



    def __init__(self, var_dim, deg, para_index):

        # x的维度
        self.var_dim = var_dim

        # x的符号表示形式
        self.var_sym = sp.symbols(['x{}'.format(i + 1) for i in range(var_dim)])

        # 次数
        self.deg = deg

        # 系数下标
        self.para_index = para_index

        # 多项式表达式, 所有的待定系数, 所有的不带系数的项
        self.poly, self.parameters, self.terms \
            = self.polynomial(self.para_index, self.deg)

        self.RealExpr = None

    # 多项式代入
    def substitute(self, points):
        '''
        parameter0 + parameter1*x1 + parameter2*x2 + parameter3*x1^{2}
        + parameter4*x2^{2} + parameter5*x1*x2

        将 x1 = ? 和 x2 = ?代入后，变成一个线性形式，
        从而可以转换成矩阵形式

        :return: A * param 的形式
        '''
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
        '''
        生成多项式

        对于x1,x2, deg = 2:
        a0 + a1*x1 + a2*x2 + a3*x1^{2} + a4*x2^{2} + a5*x1*x2

        :param para_index: 多项式系数的下标
        :param deg: 多项式次数

        :return:
            poly: 完整的多项式表达式
            parameters: 所有的待定参数
            terms: 所有的不带系数的项
        '''

        '''
        if deg == 2 and x_dim > 8:
            parameters = []
            terms = []
            poly = 0
            parameters.append(sp.symbols('parameter' + str(self.var_count)))
            self.var_count += 1
            poly += parameters[-1]
            terms.append(1)
            for i in range(self.n):
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i])
                poly += parameters[-1] * terms[-1]
                parameters.append(sp.symbols('parameter' + str(self.var_count)))
                self.var_count += 1
                terms.append(self.x[i] ** 2)
                poly += parameters[-1] * terms[-1]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    parameters.append(sp.symbols('parameter' + str(self.var_count)))
                    self.var_count += 1
                    terms.append(self.x[i] * self.x[j])
                    poly += parameters[-1] * terms[-1]
            return poly, parameters, terms
        else:
        '''
        # 存储所有的多项式待定系数
        parameters = []
        # 存储所有不考虑系数的项
        terms = []

        # x的符号表示形式
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
        # 构造替换字典
        subs_dict = dict(zip(self.parameters, param_value))
        expr_sub = self.poly.subs(subs_dict)
        self.RealExpr = sp.expand(expr_sub)

    def compute_lip_weights_method1(self, area):
        """
        方法一添加Lip常数的约束

        计算每个单项式的 Lipschitz 权重 c_j。

        area: X定义域

        返回:
            numpy.ndarray, shape=(Q,)
        """
        x_symbols = self.var_sym

        # area = np.array(area)
        low = area.low  # [low1, low2]
        high = area.up  # [upper1, upper2]
        terms = self.terms

        #low = np.asarray(X_low, dtype=float)
        #high = np.asarray(X_high, dtype=float)

        if len(low) != len(x_symbols) or len(high) != len(x_symbols):
            raise ValueError("状态空间维度不匹配")

        if np.any(low >= high):
            raise ValueError("状态空间上下界不合法")

        R = np.maximum(np.abs(low), np.abs(high))
        weights = []

        for term in terms:
            # 提取单项式指数
            poly = sp.Poly(term, *x_symbols)

            if len(poly.monoms()) != 1 or poly.coeffs()[0] != 1:
                raise ValueError(f"{term} 不是系数为 1 的单项式")

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
        """
        方法二添加Lip常数的约束

        计算梯度分量上界矩阵 D

        D[i, j] = sup_{x in X} |d phi_j(x) / d x_i|

        area: X定义域

        返回:
            numpy.ndarray, shape=(n,Q)
        """
        x_symbols = self.var_sym
        #area = np.array(area)
        low = area.low  # [low1, low2]
        high = area.up  # [upper1, upper2]
        terms = self.terms

        n = len(x_symbols)
        Q = len(terms)

        if len(low) != len(x_symbols) or len(high) != len(x_symbols):
            raise ValueError("状态空间维度不匹配")


        if np.any(low >= high):
            raise ValueError("状态空间上下界不合法")

        # 每个状态变量绝对值的最大值
        R = np.maximum(np.abs(low), np.abs(high))

        # 构造指数矩阵，shape=(Q, n)
        powers = np.array([
            sp.Poly(term, *x_symbols).monoms()[0]
            for term in terms
        ], dtype=int)

        # 构造 D 矩阵，shape=(n, Q)
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
        """
        计算所有采样点上的梯度基矩阵。

        points: shape=(M, n)
           M个状态采样点

        返回:
            A: shape=(M, n, Q)
        """
        M, n = points.shape
        x_symbols = self.var_sym
        terms = self.terms

        Q = len(terms)

        if n != len(x_symbols):
            raise ValueError("状态空间维度不匹配")

        # 1. 对所有单项式求梯度
        grad_expr = sp.Matrix([
            [sp.diff(term, x) for term in terms]
            for x in x_symbols
        ])

        # 2. 转换成 NumPy 函数
        grad_func = sp.lambdify(
            x_symbols,
            grad_expr,
            modules="numpy",
            cse=True
        )

        # 3. 批量计算
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

    # 示例
    pol = Polynomial(2, 2, 0)
    print(pol.terms)
    weights = pol.compute_lip_weights_method2([[-2, 2], [-2, 2]])
    print(weights)
