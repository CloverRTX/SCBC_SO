import numpy as np
import sympy
from scipy.stats import norm
from scipy.integrate import quad
import math
from torch.distributions import Normal

"""
默认采用 高斯分布
各个随机变量之间独立同分布
v : 随机变量
miu : 数学期望
sigma : 标准方差

v ~ N(miu, sigma ** 2)

随机变量 概率密度求解
"""

class GaussianDistribution:
    # 默认是标准正态 (1-dim)
    def __init__(self, loc = 0., scale = 1.):
        # 数学期望
        self.loc = loc

        # 标准差
        self.scale = scale

        self.x = sympy.Symbol('x')

    @property
    def mean(self):
        return self.loc

    @property
    def std(self):
        return self.scale

    @property
    def var(self):
        return self.scale ** 2

    # 符号计算较慢
    # 不可广播
    def cdf(self, a, b):
        # 位于区间[a, b]的概率
        low = min(a, b)
        upper = max(a, b)

        expr = sympy.integrate(self.pdf, (self.x, low, upper))
        return expr.evalf()

    @property
    def pdf(self):
        miu = self.loc
        sigma = self.scale
        p = (self.x - miu)**2 / (2 * sigma ** 2)
        return 1.0 / (sympy.sqrt(2 * np.pi) * sigma) * sympy.exp(-p)

    # 其余相关积分计算
    # ∫vp(v)dv 与 cdf 数学原理 相同 ([a,b]区间)
    # 可广播
    def integrate_vMulpdf(self, a, b):
        expr = sympy.integrate(self.x * self.pdf, self.x)
        F = sympy.lambdify(self.x, expr, "numpy")
        return F(b) - F(a)

    # 可广播
    def integrate_pdf(self, a, b):
        rv = norm(loc=self.loc, scale=self.scale)
        return rv.cdf(b) - rv.cdf(a)

    def RV_sample(self, num):
        dist = Normal(0.0, 0.1)
        z = dist.sample((num,))
        return z


def px(x):
    miu = 0.
    sigma = 0.1
    p = (x - miu) ** 2 / (2 * sigma ** 2)
    return x / (math.sqrt(2 * math.pi) * sigma) * math.exp(-p)



if __name__ == "__main__":
    rv = GaussianDistribution(loc=0.0, scale=0.1)
    a = np.linspace(-0.3, 0, 3)
    a = a.reshape((len(a), 1))
    b = np.linspace(0, 0.3, 3)
    b = b.reshape((len(b), 1))

    print(rv.integrate_pdf(-1, 1))
    print(rv.integrate_vMulpdf(-0.5, 0.2))
    result, error = quad(func=px, a=-0.5, b=0.2)
    print(result, error)
