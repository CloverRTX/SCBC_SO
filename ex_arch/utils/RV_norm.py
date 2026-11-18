import numpy as np
import sympy
from scipy.stats import norm
from scipy.integrate import quad
import math
from torch.distributions import Normal


class GaussianDistribution:

    def __init__(self, loc = 0., scale = 1.):

        self.loc = loc


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

    def cdf(self, a, b):

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


    def integrate_vMulpdf(self, a, b):
        expr = sympy.integrate(self.x * self.pdf, self.x)
        F = sympy.lambdify(self.x, expr, "numpy")
        return F(b) - F(a)


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
