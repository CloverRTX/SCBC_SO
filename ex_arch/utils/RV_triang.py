import torch
from scipy.stats import triang
from scipy.integrate import quad
import numpy as np

class TriangularDistribution:
    '''
             .
            /|\
           / | \
          /  |  \
         /   |   \
        a----c----b

    '''
    # 三角分布
    def __init__(self, a = -0.5, b = 0.5, c = 0.):
        # 默认的三角分布
        # a 最小值
        # b 最大值
        # c 众数
        self.a = a
        self.b = b
        self.c = c
        #self.triangular_dist = triang(c=0,
        #                              loc=self.a, scale=self.b - self.a)
        self.triangular_dist = triang(c=(self.c - self.a) / (self.b - self.a),
                                      loc = self.a, scale=self.b - self.a)
    # ∫p(v)dv
    # 可以广播
    def cdf(self, a, b):
        # 位于区间[a, b]的概率
        lower = a
        upper = b
        return self.triangular_dist.cdf(upper) - self.triangular_dist.cdf(lower)

    # ∫vp(v)dv 与 cdf 数学原理 相同 ([a,b]区间)
    # 无法广播 一循环方式进行
    # 运算结果太慢
    def v_mul_px(self, a, b):
        def integrand(x):
            return x * self.triangular_dist.pdf(x)
        expectations = []
        for i in range(a.shape[0]):
            lower = a[i]
            upper = b[i]
        # 位于区间[a, b]的数学期望
            exp, _ = quad(integrand, lower, upper)
            expectations.append(exp)
        return torch.tensor(expectations)

    # 根据具体分布而来，不可泛用
    # 优点是可以广播 且 计算速度很快
    def v_mul_px_plus(self, a, b):
        con1 = (a <= 0) & (b <= 0)
        con2 = (a < 0) & (b > 0)
        con3 = (a >= 0) & (b >= 0)

        result = np.zeros((a.shape[0], ))
        result[con1] = (b[con1] ** 3) / 3 + (b[con1] ** 2) / 2 - (a[con1] ** 3) / 3 - (a[con1] ** 2) / 2
        result[con2] = -(b[con2] ** 3) / 3 + (b[con2] ** 2) / 2 - (a[con2] ** 3) / 3 - (a[con2] ** 2) / 2
        result[con3] = -(b[con3] ** 3) / 3 + (b[con3] ** 2) / 2 + (a[con3] ** 3) / 3 - (a[con3] ** 2) / 2

        return torch.tensor(result)

    def RV_sample(self, num):
        return torch.tensor(self.triangular_dist.rvs(size=num))

if __name__ == "__main__":
    rv = TriangularDistribution(-1, 1, 0)
    #print(rv.RV_sample(100))
    a = np.random.uniform(-1,0, size=(100000,))
    print(a + 0.5)
    b = np.ones((a.shape[0], 2))
    b[:, 0] = a
    b[:, 1] = a + 0.5
    print(b)
    print("======================")
    r1 = rv.v_mul_px_plus(b[:,0], b[:,1])
    r2 = rv.cdf(b[:,0], b[:,1])
    print(r2)

