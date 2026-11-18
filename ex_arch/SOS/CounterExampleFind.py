import sympy as sp
import numpy as np
from scipy.optimize import minimize
import csv

class CounterExampleFind():

    def __init__(self, poly, spec_expr, config):
        self.poly = poly
        self.x_var_num = poly.var_dim
        self.x_var_sym = poly.var_sym

        self.X0 = config["verify"]["X0"]
        self.Xu = config["verify"]["Xu"]
        self.X = config["verify"]["X"]
        self.epsilon = config["verify"]["epsilon"]

        self.eps = 0.05
        self.cex_num = config["verify"]["cex_num"]

        self.E_B_expr = spec_expr

        self.path_X0 = config["sample_config"]["path_X0"]
        self.path_Xs = config["sample_config"]["path_Xs"]
        self.path_Xu = config["sample_config"]["path_Xu"]
        self.path_X = config["sample_config"]["path_X"]



    def find_CEX(self):
        CEX_nonneg, eye1 = self.find_nonneg()
        CEX_init, eye2 = self.find_init()
        CEX_unsafe, eye3 = self.find_unsafe()
        CEX_MECon, eye4 = self.find_MECon()

        if not eye1:
            print(f"nonneg 反例个数{CEX_nonneg.shape[0]}, 写入...")
            with open(self.path_X, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(CEX_nonneg)
                print(f"nonneg 反例个数{CEX_nonneg.shape[0]}, 写入完成")
        else:
            print(f"nonneg 无反例")

        if not eye2:
            print(f"init 反例个数{CEX_init.shape[0]}, 写入...")
            with open(self.path_X0, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(CEX_init)
                print(f"init 反例个数{CEX_init.shape[0]}, 写入完成")
        else:
            print(f"init 无反例")

        if not eye3:
            print(f"unsafe 反例个数{CEX_unsafe.shape[0]}, 写入...")
            with open(self.path_Xu, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(CEX_unsafe)
                print(f"unsafe 反例个数{CEX_unsafe.shape[0]}, 写入完成")
        else:
            print(f"unsafe 无反例")


        if not eye4:
            print(f"MECon 反例个数{CEX_MECon.shape[0]}, 写入...")
            with open(self.path_Xs, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(CEX_MECon)
                print(f"MECon 反例个数{CEX_MECon.shape[0]}, 写入完成")
        else:
            print(f"MECon 无反例")


        eye = eye1 & eye2 & eye3 & eye4
        return eye


    def find_nonneg(self):
        samples = []
        satisfy = True
        # 使用 minimize function
        opt = sp.lambdify(self.x_var_sym, self.poly.RealExpr)
        # Set an initial guess.
        x0 = self.startPoint(self.X)
        res = minimize(fun=lambda x: opt(*x), x0=x0, bounds=self.X)
        if res.fun >= 0:
            pass
        else:
            print('Counterexamples found in the nonneg con:', res.x, 'B:', self.poly.RealExpr)

            # 生成反例
            samples.append(res.x)

            samples.extend(self.generate_sample(res.x, self.cex_num - 1))
            satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(x, 'nonneg')])
        return samples, satisfy
    def find_init(self):
        samples = []
        satisfy = True
        # 使用 minimize function
        opt = sp.lambdify(self.x_var_sym, self.poly.RealExpr)
        # Set an initial guess.
        x0 = self.startPoint(self.X0)
        res = minimize(fun=lambda x: self.epsilon-opt(*x), x0=x0, bounds=self.X0)
        if res.fun >= 0:
            pass
        else:
            print('Counterexamples found in the initial set:', res.x, 'B:', self.poly.RealExpr)

            # 生成反例
            samples.append(res.x)

            samples.extend(self.generate_sample(res.x, self.cex_num - 1))
            satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(x, 'init')])
        return samples, satisfy

    def find_unsafe(self):
        samples = []
        satisfy = True
        # 使用 minimize function
        opt = sp.lambdify(self.x_var_sym, self.poly.RealExpr)
        # Set an initial guess.
        x0 = self.startPoint(self.Xu)
        res = minimize(fun=lambda x: opt(*x) - 1., x0=x0, bounds=self.Xu)
        if res.fun >= 0:
            pass
        else:
            print('Counterexamples found in the unsafe set:', res.x, 'B:', self.poly.RealExpr)

            # 生成反例
            samples.append(res.x)

            samples.extend(self.generate_sample(res.x, self.cex_num - 1))
            satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(x, 'unsafe')])
        return samples, satisfy

    def find_MECon(self):
        samples = []
        satisfy = True
        # 使用 minimize function
        opt = sp.lambdify(self.x_var_sym, self.E_B_expr)
        # Set an initial guess.
        x0 = self.startPoint(self.X)
        res = minimize(fun=lambda x: -opt(*x), x0=x0, bounds=self.X)
        if res.fun >= 0:
            pass
        else:
            print('Counterexamples found in the MECon:', res.x, 'B:', self.poly.RealExpr)

            # 生成反例
            samples.append(res.x)

            samples.extend(self.generate_sample(res.x, self.cex_num - 1))
            satisfy = False

        samples = np.array([x for x in samples if self.is_counter_example(x, 'MECon')])
        return samples, satisfy


    def generate_sample(self, center, sample_nums=10):
        result = []
        for i in range(sample_nums):
            rd = (np.random.random(self.x_var_num) - 0.5) * self.eps
            rd = rd + center
            result.append(rd)
        return result

    def is_counter_example(self, x, condition: str) -> bool:
        if condition not in ['nonneg', 'init', 'unsafe', 'MECon']:
            raise ValueError(f'{condition} is not in validation condition!')
        d = {self.x_var_sym[i]: x[i] for i in range(self.x_var_num)}

        ME_numerical = 0.
        b_numerical = 0.
        if condition == 'MECon':
            ME_numerical = self.E_B_expr.subs(d)
        else:
            b_numerical = self.poly.RealExpr.subs(d)

        if condition == 'nonneg' and b_numerical < 0 and self.inX(x):
            return True
        if condition == 'init' and b_numerical - self.epsilon > 0 and self.inX0(x):
            return True
        if condition == 'unsafe' and b_numerical - 1. < 0 and self.inXu(x):
            return True
        if condition == 'MECon' and ME_numerical > 0 and self.inX(x):
            return True
        return False

    def inX(self, x):
        for i in range(self.x_var_num):
            if x[i] < self.X[i][0] or x[i] > self.X[i][1]:
                return False
        return True

    def inX0(self, x):
        for i in range(self.x_var_num):
            if x[i] < self.X0[i][0] or x[i] > self.X0[i][1]:
                return False
        return True

    def inXu(self, x):
        for i in range(self.x_var_num):
            if x[i] < self.Xu[i][0] or x[i] > self.Xu[i][1]:
                return False
        return True

    def startPoint(self, area):
        area = np.array(area)
        low = area[:, 0]  # 每维下界
        high = area[:, 1]
        point = np.random.uniform(low, high)
        return point
