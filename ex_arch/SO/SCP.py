import time

from RL.Env import Zones
from utils.RV_triang import TriangularDistribution

import numpy as np
import sympy as sp
import torch
import matplotlib.pyplot as plt
from scipy.special import gamma
from datetime import datetime
import cvxpy as cp
import math


from scipy.special import betainc
from scipy.optimize import brentq

class SCP_solver():

    def __init__(self, SBC_model, Col_model, config, dynamic):
        self.device = config['device']
        self.SBC_model = SBC_model
        self.Col_model = Col_model
        self.config = config
        self.dynamic = dynamic
        self.x_var_num = config["dynamics"]["x_var_num"]

        self.deg = SBC_model.deg
        self.distribution = TriangularDistribution()
        self.random_var_num = config["dynamics"]["random_var_num"]
        self.mathcal_H = config["Bx_model"]["mathcal_H"]
        self.rho = config["Bx_model"]["rho"]

        self.d = len(self.SBC_model.parameters) + 3


        # ===========SCP_N_hat_N===========
        self.delta = config["SCP_N_hat_N"]["delta"]
        self.hat_M = None
        self.hat_N = config["SCP_N_hat_N"]["hat_N"]
        self.beta_s = None

        self.X = dynamic.D_zones
        self.X0 = dynamic.I_zones
        self.Xu = dynamic.U_zones

        self.Big_D_zones = dynamic.Big_D_zones

        # ===========SCP_N===========
        self.epsilon = config["SCP_N"]["epsilon"]
        self.beta = config["SCP_N"]["beta"]
        self.N = self.calcN(self.epsilon, self.beta, self.d)

        # self.miu = self.func_G_inverse_epsilon(self.epsilon, self.X)
        self.miu = config["SCP_N_hat_N"]["miu"]




    def solve(self, plot = False):

        print("============Before SO solving============")
        print(f"(Scenario Sample Number) N = {self.N}")
        print(f"(Perturbation Sample Number) hat_N = {self.hat_N}")
        print(f"(Max Safety Step Number) mathcal_H = {self.mathcal_H}")
        print(f"(Diff between E and hat_E ) delta = {self.delta}")
        print(f"(Scenario Optimization) epsilon = {self.epsilon}")
        print(f"(Scenario Optimization) beta = {self.beta}")
        print(f"(Safety Prob Preset) 1 - rho = {1 - self.rho}")
        print(f"(Regularization Coefficient) miu = {self.miu}")



        scenario = self.ScenarioSampling(self.N, self.Big_D_zones)

        if plot:
            self.dynamic.sample_view(scenario)


        # For g1
        #B(param, x) = A * param
        scenario_g1 = self.filter_g1(scenario, self.X)
        if plot:
            self.dynamic.sample_view(scenario_g1)
        matrix_g1, _ = self.SBC_model.substitute(scenario_g1)

        # For g2
        scenario_g2 = self.filter_g2(scenario, self.X0)
        if plot:
            self.dynamic.sample_view(scenario_g2)
        matrix_g2, _ = self.SBC_model.substitute(scenario_g2)


        # For g3
        scenario_g3 = self.filter_g3(scenario, self.Xu)
        if plot:
            self.dynamic.sample_view(scenario_g3)
        matrix_g3, _ = self.SBC_model.substitute(scenario_g3)


        # For g5
        #start = time.time()
        #matrix_g5 = self.g_5Matrix_opt(scenario)
        #end = time.time()


        start = time.time()

        scenario_g5 = self.filter_g1(scenario, self.X)
        matrix_g5 = self.g_5Matrix_opt_batch(scenario_g5)
        end = time.time()


        start = time.time()
        Kappa1 = self.PAC_cvx_addLipCon_based_method1(matrix_g1, matrix_g2, matrix_g3, matrix_g5)
        end = time.time()


        start = time.time()
        Kappa2 = self.PAC_cvx_addLipCon_based_method2(matrix_g1, matrix_g2, matrix_g3, matrix_g5)
        end = time.time()


        start = time.time()
        Kappa3 = self.PAC_cvx_addLipCon_based_method3(matrix_g1, matrix_g2, matrix_g3, matrix_g5)
        end = time.time()



        #================================================
        print("============After SO============")


        start = time.time()
        hat_M = self.calc_hat_M(scenario_g5)
        end = time.time()


        beta_s = hat_M / (self.delta * self.delta * self.hat_N)

        print(f"hat_M = {hat_M}")
        print(f"beta_s = {beta_s}")




        #Lx1 = self.calcLip(scenario)
        Lx2 = self.calcLip2(self.Big_D_zones)
        Lx3 = self.calcLip3(self.Big_D_zones)
        G_value = self.func_G_inverse_epsilon(self.epsilon, self.Big_D_zones)

        #print(f"(method1) Lip_x = {Lx1}")
        #print(f"Kappa  = {Kappa}")
        print(f"(method2) Lip_x_g1 = {Lx2}")
        print(f"(method2) Lip_x_g5 = {Lx3}")
        Lx = max(Lx2, Lx3)
        print(f"(method2) Lip_x = {Lx}")
        print(f"(G_ni_epsilon) G_value = {G_value}")
        print(f"Lip_x * G_ni_epsilon = {Lx * G_value}")
        print(f"Kappa1 + Lx2 * G_value = {Kappa1 + Lx * G_value}")
        print(f"Kappa2 + Lx2 * G_value = {Kappa2 + Lx * G_value}")
        print(f"Kappa3 + Lx2 * G_value = {Kappa3 + Lx * G_value}")
        print(f"eye = {min(Kappa1, Kappa2, Kappa3) + Lx * 0.0138}")
        #print(Kappa + Lx1 * G_value)
        #print(Kappa + Lx2 * G_value)
        return None


    def func_G_inverse_epsilon(self, epsilon, area):

        # area = np.array(area)

        lows = area.low  # [low1, low2]
        uppers = area.up  # [upper1, upper2]

        eta = uppers - lows

        n = len(eta)
        a = np.pi ** (n / 2) / (2 ** n * gamma(n / 2 + 1))

        epsilon_max = a * np.min(eta) ** n / np.prod(eta)

        if not (0 <= epsilon <= epsilon_max):

            raise ValueError("epsilon error")

        return (epsilon * np.prod(eta) / a) ** (1 / n)

    def ScenarioSampling(self, sample_num, area):

        # area = np.array(area)

        lows = area.low  # [low1, low2]
        uppers = area.up  # [upper1, upper2]


        scenario = lows + (uppers - lows) * np.random.rand(sample_num, self.x_var_num)

        return scenario


    def filter_g1(self, points, area):
        r = area.r
        center = area.center
        mask = np.sum((points - center) ** 2, axis=1) <= r ** 2
        return points[mask]

    def filter_g2(self, points, area):
        r = area.r
        center = area.center
        mask = np.sum((points - center) ** 2, axis=1) <= r ** 2
        return points[mask]

    def filter_g3(self, points, area1):
        r = area1.r
        center = area1.center
        mask2 = np.sum((points - center) ** 2, axis=1) <= r ** 2

        return points[mask2]



    def g_5Matrix(self, points):
        hat_N = self.hat_N
        part1 = None
        for i in range(points.shape[0]):
            data = np.zeros((hat_N, self.x_var_num + self.random_var_num))
            perturbation = self.distribution.RV_sample((hat_N, self.random_var_num))
            #perturbation = np.ones((hat_N, self.random_var_num))
            data[:, :self.x_var_num] = points[i, :]
            data[:, self.x_var_num:] = perturbation
            x_next = self.dynamic.forward(torch.from_numpy(data[:, :self.x_var_num]),
                                 torch.from_numpy(data[:, self.x_var_num:]))

            matrix, _ = self.SBC_model.substitute(x_next.numpy())
            temp = np.mean(matrix, axis=0).reshape((1, -1))
            if part1 is None:
                part1 = temp
            else:
                part1 = np.vstack((part1, temp))

        part2, _ = self.SBC_model.substitute(points)
        return part1 - part2


    def g_5Matrix_opt(self, points):
        N, n = points.shape
        M = self.hat_N


        x = np.repeat(points, M, axis=0)

        perturbation = self.distribution.RV_sample((N * M, self.random_var_num))
        #perturbation = np.ones((N * M, self.random_var_num))
        x_next = self.dynamic.forward(
            torch.from_numpy(x),
            perturbation
        )
        # 4.
        matrix, _ = self.SBC_model.substitute(x_next.numpy())
        # 5.
        part1 = matrix.reshape(N, M, -1).mean(axis=1)

        # 6.
        part2, _ = self.SBC_model.substitute(points)

        return part1 - part2


    def g_5Matrix_opt_batch(self, points):
        N, n = points.shape
        M = self.hat_N

        #
        part1 = np.empty((N, len(self.SBC_model.parameters)))
        batch_size = 500
        if batch_size > N:
            batch_size = N

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            #
            batch = points[start:end]
            K = batch.shape[0]

            # 1.
            x = np.repeat(batch, M, axis=0)


            # 1.5.
            u = self.Col_model(
                torch.as_tensor(x, dtype=torch.float32)
            ).detach()

            # 2.
            perturbation = self.distribution.RV_sample(
                (K * M, self.random_var_num)
            )

            #perturbation = np.ones(
            #    (K * M, self.random_var_num)
            #)


            # 3.
            x_next = self.dynamic.forward(
                torch.from_numpy(x),
                u,
                perturbation
            )

            # 4.
            matrix, _ = self.SBC_model.substitute(
                x_next.numpy()
            )

            # 5.
            temp = matrix.reshape(K, M, -1).mean(axis=1)

            # 6.
            part1[start:end] = temp

        # 7.
        part2, _ = self.SBC_model.substitute(points)

        return part1 - part2

    def PAC_cvx_addLipCon_based_method1(self, matrix_g1, matrix_g2, matrix_g3, matrix_g5):
        '''

        :param matrix_g1:
        :param matrix_g2:
        :param matrix_g3:
        :param matrix_g5:
        :return:
        '''

        small_number = 1e-5
        mathcal_H = self.mathcal_H
        rho = self.rho
        delta = self.delta

        # ————————————————————————————————————————————————————————
        Kappa = cp.Variable()

        var_lambda = cp.Variable()

        var_c = cp.Variable(nonneg=True)

        #
        poly_param = self.SBC_model.parameters
        #
        param = cp.Variable(len(poly_param))
        t_param = cp.Variable(len(poly_param), nonneg=True)

        I = np.eye(len(poly_param))


        # ————————————————————————————————————————————————————————
        constraints = [-matrix_g1 @ param <= Kappa,     # g1
                       matrix_g2 @ param <= 1 + Kappa,  # g2
                       -matrix_g3 @ param <= Kappa - var_lambda,   # g3
                       (1 + mathcal_H * var_c) / rho - var_lambda - Kappa <= 0,  #g4
                       matrix_g5 @ param <= var_c + Kappa - delta, #g5
                       var_lambda >= 1 + small_number,
                       I @ param <= t_param, I @ param >= -t_param]

        # Lip
        weights = self.SBC_model.compute_lip_weights_method1(self.Big_D_zones)

        #
        miu = self.miu
        L = miu * weights @ t_param
        obj = cp.Minimize(Kappa + L)

        prob = cp.Problem(obj, constraints)

        # ————————————————————————————————————————————————————————
        # print(prob.constraints)

        print("============SO solving============")

        print("DCP: ", prob.is_dcp())
        prob.solve(solver=cp.MOSEK)
        print("status:", prob.status)
        print(f"SO opt value (Kappa value + L) = {prob.value}")
        print(f"c = {var_c.value}")
        print(f"lambda = {var_lambda.value}")
        print(f"Kappa = {Kappa.value}")
        print(f"Lip Con = {prob.value - Kappa.value}")
        print(f"Psafe >= {1 - (1 + var_c.value * self.mathcal_H) / var_lambda.value}")
        print("============SO end============")

        param_value = [param[i].value for i in range(len(poly_param))]
        #
        self.SBC_model.RealExpression(param_value)

        return Kappa.value

    def PAC_cvx_addLipCon_based_method2(self, matrix_g1, matrix_g2, matrix_g3, matrix_g5):
        '''

        :param matrix_g1:
        :param matrix_g2:
        :param matrix_g3:
        :param matrix_g5:
        :return:
        '''

        small_number = 1e-5
        mathcal_H = self.mathcal_H
        rho = self.rho
        delta = self.delta

        # ————————————————————————————————————————————————————————
        Kappa = cp.Variable()

        var_lambda = cp.Variable()

        var_c = cp.Variable(nonneg=True)

        #
        poly_param = self.SBC_model.parameters
        #
        param = cp.Variable(len(poly_param))
        t_param = cp.Variable(len(poly_param), nonneg=True)

        I = np.eye(len(poly_param))


        # ————————————————————————————————————————————————————————
        constraints = [-matrix_g1 @ param <= Kappa,     # g1
                       matrix_g2 @ param <= 1 + Kappa,  # g2
                       -matrix_g3 @ param <= Kappa - var_lambda,   # g3
                       (1 + mathcal_H * var_c) / rho - var_lambda - Kappa <= 0,  #g4
                       matrix_g5 @ param <= var_c + Kappa - delta, #g5
                       var_lambda >= 1 + small_number,
                       I @ param <= t_param, I @ param >= -t_param]

        # Lip
        weights = self.SBC_model.compute_lip_weights_method2(self.Big_D_zones)

        #
        miu = self.miu

        L = miu * cp.norm(weights @ t_param, 2)

        obj = cp.Minimize(Kappa + L)

        prob = cp.Problem(obj, constraints)

        # ————————————————————————————————————————————————————————
        # print(prob.constraints)

        print("============SO solving============")
        #
        print("DCP: ", prob.is_dcp())
        prob.solve(solver=cp.MOSEK)
        print("status:", prob.status)
        print(f"SO opt value (Kappa value + L) = {prob.value}")
        print(f"c = {var_c.value}")
        print(f"lambda = {var_lambda.value}")
        print(f"Kappa = {Kappa.value}")
        print(f"Lip Con = {prob.value - Kappa.value}")
        print(f"Psafe >= {1 - (1 + var_c.value * self.mathcal_H) / var_lambda.value}")
        print("============SO end============")

        param_value = [param[i].value for i in range(len(poly_param))]
        #
        self.SBC_model.RealExpression(param_value)

        return Kappa.value


    def PAC_cvx_addLipCon_based_method3(self, matrix_g1, matrix_g2, matrix_g3, matrix_g5):
        '''

        :param matrix_g1:
        :param matrix_g2:
        :param matrix_g3:
        :param matrix_g5:
        :return:
        '''

        small_number = 1e-5
        mathcal_H = self.mathcal_H
        rho = self.rho
        delta = self.delta

        # ————————————————————————————————————————————————————————
        Kappa = cp.Variable()

        var_lambda = cp.Variable()

        var_c = cp.Variable(nonneg=True)


        poly_param = self.SBC_model.parameters

        param = cp.Variable(len(poly_param))

        L_max = cp.Variable(nonneg=True)


        # ————————————————————————————————————————————————————————
        constraints = [-matrix_g1 @ param <= Kappa,     # g1
                       matrix_g2 @ param <= 1 + Kappa,  # g2
                       -matrix_g3 @ param <= Kappa - var_lambda,   # g3
                       (1 + mathcal_H * var_c) / rho - var_lambda - Kappa <= 0,  #g4
                       matrix_g5 @ param <= var_c + Kappa - delta, #g5
                       var_lambda >= 1 + small_number]

        len_points = 1000
        points = self.ScenarioSampling(len_points, self.Big_D_zones)
        points = self.filter_g1(points, self.X)
        len_points = points.shape[0]
        # Lip
        weights = self.SBC_model.compute_gradient_matrix(points)

        #
        miu = self.miu

        for i in range(len_points):
            constraints.append(
                cp.norm(weights[i] @ param, 2) <= L_max
            )

        L = miu * L_max

        obj = cp.Minimize(Kappa + L)

        prob = cp.Problem(obj, constraints)

        # ————————————————————————————————————————————————————————
        # print(prob.constraints)

        print("============SO solving============")
        #
        print("DCP: ", prob.is_dcp())
        prob.solve(solver=cp.MOSEK)
        print("status:", prob.status)
        print(f"SO opt value (Kappa value + L) = {prob.value}")
        print(f"c = {var_c.value}")
        print(f"lambda = {var_lambda.value}")
        print(f"Kappa = {Kappa.value}")
        print(f"Lip Con = {prob.value - Kappa.value}")
        print(f"Psafe >= {1 - (1 + var_c.value * self.mathcal_H) / var_lambda.value}")
        print("============SO end============")

        param_value = [param[i].value for i in range(len(poly_param))]
        #
        self.SBC_model.RealExpression(param_value)

        return Kappa.value

    def calcN(self, epsilon, beta, d):
        N = 2 * (np.log(1 / beta) + d) / epsilon

        return math.ceil(N)


    def calc_hat_M(self, points):
        expr = self.SBC_model.RealExpr
        x_var = self.SBC_model.var_sym
        func_Bx = sp.lambdify(x_var, expr, modules='numpy')

        N = points.shape[0]
        M = self.hat_N
        m = self.random_var_num

        if M < 2:
            raise ValueError("hat_N must be at least 2")

        max_var = 0.05
        #
        batch_size = 500

        if N <= batch_size:
            batch_size = N

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = points[start:end]
            K = batch.shape[0]

            # 1.
            x = np.repeat(batch, M, axis=0)

            # 1.5.
            u = self.Col_model(
                torch.as_tensor(x, dtype=torch.float32)
            ).detach()


            # 2.
            perturbation = self.distribution.RV_sample(
                (K * M, m)
            )

            # 3.
            x_next = self.dynamic.forward(
                torch.from_numpy(x),
                u,
                perturbation
            )

            # 4.
            Y = np.asarray(
                func_Bx(*x_next.numpy().T)
            ).reshape(K, M)

            # 5.
            variances = np.var(Y, axis=1, ddof=1)

            # 6.
            max_var = max(max_var, np.max(variances))

        return max_var

    def calcLip(self, points):

        N, n = points.shape
        expr = self.SBC_model.RealExpr
        x_var = self.SBC_model.var_sym

        # 2.
        grad_expr = [sp.diff(expr, xi) for xi in x_var]

        # 3.
        grad_funcs = [
            sp.lambdify(x_var, g, 'numpy')
            for g in grad_expr
        ]

        # 4.
        grad_values = np.column_stack([
            np.broadcast_to(f(*points.T), (N,))
            for f in grad_funcs
        ])

        # 5.
        grad_norms = np.linalg.norm(grad_values, ord=2, axis=1)

        # 6.
        return np.max(grad_norms)

    def calcLip2(self, area):
        expr = self.SBC_model.RealExpr
        x_var = self.SBC_model.var_sym
        N = 500

        #area = np.array(area)
        lows = area.low  # [low1, low2]
        uppers = area.up  # [upper1, upper2]

        #
        X = lows + (uppers - lows) * np.random.rand(N, self.x_var_num)
        Y = lows + (uppers - lows) * np.random.rand(N, self.x_var_num)

        #
        f = sp.lambdify(x_var, expr, 'numpy')

        #
        fx = np.broadcast_to(f(*X.T), (N,))
        fy = np.broadcast_to(f(*Y.T), (N,))

        #
        numerator = np.abs(fx - fy)
        denominator = np.linalg.norm(X - Y, axis=1)

        # 11.
        valid = denominator > 1e-12
        ratios = np.abs(numerator[valid]) / denominator[valid]

        # ratios = numerator / denominator
        return np.max(ratios)

    def calcLip3(self, area):

        N = 500
        M = self.hat_N

        #area = np.array(area)
        lows = area.low  # [low1, low2]
        uppers = area.up  # [upper1, upper2]

        # 1.
        X = lows + (uppers - lows) * np.random.rand(N, self.x_var_num)
        Y = lows + (uppers - lows) * np.random.rand(N, self.x_var_num)

        # 2.
        expr = self.SBC_model.RealExpr
        x_var = self.SBC_model.var_sym
        f = sp.lambdify(x_var, expr, 'numpy')

        # 3.
        BX = np.asarray(f(*X.T)).reshape(-1)
        BY = np.asarray(f(*Y.T)).reshape(-1)

        # 4.
        #
        W = self.distribution.RV_sample(
            (M, self.random_var_num)
        )

        # 5.
        X_repeat = np.repeat(X, M, axis=0)
        Y_repeat = np.repeat(Y, M, axis=0)

        # 1.5.
        u_x = self.Col_model(
            torch.as_tensor(X_repeat, dtype=torch.float32)
        ).detach()

        u_y = self.Col_model(
            torch.as_tensor(Y_repeat, dtype=torch.float32)
        ).detach()



        # 6.
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().numpy()

        W_repeat = np.tile(W, (N, 1))

        # 7.
        X_next = self.dynamic.forward(
            torch.from_numpy(X_repeat),
            u_x,
            torch.from_numpy(W_repeat)
        )

        Y_next = self.dynamic.forward(
            torch.from_numpy(Y_repeat),
            u_y,
            torch.from_numpy(W_repeat)
        )

        # 8.
        B_X_next = np.asarray(
            f(*X_next.detach().cpu().numpy().T)
        ).reshape(N, M)

        B_Y_next = np.asarray(
            f(*Y_next.detach().cpu().numpy().T)
        ).reshape(N, M)

        # 9.
        diff = (
                (B_X_next - B_Y_next).mean(axis=1)
                - (BX - BY)
        )

        # 10.
        distances = np.linalg.norm(X - Y, axis=1)

        # 11.
        valid = distances > 1e-12
        ratios = np.abs(diff[valid]) / distances[valid]

        # 12.
        return np.max(ratios)


    def G(self, radius, n, R):

        if n < 1 or int(n) != n:
            raise ValueError("n error")
        if R <= 0:
            raise ValueError("R error")
        if radius < 0:
            raise ValueError("radius error")

        if radius == 0:
            return 0.0
        if radius >= 2 * R:
            return 1.0

        c1 = (2 * R ** 2 - radius ** 2) / (2 * R)
        c2 = radius ** 2 / (2 * R)

        a = (n + 1) / 2
        b = 0.5

        z1 = np.clip(1 - (c1 / R) ** 2, 0, 1)
        z2 = np.clip(1 - (c2 / radius) ** 2, 0, 1)

        I1 = betainc(a, b, z1)
        I2 = betainc(a, b, z2)


        H1 = I1 if c1 >= 0 else 2 - I1

        return float(
            0.5 * (H1 + (radius / R) ** n * I2)
        )

    def G_inverse(self, epsilon, n, R):

        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon in [0, 1]")

        if epsilon == 0:
            return 0.0
        if epsilon == 1:
            return 2.0 * R

        return brentq(
            lambda radius: self.G(radius, n, R) - epsilon,
            0.0,
            2.0 * R,
            xtol=1e-12
        )



if __name__ == '__main__':
    epsilon = 3e-6

    area = np.array([[-2, 2],[-2, 2]])

    lows = area[:, 0]  # [low1, low2]
    uppers = area[:, 1]  # [upper1, upper2]

    eta = uppers - lows

    n = len(eta)
    a = np.pi ** (n / 2) / (2 ** n * gamma(n / 2 + 1))

    epsilon_max = a * np.min(eta) ** n / np.prod(eta)

    print((epsilon * np.prod(eta) / a) ** (1 / n))