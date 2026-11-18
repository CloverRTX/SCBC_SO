import torch
from RL.Env import Zones
from dynamics.fx_handler import Fx_Handler
from matplotlib import pyplot as plt
import numpy as np
from utils.RV_triang import TriangularDistribution

class dynamicSystem():

    def __init__(self, config, plot=False):

        self.config = config
        self.name = config["dynamics"]["systemName"]


        # self.sampleHelper = Sample_Handler(config)


        self.fxHelper = Fx_Handler(config["dynamics"])

        self.fx_func = self.fxHelper.fx_incubator()
        self.fx_expr = self.fxHelper.fx_expr

        self.x_var_num = self.fxHelper.x_var_num
        self.u_var_num = self.fxHelper.col_var_num
        self.random_var_num = self.fxHelper.random_var_num

        self.alpha = config["dynamics"]["alpha"]
        self.device = config["device"]
        self.goal = config["dynamics"]["goal"]

        # self.X = config["dynamics"]["X"]
        # self.X0 = config["dynamics"]["X0"]
        # self.Xu = config["dynamics"]["Xu"]
        self.D_zones = Zones('ball', center=[0, 0], r=2)
        self.I_zones = Zones('ball', center=[1.25, 1.25], r=0.2)
        self.U_zones = Zones('ball', center=[0., 0.], r=0.2)

        self.Big_D_zones = Zones('box', low=[-2, -2], up=[2, 2])

        self.max_episode = config["dynamics"]["max_episode"]

        self.u = config["dynamics"]["col_bound"]
        # self.f = None
        self.dense = None
        self.units = None

        self.distribution = TriangularDistribution()
        if plot:
            pass


    def forward(self, x_point, u_point, v_point):
        '''
            [x1, x2, x3, ..., v]
        :param x_point:
        :param v_point:
        :return:
        '''
        return self.fxHelper.fx_calc(self.fx_func, x_point, u_point, v_point)


    def dynamic_overview2D(self, ax):
        '''
        :param ax:
        :return:
        '''
        X0_area = plt.Circle(self.I_zones.center, self.I_zones.r, fill=True, color='green', alpha=0.2)

        Xu_area = plt.Circle(self.U_zones.center, self.U_zones.r, fill=True, color='red', alpha=0.2)

        # Xu_area = plt.Circle(self.U_zones.center, self.U_zones.r, fill=False, color='red', alpha=0.2)

        X_area = plt.Circle(self.D_zones.center, self.D_zones.r, fill=True, color='grey', alpha=0.2)

        ax.add_patch(X_area)
        ax.add_patch(X0_area)
        ax.add_patch(Xu_area)

        ax.set_xlim(self.Big_D_zones.low[0] - 0.1, self.Big_D_zones.up[0] + 0.1)
        ax.set_ylim(self.Big_D_zones.low[1] - 0.1, self.Big_D_zones.up[1] + 0.1)
        ax.set_aspect('equal')


    def sample_view(self, data):
        '''

        :param data:
        :return:
        '''
        fig = plt.figure(
            figsize=(8, 8),
            dpi=200
        )

        ax = fig.add_subplot(1, 1, 1)
        self.dynamic_overview2D(ax)
        ax.set_title("scenario", loc="right")
        ax.scatter(data[:, 0], data[:, 1], alpha=1, s=0.5)
        plt.show()

    def run(self, col):
        fig = plt.figure(
            figsize=(8, 8),
            dpi=200
        )
        ax = fig.add_subplot(111)
        self.dynamic_overview2D(ax)

        step = 100
        num_traj = 80

        #area = self.I_zones
        #lows = area.low  # [low1, low2]
        #uppers = area.up  # [upper1, upper2]
        lows = np.array([1.05, 1.05])
        uppers = np.array([1.45, 1.45])


        # 1.
        start_points = lows + (uppers - lows) * np.random.rand(
            num_traj, self.x_var_num
        )

        # 2.
        x_pre = torch.as_tensor(
            start_points,
            dtype=torch.float64
        )

        # 3.
        route = [x_pre]

        # 4.
        param = next(col.parameters())
        col.eval()

        # 5.
        with torch.no_grad():

            for j in range(step):

                v = self.distribution.RV_sample(
                    (num_traj, self.random_var_num)
                )


                v = torch.as_tensor(
                    v,
                    dtype=x_pre.dtype,
                    device=x_pre.device
                )


                u = col(
                    x_pre.to(
                        dtype=param.dtype,
                        device=param.device
                    )
                )


                u = u.to(
                    dtype=x_pre.dtype,
                    device=x_pre.device
                )


                x_next = self.forward(x_pre, u, v)
                route.append(x_next)
                x_pre = x_next

        # 6.
        route = torch.stack(route, dim=1)
        route = route.cpu().numpy()

        for i in range(route.shape[0]):
            path = route[i]

            ax.scatter(
                path[:, 0],
                path[:, 1],
                alpha=1,
                s=1
            )

        plt.show()

if __name__ == "__main__":
    pass
