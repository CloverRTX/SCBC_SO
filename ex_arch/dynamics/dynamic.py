import torch

from RL.Env import Zones
from Sample.sampleHelper import Sample_Handler
from dynamics.fx_handler import Fx_Handler
from matplotlib import pyplot as plt
import numpy as np
from utils.RV_triang import TriangularDistribution

class dynamicSystem():

    def __init__(self, config, plot=False):

        self.config = config
        self.name = config["dynamics"]["systemName"]


        # 训练集采样相关
        # self.sampleHelper = Sample_Handler(config)

        # 转移函数fx相关
        self.fxHelper = Fx_Handler(config["dynamics"])
        # fx的可计算的函数
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
            行数据包括[x1, x2, x3, ..., v]
        :param x_point:
        :param v_point:
        :return: 将行数据带入状态转移方程fx后的值
        '''
        return self.fxHelper.fx_calc(self.fx_func, x_point, u_point, v_point)



    # =============== 各区域采样 =============== #
    def data_sample(self, plot = False):
        data_X0 = self.sampleHelper.X0_dataSampling(self.X0, self.x_var_num)
        data_Xu = self.sampleHelper.Xu_dataSampling(self.Xu, self.x_var_num)
        data_Xs = self.sampleHelper.Xs_dataSampling(self.X, self.x_var_num)
        data_X = self.sampleHelper.X_dataSampling(self.X, self.x_var_num)
        if plot:
            self.sample_overview(data_X0, data_Xu, data_Xs, data_X)
        return data_X0, data_Xu, data_Xs, data_X

    # ================= 绘制 ================= #

    # 采样后 绘制各个区域的采样情况
    def sample_overview(self, data_X0, data_Xu, data_Xs, data_X):
        '''
            绘制各个区域的采样数据
            这些数据用于Learner训练
        :return: None
        '''
        fig = plt.figure(
            figsize=(8, 8),
            dpi=200
        )

        if self.x_var_num == 2:
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            self.dynamic_overview2D(ax1)
            self.dynamic_overview2D(ax2)
            self.dynamic_overview2D(ax3)
            self.dynamic_overview2D(ax4)
            ax1.set_title("X0", loc="right")
            ax2.set_title("Xu", loc="right")
            ax3.set_title("Xs", loc="right")
            ax4.set_title("X", loc="right")
            ax1.scatter(data_X0[:, 0], data_X0[:, 1], alpha=1, s=0.5)
            ax2.scatter(data_Xu[:, 0], data_Xu[:, 1], alpha=1, s=0.5)
            ax3.scatter(data_Xs[:, 0], data_Xs[:, 1], alpha=1, s=0.5)
            ax4.scatter(data_X[:, 0], data_X[:, 1], alpha=1, s=0.5)
            plt.show()
        elif self.x_var_num == 3:
            pass
        else:
            print("超过3维，难以绘制图像!!!")
            pass


    # 绘制系统概图
    def dynamic_overview2D(self, ax):
        '''
        绘制系统的 X X0 Xu区域

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
        仅仅绘制某个区域

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


        # 1. 随机生成初始状态
        start_points = lows + (uppers - lows) * np.random.rand(
            num_traj, self.x_var_num
        )

        # 2. 初始化状态
        x_pre = torch.as_tensor(
            start_points,
            dtype=torch.float64
        )

        # 3. 初始化轨迹
        route = [x_pre]

        # 4. 获取控制器的数据类型
        param = next(col.parameters())
        col.eval()

        # 5. 开始仿真
        with torch.no_grad():

            for j in range(step):
                # 随机扰动
                v = self.distribution.RV_sample(
                    (num_traj, self.random_var_num)
                )

                # 转换随机扰动类型
                v = torch.as_tensor(
                    v,
                    dtype=x_pre.dtype,
                    device=x_pre.device
                )

                # 计算控制器输出
                u = col(
                    x_pre.to(
                        dtype=param.dtype,
                        device=param.device
                    )
                )

                # 与动力系统的数据类型保持一致
                u = u.to(
                    dtype=x_pre.dtype,
                    device=x_pre.device
                )

                # 计算下一时刻状态
                x_next = self.forward(x_pre, u, v)
                route.append(x_next)
                x_pre = x_next

        # 6. 整理轨迹
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


    def run_fix_point(self):
        fig = plt.figure(
            figsize=(8, 8),
            dpi=200
        )
        ax = fig.add_subplot(1, 1, 1)
        self.dynamic_overview2D(ax)

        step = 1000
        area = np.array(self.X0)
        lows = area[:, 0]  # [low1, low2]
        uppers = area[:, 1]  # [upper1, upper2]
        # 在 [0,1) 上采样，再映射到各自的区间
        start_points = lows + (uppers - lows) * np.random.rand(1, len(area))
        start_points = np.repeat(start_points, 3, axis=0)

        x_pre = torch.from_numpy(start_points)
        route = [x_pre]

        for j in range(step):
            v = []
            for i in range(self.random_var_num):
                v_temp = self.distribution.RV_sample(x_pre.shape[0]).unsqueeze(1)
                v.append(v_temp)

            v = torch.cat(v, dim=1)
            x_next = self.forward(x_pre, v)
            route.append(x_next)
            x_pre = x_next

        route = torch.cat(route, 1).cpu().detach().numpy()

        for i in range(route.shape[0]):
            path = route[i, :].reshape((-1, self.x_var_num))
            ax.scatter(path[:, 0], path[:, 1], alpha=1, s=1)

        plt.show()

if __name__ == "__main__":
    pass
