import torch
from cellPartitioning.partitioningFuncs import get_initial_partitioning, plot_partitions
from utils.Crown_IBP_API import crown
from utils.Crown_IBP_API import Affine
from cellPartitioning.partitioning import Partitions
from assistNetworks.networks import MEConditionModule
from matplotlib import pyplot as plt
import numpy as np
import csv

class CBFCertifier:
    def __init__(self,
                 config,
                 dynamics,
                 bounded_CBC_model,
                 bounded_Col_model,
                 vol_gap_threshold=0.0001 * 0.0001,
                 split_gap_stop_threshold=1e-8,
                 max_set_size=10000,
                 plot_partitions = False
                 ):

        self.config = config
        self.dynamics = dynamics
        self.device = config["device"]

        self.bounded_CBC_model = bounded_CBC_model
        self.bounded_Col_model = bounded_Col_model

        # 现有的各个区域的初始划分划分
        self.initial_partitioning = self.Initial_Partitions()

        self.vol_gap_threshold = vol_gap_threshold
        self.split_gap_stop_threshold = split_gap_stop_threshold
        self.max_set_size = max_set_size
        self.plot_partitions = plot_partitions
        self.k_induction = config["k_induction"]

    def Initial_Partitions(self):
        return get_initial_partitioning(self.config, self.dynamics).to(self.device)

    def certify(self):

        #ME_upper = self.Verify_ME_Condition()
        eye_Xu, counterexample_Xu = self.Verify_Xu_Condition()
        if not eye_Xu:
            print("不安全区域验证失败")
            filename = self.config["sample_config"]["path_Xu"]
            self.writeToCsv(filename, counterexample_Xu.center)
        else:
            print("不安全区域验证成功")

        eye_X, counterexample_X = self.Verify_X_Condition()
        if not eye_X:
            print("全域验证失败")
            filename = self.config["sample_config"]["path_X"]
            self.writeToCsv(filename, counterexample_X.center)
        else:
            print("全域验证成功")

        eye_kIME, counterexample_kIME = self.Verify_KI_ME_Condition()
        if not eye_kIME:
            print("kI数学期望条件验证失败")
            filename = self.config["sample_config"]["path_Xs"]
            self.writeToCsv(filename, counterexample_kIME.center)
        else:
            print("kI数学期望条件验证成功")
        # plot
        self.counterexample_plot([eye_Xu, eye_X, eye_kIME], [counterexample_Xu, counterexample_X, counterexample_kIME])

        if eye_Xu and eye_X and eye_kIME:
            gama = self.Verify_X0_Condition()
            ME_upper = self.Verify_ME_Condition()
            print(f"gama = {gama}")
            print(f"ME_upper = {ME_upper}")
            return True, 1 - (self.k_induction * gama + self.k_induction * (self.k_induction - 1) * ME_upper / 2.0)
        else:
            return False, None

    @torch.no_grad()
    # 计算初始域上界
    def Verify_X0_Condition(self):
        assert self.initial_partitioning.initial is not None
        return self.Bx_upper_calc(self.get_linear_bound,
                                  self.initial_partitioning.initial,
                                  self.dynamics.initial, Tag="X0")


    @torch.no_grad()
    # 验证安全区域条件 （数学期望条件）
    def Verify_ME_Condition(self):
        assert self.initial_partitioning.safe is not None
        return self.Bx_upper_calc(self.get_linear_bound_forME,
                             self.initial_partitioning.safe,
                             self.dynamics.safe, Tag = "Xs")

    @torch.no_grad()
    # 验证安全区域条件 （kI数学期望条件）
    def Verify_KI_ME_Condition(self):
        assert self.initial_partitioning.safe is not None
        return self.Bx_upper_verify(self.get_linear_bound_for_kIME,
                                  self.initial_partitioning.safe,
                                  0.0, self.dynamics.safe, Tag="Xs")


    @torch.no_grad()
    # 验证不安全条件
    def Verify_Xu_Condition(self):
        assert self.initial_partitioning.unsafe is not None
        return self.Bx_lower_verify(self.get_linear_bound,
                                    self.initial_partitioning.unsafe,
                                    1.0, self.dynamics.unsafe, Tag="Xu")

    @torch.no_grad()
    # 验证全域条件
    def Verify_X_Condition(self):
        assert self.initial_partitioning.unsafe is not None
        return self.Bx_lower_verify(self.get_linear_bound,
                                    self.initial_partitioning.state_space,
                                    0.0, self.dynamics.state_space, Tag="X")

    # ============================分枝定界============================

    def get_linear_bound(self, partitions):
        center = partitions.center
        eps = partitions.diff
        result_dict = crown(self.bounded_CBC_model, center, eps, bound_lower=True, bound_upper=True, needed_A=True)

        lb_A = result_dict["lb_A"]
        lb_bias = result_dict["lb_bias"]
        ub_A = result_dict["ub_A"]
        ub_bias = result_dict["ub_bias"]

        lb_Affine = Affine(lb_A, lb_bias, partitions.lower, partitions.upper)
        ub_Affine = Affine(ub_A, ub_bias, partitions.lower, partitions.upper)

        return lb_Affine, ub_Affine, result_dict["lb"], result_dict["ub"]

    # 重点
    def get_linear_bound_forME(self, partitions):
        MEConHelper = MEConditionModule(self.config, self.bounded_CBC_model, self.bounded_Col_model, self.dynamics)
        result_dict = MEConHelper.getLinearBoundOfMECon(partitions)
        lb_Affine = result_dict["Affine_lower"]
        ub_Affine = result_dict["Affine_upper"]

        return lb_Affine, ub_Affine

    # 重点
    def get_linear_bound_for_kIME(self, partitions):
        MEConHelper = MEConditionModule(self.config, self.bounded_CBC_model, self.bounded_Col_model, self.dynamics)
        result_dict = MEConHelper.getLinearBoundOf_kIMECon(partitions)
        lb = result_dict["Affine_lower"]
        ub = result_dict["Affine_upper"]
        return lb, ub

    def split(self, partitions, lb_Affine, ub_Affine):
        """
        细分partitions
        :param partitions: 待split的Partitions
        :return: 对每个Cell找到合适的维度的中点，然后划分更细致的Partitions
                 返回新的Partitions
        """

        #lower, upper = self.get_linear_bound(partitions)
        # lb_Affine, ub_Affine, lb , ub = self.get_linear_bound(partitions)

        split_dim = ((lb_Affine.A.abs() + ub_Affine.A.abs())[:, 0] * partitions.width).argmax(dim=-1)

        partition_indices = torch.arange(0, len(partitions), device=self.device)
        split_dim = (partition_indices, split_dim)

        lower, upper = partitions.lower, partitions.upper
        mid = partitions.center

        p1_lower = lower.clone()
        p1_upper = upper.clone()
        p1_upper[split_dim] = mid[split_dim]

        p2_lower = lower.clone()
        p2_upper = upper.clone()
        p2_lower[split_dim] = mid[split_dim]

        lower, upper = torch.cat([p1_lower, p2_lower]), torch.cat([p1_upper, p2_upper])

        assert torch.all(lower <= upper + 1e-8)

        return Partitions((lower, upper))

    # 与目标域范围比对
    def region_prune(self, partitions, contain_func):
        center = partitions.center
        epsilon = partitions.diff

        overlaps = contain_func(center, epsilon)

        return Partitions((partitions.lower[overlaps], partitions.upper[overlaps]))

    # ====================================================
    # 全域条件和不安全区域条件
    # ====================================================
    @torch.no_grad()
    # 验证Bx >= lower_bound 条件 （适用于 障碍证书 的 全域条件 不安全条件）
    def Bx_lower_verify(self, get_linear_method, partitions, lower_bound, contain_func, Tag = None):

        '''
        :param partitions: 归属域的cell集
        :param lower_bound: 目标要求 >= lower_bound
        :param contain_func: 归属于判断条件
        :return:
        '''
        lb_Affine_lower = None
        partitions_split = partitions
        while True:
            # 有一个初始分区

            # 对每个q进行crown
            lb_Affine, ub_Affine, lb, ub = get_linear_method(partitions_split)

            lb_Affine_lower = lb.view(-1)

            # 修减
            partitions_prune, prune_all = self.prune_for_Bx_lower_verify(partitions_split, lb_Affine_lower, ub, lower_bound)

            if prune_all:
                # 全部都可以修剪
                break

            lb_Affine, ub_Affine, lb, ub = get_linear_method(partitions_prune)

            gap = (ub - lb).min()
            vol_gap = partitions_prune.volumes.min()

            lb_Affine_lower = lb.view(-1)

            # 停止继续划分
            if self.should_stop_for_Bx_lower_verify(partitions_prune, lb_Affine_lower, lower_bound, gap, vol_gap):
                break

            # 细分
            partitions_split = self.split(partitions_prune, lb_Affine, ub_Affine)
            # plot_partitions(partitions, self.dynamics, Tag)

            # 与目标域比对
            partitions_split = self.region_prune(partitions_split, contain_func)
            # plot_partitions(partitions, self.dynamics, Tag)

        if lb_Affine_lower.min() - lower_bound >= 0:
            return True, None
        else:
            print(len(partitions_prune))
            index = lb_Affine_lower - lower_bound < 0.
            # print(partitions.center[index])
            return False, Partitions((partitions_prune.lower[index], partitions_prune.upper[index]))

    def should_stop_for_Bx_lower_verify(self, partitions, lb_Affine_lower, lower_bound, gap, vol_gap):
        """
        :param partitions: 区域所对应的Partitions（Cell集合）
        :param lb_Affine_lower: 根据区间近似算法得到Partitions的下线性边界 的下界
        :param lower_bound: 障碍证书条件边界
        :param gap: 划分细度
        :param vol_gap: cell超体积大小
        :return: 是否需要继续细分 Partitions; True: 表示停止后续细分
        (Partitions中Cell个数是否超过上界) or
        (Partitions是否已经满足障碍证书条件 (Bx >= 1.) ) or
        {(是否达到划分细度上限) and (cell超体积达到下限)}
        """

        # 刻画cell的划分细度
        #gap = (ub_Affine_lower.min() - lb_Affine_lower.min()).item()

        abs_min = lb_Affine_lower.min().item()

        # logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, lower bound: {min.min().item()}')
        # print(f'[DEBUG][{label}] Gap: {gap}, set size: {len(set)}, lower bound: {min.min().item()}')

        return len(partitions) > self.max_set_size or \
               abs_min >= lower_bound or \
               (gap <= self.split_gap_stop_threshold or vol_gap <= self.vol_gap_threshold)

    def prune_for_Bx_lower_verify(self, partitions, lb_Affine_lower, ub, lower_bound):
        """
        :param partitions: 区域所对应的Partitions（Cell集合）
        :param lb_Affine_lower: 根据区间近似算法得到Partitions的边界下线性边界 的下界
        :param lower_bound: 障碍证书边界条件
        :return: 1、全部Partitions都符合障碍证书条件  都可以忽略不考虑了
                 2、返回修剪后的Partitions
        """
        lst = [False for i in range(len(partitions))]
        min_upper = np.inf
        index = -1
        for i in range(len(partitions)):
            if ub[i, 0] < min_upper:
                min_upper = ub[i, 0]
                index = i
        for i in range(len(partitions)):
            if i != index and min_upper < lb_Affine_lower[i]:
                lst[i] = True

        '''
        
        lst = [False for i in range(len(partitions))]
        for i in range(len(partitions)):
            if lst[i] == False:
                for j in range(len(partitions)):
                    if ub[i, 0] < lb_Affine_lower[j] and i != j and lst[j] == False:
                        lst[j] = True
        '''

        prune = (lb_Affine_lower >= lower_bound) | torch.tensor(lst).to(self.device)
        keep = ~prune

        # 全部都可以修剪
        if torch.all(prune):
            return partitions, True

        # 修剪
        return Partitions((partitions.lower[keep], partitions.upper[keep])), False

    # ====================================================
    # kI数学期望条件
    # ====================================================

    @torch.no_grad()
    # 验证Bx <= upper_bound 条件 （适用于 障碍证书 的 kI数学期望条件）
    def Bx_upper_verify(self, get_linear_method, partitions, upper_bound, contain_func, Tag = None):

        '''
        :param get_linear_method:
        :param partitions: 归属域的cell集
        :param upper_bound: 目标要求 >= lower_bound
        :param contain_func: 归属于判断条件
        :return:
        '''
        ub_Affine_upper = None
        partitions_split = partitions
        while True:
            # 有一个初始分区

            # 对每个q进行crown
            lb_Affine, ub_Affine = get_linear_method(partitions_split)
            lb = lb_Affine.lower_Ax_b()
            ub = ub_Affine.upper_Ax_b()
            ub_Affine_upper = ub.view(-1)

            # 修减
            partitions_prune, prune_all = self.prune_for_Bx_upper_verify(partitions_split, lb, ub_Affine_upper, upper_bound)

            if prune_all:
                # 全部都可以修剪
                break

            if len(partitions_prune) != len(partitions_split):
                lb_Affine, ub_Affine = get_linear_method(partitions_prune)
            lb = lb_Affine.lower_Ax_b()
            ub = ub_Affine.upper_Ax_b()
            gap = (ub - lb).min()
            vol_gap = partitions_prune.volumes.min()
            ub_Affine_upper = ub.view(-1)
            print(ub_Affine_upper.max())

            # 停止继续划分
            if self.should_stop_for_Bx_upper_verify(partitions_prune, ub_Affine_upper, upper_bound, gap, vol_gap):
                break

            # 需要继续划分
            # 细分
            partitions_split = self.split(partitions_prune, lb_Affine, ub_Affine)
            # plot_partitions(partitions_split, self.dynamics, Tag)

            # 与目标域比对
            partitions_split = self.region_prune(partitions_split, contain_func)
            # plot_partitions(partitions_split, self.dynamics, Tag)

        if ub_Affine_upper.max() - upper_bound <= 1e-8:
            return True, None
        print(len(partitions_prune))
        index = ub_Affine_upper - upper_bound > 0.
        # print(partitions.center[index])
        return False, Partitions((partitions_prune.lower[index], partitions_prune.upper[index]))

    def should_stop_for_Bx_upper_verify(self, partitions, ub_Affine_upper, upper_bound, gap, vol_gap):
        """
        :param partitions: 区域所对应的Partitions（Cell集合）
        :param ub_Affine_upper: 根据区间近似算法得到Partitions的上线性边界 的上界
        :param upper_bound: 障碍证书条件边界
        :param gap: 划分细度
        :param vol_gap: 超体积
        :return: 是否需要继续细分 Partitions
        (Partitions中Cell个数是否超过上界) or
        (Partitions是否已经满足障碍证书条件 (E - Bx <= ?) ) or
        {(是否达到划分细度上限) and (cell超体积超过上限)}
        """

        abs_max = ub_Affine_upper.max().item()

        # logger.debug(f'[{label}] Gap: {gap}, set size: {len(set)}, lower bound: {min.min().item()}')
        # print(f'[DEBUG][{label}] Gap: {gap}, set size: {len(set)}, lower bound: {min.min().item()}')

        return len(partitions) > self.max_set_size or \
               abs_max <= upper_bound or \
               (gap <= self.split_gap_stop_threshold or vol_gap <= self.vol_gap_threshold)

    def prune_for_Bx_upper_verify(self, partitions, lb, ub_Affine_upper, upper_bound):
        """
        :param partitions: 区域所对应的Partitions（Cell集合）
        :param ub_Affine_upper: 根据区间近似算法得到Partitions的边界上线性边界 的上界
        :param upper_bound: 障碍证书边界条件
        :return: 1、全部Partitions都符合障碍证书条件  都可以忽略不考虑了
                 2、返回修剪后的Partitions
        """
        lst = [False for i in range(len(partitions))]
        max_low = -np.inf
        index = -1
        for i in range(len(partitions)):
            if lb[i, 0] > max_low:
                max_low = lb[i, 0]
                index = i
        for i in range(len(partitions)):
            if i != index and max_low > ub_Affine_upper[i]:
                lst[i] = True
        '''
        for i in range(len(partitions)):
            if lst[i] == False:
                for j in range(len(partitions)):
                    if lb[i, 0] > ub_Affine_upper[j] and i != j and lst[j] == False:
                        lst[j] = True
        '''

        prune = (ub_Affine_upper <= upper_bound + 1e-8) | torch.tensor(lst).to(self.device)
        keep = ~prune

        # 全部都可以修剪
        if torch.all(prune):
            return partitions, True

        # 修剪
        return Partitions((partitions.lower[keep], partitions.upper[keep])), False


    # ====================================================
    # 初始条件 和 数学期望条件
    # ====================================================
    @torch.no_grad()
    # 计算Bx <= upper_bound  （适用于 障碍证书 的 初始域条件）
    def Bx_upper_calc(self, get_linear_method, partitions, contain_func, Tag=None):

        '''
        :param get_linear_method:
        :param partitions: 归属域的cell集
        :param contain_func: 归属于判断条件
        :return:
        '''

        ub_Affine_upper = None
        while True:
            # 有一个初始分区
            # 对每个q进行crown

            if Tag == 'Xs':
                lb_Affine, ub_Affine = get_linear_method(partitions)
            else:
                lb_Affine, ub_Affine, lb, ub = get_linear_method(partitions)
            lb = lb_Affine.lower_Ax_b()
            ub = ub_Affine.upper_Ax_b()

            ub_Affine_upper = ub.view(-1)

            # 修减
            partitions, prune_all = self.prune_for_Bx_upper_calc(partitions, lb, ub_Affine_upper)

            if prune_all:
                # 全部都可以修剪
                break

            if Tag == 'Xs':
                lb_Affine, ub_Affine = get_linear_method(partitions)
            else:
                lb_Affine, ub_Affine, lb, ub = get_linear_method(partitions)

            # lb_Affine, ub_Affine, lb, ub = get_linear_method(partitions)
            lb = lb_Affine.lower_Ax_b()
            ub = ub_Affine.upper_Ax_b()
            gap = (ub - lb).min()
            vol_gap = partitions.volumes.min()
            ub_Affine_upper = ub.view(-1)

            # 停止继续划分
            if self.should_stop_for_Bx_upper_calc(partitions, gap, vol_gap):
                break

            # 细分
            partitions = self.split(partitions, lb_Affine, ub_Affine)
            # plot_partitions(partitions, self.dynamics, Tag)

            # 与目标域比对
            partitions = self.region_prune(partitions, contain_func)
            # plot_partitions(partitions, self.dynamics, Tag)

        return ub_Affine_upper.max()

    def should_stop_for_Bx_upper_calc(self, partitions, gap, vol_gap):
        """
        :param partitions: 区域所对应的Partitions（Cell集合）
        :param gap: 上次划分细度
        :param vol_gap: cell超体积
        :return: 是否需要继续细分 Partitions
        (Partitions中Cell个数是否超过上界) or
        {(是否达到划分细度上限) and (cell超体积达到上限)}
        """
        return len(partitions) > self.max_set_size or \
               (gap <= self.split_gap_stop_threshold or vol_gap <= self.vol_gap_threshold)

    def prune_for_Bx_upper_calc(self, partitions, lb, ub_Affine_upper):
        """
        :param partitions: 区域所对应的Partitions（Cell集合）
        :param lb:
        :param upper_bound:
        :return: partitions
        """
        lst = [False for i in range(len(partitions))]
        for i in range(len(partitions)):
            if lst[i] == False:
                for j in range(len(partitions)):
                    if lb[i, 0] > ub_Affine_upper[j] and i != j and lst[j] == False:
                        lst[j] = True

        #largest_lower_bound = lb_Affine_upper.max()

        prune = torch.tensor(lst).to(self.device)
        keep = ~prune

        # 全部都可以修剪
        if torch.all(prune):
            return partitions, True

        # 修剪
        return Partitions((partitions.lower[keep], partitions.upper[keep])), False

    # ============================分枝定界_end============================


    # ============================辅助函数============================
    # 反例添加
    def writeToCsv(self, filename, data):
        if len(data) == 0:
            return
        data = data.cpu().numpy()
        num = 0
        with open(filename, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            for i in range(data.shape[0]):
                num = num + 1
                csv_write.writerow(data[i, :])
            print(f"{filename},更新了{num}条记录")

    # 反例绘制
    def counterexample_plot(self, eye_lst, counterexample_lst):
        fig = plt.figure(
            figsize=(8, 8),
            dpi=200
        )
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        self.dynamics.dynamic_overview(ax1)
        self.dynamics.dynamic_overview(ax2)
        self.dynamics.dynamic_overview(ax3)
        ax_lst = [ax1, ax2, ax3]

        for i in range(len(eye_lst)):
            if not eye_lst[i]:
                partitions = counterexample_lst[i]
                center = partitions.center.cpu().numpy()
                lower = partitions.lower.cpu().numpy()
                width = partitions.width.cpu().numpy()

                ax_lst[i].scatter(center[:, 0], center[:, 1], alpha=1, s=1)
                for l, w in zip(lower, width):
                    rect = plt.Rectangle(l, w[0], w[1], fill=True, color='purple', alpha=0.5)
                    ax_lst[i].add_patch(rect)
            else:
                ax_lst[i].set_title("No counterexamples", loc="right")
        plt.show()