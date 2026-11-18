import torch
from matplotlib import pyplot as plt

from cellPartitioning.partitioning import Partitioning, Partitions

# !need fix! : 针对不同的动力系统是需要修改的方法
# 绘制分区
def plot_partitions(partitions, dynamics, tag = "X0"):
    """
    绘制分区划分情况
    :param partitions: 分区信息
    :param dynamics: 动力系统
    :param tag: 针对哪个域 (X0,Xu,Xs,X) ?
    :return:
    """

    fig = plt.figure(
        figsize=(8, 8),
        dpi=200
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    lower = partitions.lower.cpu().numpy()
    width = partitions.width.cpu().numpy()

    dynamics.dynamic_overview(ax1)
    for l, w in zip(lower, width):
        rect = plt.Rectangle(l, w[0], w[1], fill = True, color = 'purple', alpha = 0.5)
        ax2.add_patch(rect)

    if tag == "X0":
        rect0 = plt.Rectangle((-0.25, -0.1), 0.05, 0.2, color='red', fill=False)
        rect1 = plt.Rectangle((0.2, -0.1), 0.05, 0.2, color='red', fill=False)
        ax2.add_patch(rect0)
        ax2.add_patch(rect1)

    if tag == "Xu":
        rect0 = plt.Rectangle((-1.5, -1.5), 0.1, 1.5, color='red', fill=False)
        rect1 = plt.Rectangle((1.4, 0.), 0.1, 1.5, color='red', fill=False)
        ax2.add_patch(rect0)
        ax2.add_patch(rect1)

    if tag == "Xs":
        rect0 = plt.Rectangle((-1.5, 0.), 2.9, 1.5, color='red', fill=False)
        rect1 = plt.Rectangle((-1.4, -1.5), 2.9, 1.5, color='red', fill=False)
        ax2.add_patch(rect0)
        ax2.add_patch(rect1)

    if tag == "X":
        rect = plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, color='red')
        ax2.add_patch(rect)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    plt.show()

# !need fix! : 针对不同的动力系统是需要修改的方法
def get_initial_partitioning(config, dynamics):
    '''
    获取目标域的一个初始的划分，以进行后续的工作
    :param config: 配置信息
    :param dynamics: 动力系统
    :return:
    '''
    partitioning_config = config['partitioning']

    x_lower, x_upper = -1.5, 1.5

    # 划分每个维度
    x1_space = torch.linspace(x_lower, x_upper, partitioning_config['initial_num_slices'][0] + 1)
    x2_space = torch.linspace(x_lower, x_upper, partitioning_config['initial_num_slices'][1] + 1)

    # cell的diff (关于中心点 center)
    cell_width = torch.stack([(x1_space[1] - x1_space[0]) / 2, (x2_space[1] - x2_space[0]) / 2])
    x1_slice_centers = (x1_space[:-1] + x1_space[1:]) / 2
    x2_slice_centers = (x2_space[:-1] + x2_space[1:]) / 2

    # 笛卡尔积
    cell_centers = torch.cartesian_prod(x1_slice_centers, x2_slice_centers)

    lower_x, upper_x = cell_centers - cell_width, cell_centers + cell_width

    # 计算覆盖 判断归属
    initial_mask = dynamics.initial(cell_centers, cell_width)
    safe_mask = dynamics.safe(cell_centers, cell_width)
    unsafe_mask = dynamics.unsafe(cell_centers, cell_width)

    partitioning = Partitioning(
        (lower_x[initial_mask], upper_x[initial_mask]),
        (lower_x[safe_mask], upper_x[safe_mask]),
        (lower_x[unsafe_mask], upper_x[unsafe_mask]),
        (lower_x, upper_x)
    )

    # plot_partitioning(partitioning, safe_set_type)

    return partitioning

# 针对正态分布的扰动
# !need fix! : 针对不同的动力系统是需要修改的方法
def get_RV_partitions_deprecated(config):
    """
    此次工作中的扰动都是选取高斯分布 E = miu ,Dx = sigma^2
    v ~ N (miu, sigma ^ 2) or N (miu, sigma) [不同的写法 同样的含义]
    根据切比雪夫不等式
    P{|v - E(v)| < tao} >= 1 - Dx/tao^2

    此次实验中 v的 siga = 0.1
    故取tao = 1作为分界
    此时P >= 0.99

    只考虑RV域 [-1, 1], 在[-1, 1] 可再细分
    """
    v_lower, v_upper = -1., 1.
    # 划分每个维度
    v_space = torch.linspace(v_lower, v_upper, config["partitioning"]["RV_num_slices"] + 1)
    cell_width = torch.stack([(v_space[1] - v_space[0]) / 2])
    v_slice_centers = (v_space[:-1] + v_space[1:]) / 2

    lower_v, upper_v = v_slice_centers - cell_width, v_slice_centers + cell_width

    return Partitions((lower_v, upper_v))

#
def get_RV_partitions(config):
    v_lower, v_upper = -0.1, 0.1
    # 划分每个维度
    v_space = torch.linspace(v_lower, v_upper, config["partitioning"]["RV_num_slices"] + 1)
    cell_width = torch.stack([(v_space[1] - v_space[0]) / 2])
    v_slice_centers = (v_space[:-1] + v_space[1:]) / 2

    lower_v, upper_v = v_slice_centers - cell_width, v_slice_centers + cell_width

    return Partitions((lower_v, upper_v))

