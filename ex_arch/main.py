from utils import superpHelper
from dynamics.dynamic import dynamicSystem
from SO.SCP import SCP_solver as SCP_solver
from BFModels import ModelHelper
from utils.polynomial import Polynomial
from RL.train_controller import train_by_ddpg, ddpg_init
from BFModels import Controller


def main():
    # 设置tensor显示精度
    # torch.set_printoptions(precision=12)

    #1. 读取超参数配置文件
    config = superpHelper.read_yaml("config/superp.yaml")

    # 2. 构建黑盒动力系统
    blackBox_dynamics = dynamicSystem(config)


    # 3. 强化学习生成控制器
    #agent, env, replay_buffer = ddpg_init(blackBox_dynamics)
    #tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
    #print(train_time)

    #ModelHelper.save(tmp_controller, config, "RL_control", "First_train")
    #return None

    tmp_controller = Controller.ColNetwork(2, 64, 2, 0.005)
    ModelHelper.load(tmp_controller, config, "RL_control", "First_train")
    # return None
    blackBox_dynamics.run(tmp_controller)
    #return None
    #return None
    # 4. 构建多项式SBC模板
    x_dim = config["Bx_model"]["input_dim"]
    SBC_deg = config["Bx_model"]["deg"]
    SBC_model = Polynomial(x_dim, SBC_deg, 0)

    #print(len(SBC_model.parameters))
    #return None
    # 3. 计算SBC
    scp_solver = SCP_solver(SBC_model, tmp_controller, config, blackBox_dynamics)
    scp_solver.solve()

    return None

    #ModelHelper.sympy_save(scp_solver.SBC_model.RealExpr, config, "poly", "SBC")

    epsilon = 1.5e-5
    n = 3
    R = 0.8
    cc = scp_solver.G_inverse(epsilon, n, R)
    print(cc)




if __name__ == "__main__":
    main()
