from utils import superpHelper
from dynamics.dynamic import dynamicSystem
from SO.SCP import SCP_solver as SCP_solver
from BFModels import ModelHelper
from utils.polynomial import Polynomial
from RL.train_controller import train_by_ddpg, ddpg_init
from BFModels import Controller


def main():

    # torch.set_printoptions(precision=12)

    config = superpHelper.read_yaml("config/superp.yaml")

    blackBox_dynamics = dynamicSystem(config)


    #agent, env, replay_buffer = ddpg_init(blackBox_dynamics)
    #tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
    #print(train_time)

    #ModelHelper.save(tmp_controller, config, "RL_control", "First_train")
    #return None

    tmp_controller = Controller.ColNetwork(2, 64, 2, 0.005)
    ModelHelper.load(tmp_controller, config, "RL_control", "First_train")

    #blackBox_dynamics.run(tmp_controller)

    x_dim = config["Bx_model"]["input_dim"]
    SBC_deg = config["Bx_model"]["deg"]
    SBC_model = Polynomial(x_dim, SBC_deg, 0)

    scp_solver = SCP_solver(SBC_model, tmp_controller, config, blackBox_dynamics)
    scp_solver.solve()

    #ModelHelper.sympy_save(scp_solver.SBC_model.RealExpr, config, "poly", "SBC")

if __name__ == "__main__":
    main()
