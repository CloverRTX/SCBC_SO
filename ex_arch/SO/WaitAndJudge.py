import numpy as np
import torch
import cvxpy as cp

def ScenarioSampling(SBC_model, sample_num, area, device):
    '''
    场景采样
    :param SBC_model: NN_SBC
    :param sample_num: 采样数
    :param area: X
    :return: 采样后的数组  [[--x1 NN(x1)--],[--x2 NN(x2)--],...]
    '''
    area = np.array(area)

    lows = area[:, 0]  # [low1, low2]
    uppers = area[:, 1]  # [upper1, upper2]

    # 在 [0,1) 上采样，再映射到各自的区间
    scenario = lows + (uppers - lows) * np.random.rand(sample_num, len(area))

    # print(scenario.shape)
    tensor_scenario = torch.from_numpy(scenario).float().to(device)

    # print(SBC_model(tensor_scenario))
    tensor_out = SBC_model(tensor_scenario)

    # 拼接
    combined = torch.cat([tensor_scenario, tensor_out], dim=1)

    out = combined.detach().cpu().numpy()

    return out


def wait_and_judge_cvx(scenario, poly):
    '''

    :param scenario: 场景采样
    :param poly: 多项式
    :return: 求解完成后的P(x)
    '''

    # ————————————————————————————定义变量————————————————————————————
    e = cp.Variable(nonneg=True)

    # 多项式参数
    matrix, poly_param = poly.substitute(scenario)
    Bx = scenario[:, -1]
    # 多项式系数在CVX中的变量表示形式
    param = cp.Variable(len(poly_param))

    # ————————————————————————————添加约束————————————————————————————
    constraints = [matrix @ param >= Bx - e, matrix @ param <= Bx + e]

    obj = cp.Minimize(e)

    prob = cp.Problem(obj, constraints)

    # ————————————————————————————求解————————————————————————————
    # print(prob.constraints)

    print("============SO solving============")
    # 凸性判断
    print("DCP: ", prob.is_dcp())
    prob.solve(solver=cp.MOSEK)
    print("status:", prob.status)
    print(f"SO opt value (e value) = {prob.value}")
    print("============SO end============")

    param_value = [param[i].value for i in range(len(poly_param))]

    # 将系数值代入多项式
    poly.RealExpression(param_value)


def wait_and_judge_gurobi(scenario, poly):
    '''

    :param scenario: 场景采样
    :param poly: 多项式
    :return:
    '''


    # debug
    # m.computeIIS()
    # m.write("model1.ilp")


    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    # m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    # ————————————————————————————定义变量————————————————————————————
    e = m.addVar(vtype=GRB.CONTINUOUS, lb=0)

    # 多项式参数
    matrix, poly_param = poly.substitute(scenario)

    pass


    para_len = Bx.input_dim

    para = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0')





    m.setObjective(y_layer, GRB.MINIMIZE)
    # 优化
    m.optimize()

    # m.computeIIS()
    # m.write("model1.ilp")
    counter_ex = solution_output(m)

    if Xu_counter_ex_T_F(Bx, counter_ex):
        # 不存在反例
        return True, []
    else:
        # 存在反例
        return False, counter_ex



        # 不存在反例
        # if m.objVal > landa:
        #    return True, []
        # 存在反例
        # else:
        #    return False, counter_ex
