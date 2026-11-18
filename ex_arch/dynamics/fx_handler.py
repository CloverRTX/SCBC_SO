import numpy as np
import sympy as sp
import torch
from dynamics import operator_sym


# 转移函数fx类

class Fx_Handler:
    def __init__(self, config):
        # 转移方程
        self.f_x = config["f_x"]

        # 状态——变量串
        self.x_var_str = config["x_var_str"]
        # 状态——变量个数
        self.x_var_num = config["x_var_num"]
        #状态——变量
        self.x_var = sp.symbols(self.x_var_str, seq=True)

        # 随机变量——变量串
        self.random_var_str = config["random_var_str"]
        # 随机变量——变量个数
        self.random_var_num = config["random_var_num"]
        # 随机变量——变量
        self.random_var = sp.symbols(self.random_var_str, seq=True)

        # 控制变量——变量串
        self.col_var_str = config["col_var_str"]
        # 控制变量——变量个数
        self.col_var_num = config["col_var_num"]
        # 控制变量——变量
        self.col_var = sp.symbols(self.col_var_str, seq=True)


        # 变量总个数
        self.var_num = config["var_num"]

        self.alpha = config["alpha"]

        # x_next = f(x, u, v)
        self.fx_expr = self.get_fx_expr()


    def get_fx_expr(self):
        f_x = ()
        for i in range(self.x_var_num):
            f_x = f_x + (self.f_x[i],)
        # string 转 sympy表达式
        fx_expr_temp = sp.simplify(f_x)

        fx_expr = tuple(self.x_var[i] + self.alpha * fx_expr_temp[i] for i in range(self.x_var_num))

        return fx_expr

    def fx_incubator(self, expr_type = operator_sym.operator_dict):
        '''
            产生 fx 表达式

        :param expr_type: 在此基础上转化sympy表达式，默认是numpy
        :return: fx_func
        '''
        assert self.var_num > 0, "无效的状态变量x"
        assert(self.x_var_num + self.col_var_num + self.random_var_num == self.var_num), "变量个数不一致"

        # 输出，检验转换是否正确
        print("+----------------------------------------------------------+")
        print("|转换后的Sympy表达式 ：")
        for i in range(self.x_var_num):
             #str(i) + "_dot"
            print(f"|\t{self.x_var[i]}_next = {self.fx_expr[i]}")
        print("+----------------------------------------------------------+")

        # sympy表达式转函数，加快计算
        # 用torch运算替换
        fx_func = sp.lambdify(self.x_var + self.col_var + self.random_var, self.fx_expr, expr_type)
        #print(fx_func_np)

        return fx_func

    def fx_calc(self, fx_func, x_point, u_point, v_point):
        '''

        :param fx_: 向量场
        :param x_point: 传入的状态值，X点的坐标
                    一般是一个tensor数组
                    一行代表一个状态X的坐标
                    列数表示X的维度
                    针对例子不同  该行数据包括[x1, x2, x3, ..., xn]
        :param u_point: 传入的控制器值
                    一般是一个tensor数组
                    一行代表一个控制输入
                    列数表示控制器的维度
                    针对例子不同  该行数据包括[u1, u2, u3, ..., un]
        :param v_point: 传入的扰动值
                    一般是一个tensor数组
                    一行代表一个扰动输入
                    列数表示扰动的维度
                    针对例子不同  该行数据包括[v1, v2, v3, ..., vn]
        :return: 计算结果
                一般是一个tensor数组
                一行代表状态X的向量场
                每列代表对应分量

        '''
        # arr = torch.tensor([[1., 2, 3], [2, 3, 4], [3, 4, 5]], requires_grad = True)
        # u_value = u_value.reshape((u_value.shape[0], 1))

        # 兼容单个状态点
        if x_point.ndim == 1:
            x_point = x_point.unsqueeze(0)

        if u_point.ndim == 1:
            u_point = u_point.unsqueeze(0)

        if v_point.ndim == 1:
            v_point = v_point.unsqueeze(0)

        # 提取每个状态、控制和扰动分量
        args = (
            *x_point.unbind(dim=1),
            *u_point.unbind(dim=1),
            *v_point.unbind(dim=1)
        )

        # 批量计算动力系统
        result = fx_func(*args)

        # 合并状态分量
        x_next = torch.stack(result, dim=1)

        return x_next

if __name__ == "__main__":
    from utils import superpHelper
    config = superpHelper.read_yaml("../config/superp.yaml")
    fxHelper = Fx_Handler(config["dynamics"])
    fx_func = fxHelper.fx_incubator()
    fx_expr = fxHelper.fx_expr

    x_point = torch.tensor([[1, 2], [2, 3]])
    u_point = torch.tensor([[1, 2], [2, 3]])
    v_point = torch.tensor([[1, 2], [2, 3]])

    result = fxHelper.fx_calc(fx_func, x_point, u_point, v_point)
    print(result)
