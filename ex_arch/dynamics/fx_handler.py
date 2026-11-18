import numpy as np
import sympy as sp
import torch
from dynamics import operator_sym




class Fx_Handler:
    def __init__(self, config):

        self.f_x = config["f_x"]


        self.x_var_str = config["x_var_str"]

        self.x_var_num = config["x_var_num"]

        self.x_var = sp.symbols(self.x_var_str, seq=True)


        self.random_var_str = config["random_var_str"]

        self.random_var_num = config["random_var_num"]

        self.random_var = sp.symbols(self.random_var_str, seq=True)


        self.col_var_str = config["col_var_str"]

        self.col_var_num = config["col_var_num"]

        self.col_var = sp.symbols(self.col_var_str, seq=True)



        self.var_num = config["var_num"]

        self.alpha = config["alpha"]

        # x_next = f(x, u, v)
        self.fx_expr = self.get_fx_expr()


    def get_fx_expr(self):
        f_x = ()
        for i in range(self.x_var_num):
            f_x = f_x + (self.f_x[i],)

        fx_expr_temp = sp.simplify(f_x)

        fx_expr = tuple(self.x_var[i] + self.alpha * fx_expr_temp[i] for i in range(self.x_var_num))

        return fx_expr

    def fx_incubator(self, expr_type = operator_sym.operator_dict):
        '''


        :param expr_type:
        :return: fx_func
        '''
        assert self.var_num > 0, "wrong"
        assert(self.x_var_num + self.col_var_num + self.random_var_num == self.var_num), "wrong"


        print("+----------------------------------------------------------+")
        print("|Sympy")
        for i in range(self.x_var_num):
             #str(i) + "_dot"
            print(f"|\t{self.x_var[i]}_next = {self.fx_expr[i]}")
        print("+----------------------------------------------------------+")



        fx_func = sp.lambdify(self.x_var + self.col_var + self.random_var, self.fx_expr, expr_type)
        #print(fx_func_np)

        return fx_func

    def fx_calc(self, fx_func, x_point, u_point, v_point):
        '''

        :param fx_:
        :param x_point:
                    [x1, x2, x3, ..., xn]
        :param u_point:
                    [u1, u2, u3, ..., un]
        :param v_point:
                    [v1, v2, v3, ..., vn]
        :return:
        '''
        # arr = torch.tensor([[1., 2, 3], [2, 3, 4], [3, 4, 5]], requires_grad = True)
        # u_value = u_value.reshape((u_value.shape[0], 1))

        #
        if x_point.ndim == 1:
            x_point = x_point.unsqueeze(0)

        if u_point.ndim == 1:
            u_point = u_point.unsqueeze(0)

        if v_point.ndim == 1:
            v_point = v_point.unsqueeze(0)

        #
        args = (
            *x_point.unbind(dim=1),
            *u_point.unbind(dim=1),
            *v_point.unbind(dim=1)
        )

        #
        result = fx_func(*args)

        #
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
