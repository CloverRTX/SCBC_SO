import numpy as np
import pandas as pd
import torch

class Sample_Handler():

    def __init__(self, config):
        self.path_X0 = config["sample_config"]["path_X0"]
        self.path_Xs = config["sample_config"]["path_Xs"]
        self.path_Xu = config["sample_config"]["path_Xu"]
        self.path_X = config["sample_config"]["path_X"]

        self.X0_sample_len = config["sample_config"]["X0_sample_len"]
        self.Xs_sample_len = config["sample_config"]["Xs_sample_len"]
        self.Xu_sample_len = config["sample_config"]["Xu_sample_len"]
        self.X_sample_len = config["sample_config"]["X_sample_len"]

        # self.eps = config["sample_config"]["small_eps"]

    def X0_dataSampling(self, X0, x_var_num, need_write = True):
        '''
        初始区域的采样

        :param need_write: 记录采样得到的数据
        :return: 返回采样的数据
        '''

        lst = []
        for i in range(x_var_num):
            x_data = np.linspace(X0[i][0], X0[i][1], self.X0_sample_len)
            lst.append(x_data)

        s = np.array(np.meshgrid(*lst))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        if need_write:
            result = np.unique(s, axis=0)
            pd.DataFrame(result).to_csv(self.path_X0, header=None, index=None)
        return torch.tensor(s).float()

    def Xu_dataSampling(self, Xu, x_var_num, need_write = True):
        '''
            不安全区域的采样

            :param need_write: 记录采样得到的数据
            :return: 返回采样的数据
        '''

        lst = []
        for i in range(x_var_num):
            x_data = np.linspace(Xu[i][0], Xu[i][1], self.Xu_sample_len)
            lst.append(x_data)

        s = np.array(np.meshgrid(*lst))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        if need_write:
            result = np.unique(s, axis=0)
            pd.DataFrame(result).to_csv(self.path_Xu, header=None, index=None)
        return torch.tensor(s).float()

    def Xs_dataSampling(self, Xs, x_var_num, need_write = True):
        '''
        安全区域的采样

        :param need_write: 记录采样得到的数据
        :return: 返回采样的数据
        '''

        lst = []
        for i in range(x_var_num):
            x_data = np.linspace(Xs[i][0], Xs[i][1], self.Xs_sample_len)
            lst.append(x_data)

        s = np.array(np.meshgrid(*lst))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        if need_write:
            result = np.unique(s, axis=0)
            pd.DataFrame(result).to_csv(self.path_Xs, header=None, index=None)
        return torch.tensor(s).float()

    def X_dataSampling(self, X, x_var_num, need_write = True):
        '''
        全区域的采样

        :param need_write: 记录采样得到的数据
        :return: 返回采样的数据
        '''
        lst = []
        for i in range(x_var_num):
            x_data = np.linspace(X[i][0], X[i][1], self.X_sample_len)
            lst.append(x_data)

        s = np.array(np.meshgrid(*lst))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        if need_write:
            result = np.unique(s, axis=0)
            pd.DataFrame(result).to_csv(self.path_X, header=None, index=None)
        return torch.tensor(s).float()

if __name__ == "__main__":
    pass
