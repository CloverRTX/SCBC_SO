import csv
import torch.distributions
import copy
import torch
import numpy as np
from Dataset.pair import Pair

class StochasticSystemDataset():

    def __init__(self, data_X0, data_Xu, data_Xs, data_X, config):

        self.config = config

        self.device = config["device"]

        # X0
        self.path_X0 = config["sample_config"]["path_X0"]

        # Xu
        self.path_Xu = config["sample_config"]["path_Xu"]

        # X
        self.path_X = config["sample_config"]["path_X"]

        # Xs
        self.path_Xs = config["sample_config"]["path_Xs"]

        # 源训练集
        self.data_X0 = data_X0
        self.data_Xu = data_Xu
        self.data_Xs = data_Xs
        self.data_X = data_X

        self.X0_batch_size = config["sample_config"]["X0_batch_size"]
        self.Xu_batch_size = config["sample_config"]["Xu_batch_size"]
        self.Xs_batch_size = config["sample_config"]["Xs_batch_size"]
        self.X_batch_size = config["sample_config"]["X_batch_size"]

        self.getLen()

    def reload(self):

        with open(self.path_X0, "r") as fp:
            data = list(csv.reader(fp))
            # 元素数据类型float
            self.data_X0 = torch.tensor(np.array(data).astype(float)).to(torch.float32).to(self.device)

        with open(self.path_Xu, "r") as fp:
            data = list(csv.reader(fp))
            # 元素数据类型float
            self.data_Xu = torch.tensor(np.array(data).astype(float)).to(torch.float32).to(self.device)

        with open(self.path_X, "r") as fp:
            data = list(csv.reader(fp))
            # 元素数据类型float
            self.data_X = torch.tensor(np.array(data).astype(float)).to(torch.float32).to(self.device)

        with open(self.path_Xs, "r") as fp:
            data = list(csv.reader(fp))
            # 元素数据类型float
            self.data_Xs = torch.tensor(np.array(data).astype(float)).to(torch.float32).to(self.device)

        self.getLen()

    def batchDivision(self):

        X0_batch_size = self.X0_batch_size
        Xu_batch_size = self.Xu_batch_size
        Xs_batch_size = self.Xs_batch_size
        X_batch_size = self.X_batch_size


        temp_data_X0 = self.arr_split(self.data_X0, X0_batch_size)
        temp_data_Xu = self.arr_split(self.data_Xu, Xu_batch_size)
        temp_data_Xs = self.arr_split(self.data_Xs, Xs_batch_size)
        temp_data_X = self.arr_split(self.data_X, X_batch_size)

        num_max = max(len(temp_data_X0), len(temp_data_Xu))
        num_max = max(num_max, len(temp_data_X))
        num_max = max(num_max, len(temp_data_Xs))

        # 处理init 分batch
        num_init = len(temp_data_X0)
        for i in range(num_max - num_init):
            temp = copy.deepcopy(temp_data_X0[i])
            temp_data_X0.append(temp)

        # 处理unsafe 分batch
        num_unsafe = len(temp_data_Xu)
        for i in range(num_max - num_unsafe):
            temp = copy.deepcopy(temp_data_Xu[i])
            temp_data_Xu.append(temp)

        # 处理X_safe 分batch
        num_safe = len(temp_data_Xs)
        for i in range(num_max - num_safe):
            temp = copy.deepcopy(temp_data_Xs[i])
            temp_data_Xs.append(temp)

        # 处理X 分batch
        num_X = len(temp_data_X)
        for i in range(num_max - num_X):
            temp = copy.deepcopy(temp_data_X[i])
            temp_data_X.append(temp)


        batchSet = []

        for i in range(num_max):
            part_data_X0 = temp_data_X0[i]
            part_data_Xu = temp_data_Xu[i]
            part_data_X = temp_data_X[i]

            pair_part_data_Xs = []
            for j in range(temp_data_Xs[i].shape[0]):
                pair_part_data_Xs.append(Pair(temp_data_Xs[i][j, :], self.config))
            batchSet.append((part_data_X0, part_data_Xu, pair_part_data_Xs, part_data_X))

        return batchSet

    def fullBatchData(self):
        pairs_data_Xs = []
        temp_data_Xs = copy.deepcopy(self.data_Xs)
        for i in range(temp_data_Xs.shape[0]):
            pairs_data_Xs.append(Pair(temp_data_Xs[i, :], self.config))

        return copy.deepcopy(self.data_X0), copy.deepcopy(self.data_Xu), pairs_data_Xs, \
               copy.deepcopy(self.data_X)


    def arr_split(self, data, line_num):
        total_line = data.shape[0]
        yu = total_line % line_num
        new_data = copy.deepcopy(data[0 : total_line - yu])
        last_data = copy.deepcopy(data[total_line - yu :, :])
        if total_line // line_num > 0:
            lst = list(torch.tensor_split(new_data, total_line // line_num, axis = 0))
            if yu > 0:
                lst.append(last_data)
            return lst
        else:
            return [last_data]

    def getLen(self):
        print(f"The sample number of X0 : {self.data_X0.shape[0]}")
        print(f"The sample number of Xu : {self.data_Xu.shape[0]}")
        print(f"The sample number of Xs : {self.data_Xs.shape[0]}")
        print(f"The sample number of X : {self.data_X.shape[0]}")
