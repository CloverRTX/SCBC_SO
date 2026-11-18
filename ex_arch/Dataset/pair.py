import torch
from utils.RV_triang import TriangularDistribution

class Pair:
    def __init__(self, x, config):
        self.x = x
        self.config = config
        self.num = config["sample_config"]["RV_sample_nums"]
        self.x_var_num = config["dynamics"]["x_var_num"]
        self.random_var_num = config["dynamics"]["random_var_num"]
        self.rv_values = None
        # self.k_inductive = config["k_inductive"]
        self.tensorTable_RV = None
        self.distribution = TriangularDistribution()

    def getX(self):
        return self.x

    def getRVs(self):
        if self.rv_values == None:
            self.rv_values = []
            self.rv_values = self.distribution.RV_sample((self.num, self.random_var_num))
        return self.rv_values

    def getTensorTable(self):
        if self.tensorTable_RV == None:
            self.toTensorTable_RV()
        return self.tensorTable_RV

    def toTensorTable_RV(self):
        self.getRVs()
        result = torch.empty((self.num, self.x_var_num + self.random_var_num))
        result[:, :self.x_var_num] = self.x
        result[:, self.x_var_num:] = self.rv_values
        self.tensorTable_RV = result