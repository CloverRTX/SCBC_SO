import torch
from BFModels.BarrierFunc import BarrierFuncNetwork
from BFModels.Controller import ColNetwork
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange, tqdm
from utils.Crown_IBP_API import crown, wrapModule
from utils.Crown_IBP_API import HyperRectangle
from Log.LogHandler import info_log

import time

class Learner():

    def __init__(self, dataset, dynamic, config, tolerance_threshold = 1e-5):
        self.SBC_model = None
        self.trainingDataset = dataset
        self.dynamic = dynamic
        self.config = config
        self.device = config["device"]
        self.tolerance_threshold = tolerance_threshold
        self.x_var_num = config["dynamics"]["x_var_num"]
        self.random_var_num = config["dynamics"]["random_var_num"]
        self.RV_sample_nums = config["sample_config"]["RV_sample_nums"]
        self.epsilon = self.config["Bx_model"]["epsilon"]

    def init_NN_Model(self):
        Bx_input_dim = self.config["Bx_model"]["input_dim"]

        Bx_output_dim = self.config["Bx_model"]["output_dim"]

        self.SBC_model = BarrierFuncNetwork(input_dim=Bx_input_dim,
                                               output_dim=Bx_output_dim).to(device=self.device)


    # fullBatch训练
    def train(self):
        lr = 1e-2
        optimizer_Bx = torch.optim.Adam(self.SBC_model.parameters(), lr)

        # 设置指数衰减学习率
        #scheduler_Bx = ExponentialLR(optimizer_Bx, gamma=0.9)

        self.SBC_model.train()

        # 划分batch
        traindata = self.trainingDataset.fullBatchData()

        epoch = 0
        # data = trainingDataset.batchDivision()

        # 遍历 epoch
        while True:
            epoch += 1
            batch_num = 0.

            optimizer_Bx.zero_grad()
            # optimizer_Col.zero_grad()

            # 求各个条件的Loss
            X_loss, X0_loss, Xu_loss, ME_loss = self.step(traindata)

            epoch_loss = X_loss + X0_loss + Xu_loss + ME_loss

            if epoch % 50 == 1:
                print(f"==========[train] epoch.{epoch}==============")
                print(f"当前优化器学习率 = {optimizer_Bx.param_groups[0]['lr']}")
                print(f"X_loss = {X_loss}")
                print(f"X0_loss = {X0_loss}")
                print(f"Xu_loss = {Xu_loss}")
                print(f"ME_loss = {ME_loss}")
                print(f"epoch = {epoch} epoch_Loss = {epoch_loss}")
                print(f"==========[end] epoch : {epoch}==============")

            # 训练结束条件
            if epoch_loss <= self.tolerance_threshold:
                print(f"==========[train] epoch.{epoch}==============")
                print(f"当前优化器学习率 = {optimizer_Bx.param_groups[0]['lr']}")
                print(f"X_loss = {X_loss}")
                print(f"X0_loss = {X0_loss}")
                print(f"Xu_loss = {Xu_loss}")
                print(f"ME_loss = {ME_loss}")
                print(f"epoch = {epoch} epoch_Loss = {epoch_loss}")
                print(f"==========[end] epoch : {epoch}==============")
                break


            epoch_loss.backward()
            optimizer_Bx.step()

            if epoch == 2000:
                optimizer_Bx.param_groups[0]['lr'] = 1e-3
                print(f"修改后学习率: {optimizer_Bx.param_groups[0]['lr']}")
            if epoch == 10000:
                optimizer_Bx.param_groups[0]['lr'] = 1e-4
                print(f"修改后学习率: {optimizer_Bx.param_groups[0]['lr']}")
            #if epoch == 400:
                #optimizer_Bx.param_groups[0]['lr'] = 1e-5
                #print(f"修改后学习率: {optimizer_Bx.param_groups[0]['lr']}")
            #if epoch == 500:
                #optimizer_Bx.param_groups[0]['lr'] = 1e-6
                #print(f"修改后学习率: {optimizer_Bx.param_groups[0]['lr']}")
            # 每个epoch 改变学习率
            #scheduler_Bx.step()


            # print(f"==========[retain] epoch.{epoch}==============")
            # print(f"totalLoss = {Loss}")
            # print(f"total_Xu_loss = {total_Xu_loss}")
            # print(f"total_X_loss = {total_X_loss}")
            # print(f"total_ME_loss = {total_ME_loss}")
            # print(f"gama = {gama}")
            # print(f"P_safe = {1 - gama}")
            #info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, max_gama, max_ME_upper, "retain")
            #info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, max_gama, max_ME_upper, "retain")


    def train_batch(self):
        lr = 0.005
        optimizer_Bx = torch.optim.Adam(self.SBC_model.parameters(), lr)

        # 设置指数衰减学习率
        scheduler_Bx = ExponentialLR(optimizer_Bx, gamma=0.9)

        self.SBC_model.train()

        # 划分batch
        batchSet = self.trainingDataset.batchDivision()

        epoch = 0
        # data = trainingDataset.batchDivision()

        # 遍历 epoch
        while True:
            epoch += 1
            batch_num = 0.

            epoch_Xu_loss = 0.
            epoch_X_loss = 0.
            epoch_ME_loss = 0.
            epoch_X0_loss = 0.

            # 遍历 batch
            for batch_data in batchSet:
                batch_num = batch_num + 1
                optimizer_Bx.zero_grad()
                # optimizer_Col.zero_grad()

                # 求各个条件的Loss
                X_loss, X0_loss, Xu_loss, ME_loss = self.step(batch_data)

                batch_loss = X_loss + X0_loss + Xu_loss + ME_loss
                epoch_X_loss = epoch_X_loss + X_loss
                epoch_X0_loss = epoch_X0_loss + X0_loss
                epoch_Xu_loss = epoch_Xu_loss + Xu_loss
                epoch_ME_loss = epoch_ME_loss + ME_loss

                print(f"==========[train] epoch.{epoch}==============")
                print(f"==========[train] batch.{batch_num}==============")
                print(f"X_loss = {X_loss}")
                print(f"X0_loss = {X0_loss}")
                print(f"Xu_loss = {Xu_loss}")
                print(f"ME_loss = {ME_loss}")
                print(f"epoch = {epoch}; batch = {batch_num} batch_Loss = {batch_loss}")
                print(f"==========[end] epoch : {epoch}; batch = {batch_num}==============")

                batch_loss.backward()
                optimizer_Bx.step()
                # optimizer_Col.step()

            print(f"epoch = {epoch}, epoch_Loss = {epoch_X_loss + epoch_X0_loss + epoch_Xu_loss + epoch_ME_loss}")
            print(f"==========[end] epoch : {epoch}==================")
            # 每个epoch 改变学习率
            scheduler_Bx.step()

            # 训练结束条件
            if epoch_Xu_loss + epoch_X_loss + epoch_ME_loss + epoch_X0_loss <= self.tolerance_threshold:
                break
                # print(f"==========[retain] epoch.{epoch}==============")
                # print(f"totalLoss = {Loss}")
                # print(f"total_Xu_loss = {total_Xu_loss}")
                # print(f"total_X_loss = {total_X_loss}")
                # print(f"total_ME_loss = {total_ME_loss}")
                # print(f"gama = {gama}")
                # print(f"P_safe = {1 - gama}")
                #info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, max_gama, max_ME_upper, "retain")
            #info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, max_gama, max_ME_upper, "retain")


    # fullBatch
    def retrain(self, optimizer_Bx, trainingDataset):
        epoch = 0
        data = trainingDataset.fullBatchData()
        while True:
            epoch += 1
            optimizer_Bx.zero_grad()
            # optimizer_Col.zero_grad()
            Loss, Xu_loss, X_loss, kI_ME_loss, gama, ME_upper = self.step(data, 0.5, retrain=True)
            total_Xu_loss = Xu_loss
            total_X_loss = X_loss
            total_kI_ME_loss = kI_ME_loss

            print(f"==========[retain] epoch.{epoch}==============")
            print(f"totalLoss = {Loss}")
            print(f"total_Xu_loss = {total_Xu_loss}")
            print(f"total_X_loss = {total_X_loss}")
            print(f"total_kI_ME_loss = {total_kI_ME_loss}")
            print(f"gama = {gama}")
            print(f"ME_upper = {ME_upper}")
            print(f"P_safe = {-1}")
            if total_Xu_loss + total_X_loss + total_kI_ME_loss <= self.tolerance_threshold:
                #print(f"==========[retain] epoch.{epoch}==============")
                #print(f"totalLoss = {Loss}")
                #print(f"total_Xu_loss = {total_Xu_loss}")
                #print(f"total_X_loss = {total_X_loss}")
                #print(f"total_ME_loss = {total_ME_loss}")
                #print(f"gama = {gama}")
                #print(f"P_safe = {1 - gama}")
                info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, gama, ME_upper, "retain")
                break

            Loss.backward()
            optimizer_Bx.step()
            # optimizer_Col.step()
            info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, gama, ME_upper, "retain")

    def retrain_batch(self, optimizer_Bx, trainingDataset):
        epoch = 0
        data = trainingDataset.batchDivision()
        kappa = 0.
        while True:
            epoch += 1
            total_Xu_loss = 0.
            total_X_loss = 0.
            total_kI_ME_loss = 0.
            total_X0_loss = 0.
            max_gama = -100.
            max_ME_upper = -100
            batch_num = 0.
            for batch_data in data:
                batch_num = batch_num + 1
                optimizer_Bx.zero_grad()
                # optimizer_Col.zero_grad()
                Loss, Xu_loss, X_loss, kI_ME_loss, gama, ME_upper = self.step(batch_data)

                sub_loss1 = (gama - 1.).clamp(min=0).sum()
                sub_loss2 = (-gama - 0).clamp(min=0).sum()
                # 要求初始区域的界在[0,1]之间
                loss_vio = sub_loss1 + sub_loss2
                Loss += loss_vio


                total_Xu_loss = total_Xu_loss + Xu_loss
                total_X_loss = total_X_loss + X_loss
                total_kI_ME_loss = total_kI_ME_loss + kI_ME_loss
                total_X0_loss = total_X0_loss + loss_vio
                max_gama = max(max_gama, gama)
                max_ME_upper = max(max_ME_upper, ME_upper)

                print(f"==========[retain] epoch.{epoch}==============")
                print(f"==========[retain] batch.{batch_num}==============")
                print(f"Xu_loss = {Xu_loss}")
                print(f"X_loss = {X_loss}")
                print(f"kI_ME_loss = {kI_ME_loss}")
                print(f"gama = {max_gama}")
                print(f"gama_loss = {loss_vio}")
                print(f"ME_upper = {max_ME_upper}")


                Loss.backward()
                optimizer_Bx.step()
                # optimizer_Col.step()

            print(f"==========[retain] epoch.{epoch}==============")
            print(f"total_Xu_loss = {total_Xu_loss}")
            print(f"total_X_loss = {total_X_loss}")
            print(f"total_kI_ME_loss = {total_kI_ME_loss}")
            print(f"gama = {max_gama}")
            print(f"gama_loss = {total_X0_loss}")
            print(f"ME_upper = {max_ME_upper}")
            print(f"total_loss = {total_Xu_loss + total_X_loss + total_kI_ME_loss + total_X0_loss}")
            #print(f"P_safe = {1 - max_gama}")
            if total_Xu_loss + total_X_loss + total_kI_ME_loss + total_X0_loss <= self.tolerance_threshold:
                # print(f"==========[retain] epoch.{epoch}==============")
                # print(f"totalLoss = {Loss}")
                # print(f"total_Xu_loss = {total_Xu_loss}")
                # print(f"total_X_loss = {total_X_loss}")
                # print(f"total_ME_loss = {total_ME_loss}")
                # print(f"gama = {gama}")
                # print(f"P_safe = {1 - gama}")
                info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, max_gama, max_ME_upper, "retain")
                break

            info_log(epoch, total_Xu_loss, total_X_loss, total_kI_ME_loss, max_gama, max_ME_upper, "retain")

    # 反例训练
    def counterexample_train(self):
        lr = 0.001
        self.trainingDataset.reload()
        optimizer_Bx = torch.optim.Adam(self.bounded_CBC_model.parameters(), lr)
        optimizer_Col = torch.optim.Adam(self.bounded_Col_model.parameters(), lr)
        self.retrain(optimizer_Bx, optimizer_Col, self.trainingDataset)

    # 首次训练
    def start_train(self):
        self.init_NN_Model()
        self.train()

    def start_CEX_train(self):
        self.trainingDataset.reload()
        self.train()

    # step 每个batch所对应的训练过程
    def step(self, batch_data):

        X0_batch_data = batch_data[0].to(self.device)
        Xu_batch_data = batch_data[1].to(self.device)
        ME_batch_data = batch_data[2]
        X_batch_data = batch_data[3].to(self.device)

        # start_time = time.time()
        # end_time = time.time()
        # print("程序1运行时间为：", end_time - start_time, "秒")


        X_loss = self.X_Loss_Func(X_batch_data)

        X0_loss = self.X0_Loss_Func(X0_batch_data)

        Xu_loss = self.Xu_Loss_Func(Xu_batch_data)

        ME_loss = self.ME_Loss_Func_traditional(ME_batch_data)

        return X_loss, X0_loss, Xu_loss, ME_loss

    # 全区域Loss函数
    def X_Loss_Func(self, data):
        val = self.SBC_model(data)
        loss = (-val).clamp(min=0).sum()
        return loss

    # 初始区域Loss函数
    def X0_Loss_Func(self, data):
        val = self.SBC_model(data)
        loss = (val - self.epsilon).clamp(min=0).sum()
        return loss

    # 不安全区域Loss函数
    def Xu_Loss_Func(self, data):
        val = self.SBC_model(data)
        loss = (1. - val).clamp(min=0).sum()
        return loss

    # ME条件Loss函数
    def ME_Loss_Func_traditional(self, data):
        x_point = torch.empty((len(data), self.x_var_num)).to(self.device)
        # x_point[0, :] = data[0].getX().to(self.device)

        x_rv_table = torch.empty((len(data) * self.RV_sample_nums, self.x_var_num + self.random_var_num)).to(torch.float32).to(self.device)
        # x_rv_table_temp = data[0].getTensorTable_kI().to(self.device)

        for i in range(len(data)):
            pair_data = data[i]
            x_rv_table[i * self.RV_sample_nums: (i + 1) * self.RV_sample_nums,:] = pair_data.getTensorTable().to(self.device)
            x_point[i, :] = pair_data.getX().to(self.device)

        # u = self.bounded_Col_model(x_rv_table[:, :self.x_var_num])
        x_next = self.dynamic.forward(x_rv_table[:, :self.x_var_num], x_rv_table[:, self.x_var_num:])

        B_x = self.SBC_model(x_point)
        B_x_next = self.SBC_model(x_next)
        lst = torch.split(B_x_next, self.RV_sample_nums, dim=0)
        B_x_next_mean = torch.empty(len(lst), 1).to(self.device)

        index = 0
        for ele in lst:
            B_x_next_mean[index, :] = ele.mean()
            index = index + 1

        return (B_x_next_mean - B_x).clamp(min=0).sum()

    # 先弃用

    # k归纳数学期望条件Loss函数
    def kI_ME_Loss_Func_traditional(self, data):

        x_point = torch.empty((len(data), self.x_var_num)).to(self.device)
        # x_point[0, :] = data[0].getX().to(self.device)

        x_rv_table = torch.empty((len(data) * self.RV_sample_nums, self.x_var_num + self.random_var_num * self.k_inductive)).to(self.device)
        # x_rv_table_temp = data[0].getTensorTable_kI().to(self.device)

        for i in range(len(data)):
            pair_data = data[i]
            x_rv_table[i * self.RV_sample_nums: (i + 1) * self.RV_sample_nums,:] = pair_data.getTensorTable_kI().to(self.device)
            x_point[i, :] = pair_data.getX().to(self.device)
        # x_rv_table = x_rv_table.to(torch.float32).to(self.device)
        # x_point = x_point.to(torch.float32).to(self.device)

        x_pre = x_rv_table[:, :self.x_var_num]
        for i in range(self.k_inductive):
            # u = self.bounded_Col_model(x_pre)
            v = x_rv_table[:, self.x_var_num + self.random_var_num * i : self.x_var_num + self.random_var_num * (i + 1)]
            x_next = self.dynamic.forward(x_pre, v)
            x_pre = x_next


        kI_x_point = x_pre
        B_x = self.bounded_kSBC_model(x_point)
        kI_B_x = self.bounded_kSBC_model(kI_x_point)
        lst = torch.split(kI_B_x, self.RV_sample_nums, dim=0)
        kI_B_x_mean = torch.empty(len(lst), 1).to(torch.float32).to(self.device)

        index = 0
        for ele in lst:
            kI_B_x_mean[index, :] = ele.mean()
            index = index + 1

        return (kI_B_x_mean - B_x).clamp(min=0).sum()

