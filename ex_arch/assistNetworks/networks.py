import numpy as np
import torch
from cellPartitioning.partitioningFuncs import get_RV_partitions
from utils.Crown_IBP_API import crown, HyperRectangle, crown_new
from utils.RV_triang import TriangularDistribution
from utils.Crown_IBP_API import Affine

# !need fix! : 针对不同的动力系统是需要修改的方法
class MEConditionModule:
    def __init__(self, config, bounded_CBC_model, bounded_Col_model, dynamic):

        self.config = config

        self.device = config["device"]
        self.k_induction = config["k_induction"]
        self.x_var_num = config["dynamics"]["x_var_num"]
        self.col_var_num = config["dynamics"]["col_var_num"]
        self.random_var_num = config["dynamics"]["random_var_num"]

        self.bounded_CBC_model = bounded_CBC_model
        self.bounded_Col_model = bounded_Col_model
        self.dynamic = dynamic
        self.distribution = TriangularDistribution()

        self.col_var_num = config["dynamics"]["col_var_num"]
        self.RV_num_slices = config["partitioning"]["RV_num_slices"]


    def getLinearBoundOfMECon(self, x_partitions, bounded_lower=True, bounded_upper=True):

        RV_partitions = get_RV_partitions(self.config)

        fx_lb, fx_ub, x_lb, x_ub, u_lb, u_ub, v_lb, v_ub = self.dynamic.dynamics_IntervalPropagation_for_verify(
            x_partitions=x_partitions,
            v_partitions=RV_partitions,
            bounded_Col_model=self.bounded_Col_model
        )

        result = self.approximate_process_via_partitions(
            fx_lb=fx_lb,
            fx_ub=fx_ub,
            bounded_CBC_model=self.bounded_CBC_model,
            bounded_lower=bounded_lower,
            bounded_upper=bounded_upper
        )

        result_dict_Ebx = self.ME_processing_process(
            lb_sets=result["lb_sets"],
            ub_sets=result["ub_sets"],
            u_lower=u_lb,
            u_upper=u_ub,
            v_lower=v_lb,
            v_upper=v_ub,
            bounded_lower=bounded_lower,
            bounded_upper=bounded_upper
        )

        result_dict_Bx = crown_new(self.bounded_CBC_model, x_partitions.center, x_partitions.diff,
                                   bound_lower=bounded_lower, bound_upper=bounded_upper, needed_A=True)

        result_dict = {}

        if bounded_lower:
            Ebx_A_lb = result_dict_Ebx["lb_A"]
            Ebx_bias_lb = result_dict_Ebx["lb_bias"]

            Bx_A_ub = result_dict_Bx["ub_A"]
            Bx_bias_ub = result_dict_Bx["ub_bias"]
            Affine_lower = Affine(Ebx_A_lb - Bx_A_ub, Ebx_bias_lb - Bx_bias_ub, x_partitions.lower, x_partitions.upper)
            result_dict["Affine_lower"] = Affine_lower

        if bounded_upper:
            Ebx_A_ub = result_dict_Ebx["ub_A"]
            Ebx_bias_ub = result_dict_Ebx["ub_bias"]

            Bx_A_lb = result_dict_Bx["lb_A"]
            Bx_bias_lb = result_dict_Bx["lb_bias"]
            Affine_upper = Affine(Ebx_A_ub - Bx_A_lb, Ebx_bias_ub - Bx_bias_lb, x_partitions.lower, x_partitions.upper)
            result_dict["Affine_upper"] = Affine_upper

        return result_dict

    def approximate_process_via_partitions(self, fx_lb, fx_ub, bounded_CBC_model, bounded_lower=True,
                                           bounded_upper=True):
        """
            近似过程
            已有f(x,u,v)的上下界
            可以获取
            A * y + b <= B(y) <= A' * y + b'
            转化为 B(f(x, u , v)) <= Aq * x + Au * u + Aw0 * v0 + Aw1 * v1 + b

            转化由transform()完成
            返回 Aq Au Aw b
        :param fx_lb:
        :param fx_ub:
        :param bounded_CBC_model:
        :param bounded_lower:
        :param bounded_upper:
        :return: result_dict
        """

        def transform(A_y, b_y):
            Aq = torch.empty_like(A_y)
            # bias = torch.tensor(b_y).to(torch.float32).to(self.device)
            bias = b_y.clone().detach().to(torch.float32).to(self.device)
            bias = torch.unsqueeze(bias, -1)
            W1 = A_y[:, 0, 0]
            W2 = A_y[:, 0, 1]

            # Au = torch.tensor(W1)
            # Au = torch.empty_like(A_y)
            Au = 0.45 * W1 + 0.5 * W2
            # Au[:, 0, 0] = W1.clone().detach().to(torch.float32).to(self.device)
            # Au[:, 0, 1] = W2.clone().detach().to(torch.float32).to(self.device)
            # Au = W2.clone().detach().to(torch.float32).to(self.device)
            Au = torch.unsqueeze(Au, -1)
            Au = torch.unsqueeze(Au, -1)

            # Aw = W2.clone().detach().to(torch.float32).to(self.device)
            # Aw = torch.unsqueeze(Aw, -1)
            # Aw = torch.unsqueeze(Aw, -1)
            Aw0 = 0.1 * W1
            Aw0 = torch.unsqueeze(Aw0, -1)
            Aw0 = torch.unsqueeze(Aw0, -1)

            Aw1 = 0.005 * W2
            Aw1 = torch.unsqueeze(Aw1, -1)
            Aw1 = torch.unsqueeze(Aw1, -1)

            Aq[:, 0, 0] = W1
            Aq[:, 0, 1] = 0.045 * W1 + 0.9 * W2

            return Aq, Au, Aw0, Aw1, bias

        rect = HyperRectangle(fx_lb, fx_ub)

        result_dict = crown_new(bounded_CBC_model, rect.center, rect.diff,
                                bound_lower=bounded_lower, bound_upper=bounded_upper, needed_A=True)
        lb_A = result_dict["lb_A"]
        lb_bias = result_dict["lb_bias"]
        ub_A = result_dict["ub_A"]
        ub_bias = result_dict["ub_bias"]

        result_dict = {}
        if bounded_lower:
            result_dict["lb_sets"] = transform(lb_A, lb_bias)
        if bounded_upper:
            result_dict["ub_sets"] = transform(ub_A, ub_bias)

        return result_dict

    def ME_processing_process(self, lb_sets, ub_sets,
                              u_lower, u_upper, v_lower, v_upper, bounded_lower=True, bounded_upper=True):
        """
            期望处理过程

            前提: 已有 B(f(x,u,v)) <= Aq * x + Au * u + Aw0 * v0 + Aw1 * v1 + b
            先处理 Au * u + b
            求 E(B(f(x,u,v))) <= A * x + b

        :param lb_sets:
        :param ub_sets:
        :param v_lower:
        :param v_upper:
        :param bounded_lower:
        :param bounded_upper:
        :return:
        """
        A_u_lower_lb = None
        A_u_upper_ub = None
        if bounded_lower:
            # 与下界相关
            A_u_lower_lb = Affine(lb_sets[1], torch.squeeze(lb_sets[-1], -1), u_lower, u_upper).lower_Ax_b()
        if bounded_upper:
            # 与上界相关
            A_u_upper_ub = Affine(ub_sets[1], torch.squeeze(ub_sets[-1], -1), u_lower, u_upper).upper_Ax_b()
        bias_lb = A_u_lower_lb
        bias_ub = A_u_upper_ub

        # 处理随机
        v_lower = v_lower.cpu().numpy()
        v_upper = v_upper.cpu().numpy()

        rv_integrate_pdf_lst = []
        rv_integrate_vMulpdf_lst = []


        for i in range(self.random_var_num):
            v_sub_lower = v_lower[:, i]
            v_sub_upper = v_upper[:, i]

            rv_integrate_pdf = torch.tensor(self.distribution.cdf(v_sub_lower, v_sub_upper)).to(torch.float32).to(self.device)
            rv_integrate_vMulpdf = torch.tensor(self.distribution.v_mul_px_plus(v_sub_lower, v_sub_upper)).to(torch.float32).to(
                self.device)
            rv_integrate_pdf_lst.append(torch.unsqueeze(rv_integrate_pdf, -1))
            rv_integrate_vMulpdf_lst.append(torch.unsqueeze(rv_integrate_vMulpdf, -1))


            # rv_integrate_pdf = torch.tensor(self.distribution.integrate_pdf(v_sub_lower, v_sub_upper)).to(torch.float32).to(self.device)
            # rv_integrate_vMulpdf = torch.tensor(self.distribution.integrate_vMulpdf(v_sub_lower, v_sub_upper)).to(torch.float32).to(self.device)
            # rv_integrate_pdf_lst.append(torch.unsqueeze(rv_integrate_pdf, -1))
            # rv_integrate_vMulpdf_lst.append(torch.unsqueeze(rv_integrate_vMulpdf, -1))

        result_dict = {}
        if bounded_lower:
            bias_lb = torch.unsqueeze(bias_lb, -1)
            Aq_lb = lb_sets[0]
            Aw0_lb = lb_sets[2]
            Aw1_lb = lb_sets[3]

            Aq_lb[:, 0, :] = Aq_lb[:, 0, :] * rv_integrate_pdf_lst[0] * rv_integrate_pdf_lst[1]
            bias_lb[:, 0, :] = bias_lb[:, 0, :] * rv_integrate_pdf_lst[0] * rv_integrate_pdf_lst[1]
            Aw0_lb[:, 0, :] = Aw0_lb[:, 0, :] * rv_integrate_vMulpdf_lst[0] * rv_integrate_pdf_lst[1]
            Aw1_lb[:, 0, :] = Aw1_lb[:, 0, :] * rv_integrate_vMulpdf_lst[1] * rv_integrate_pdf_lst[0]
            new_bias_lb = bias_lb + Aw0_lb + Aw1_lb

            # u_center = (u_upper + u_lower) / 2.0
            # u_eps = (u_upper - u_lower) / 2.0

            Aq_lb = torch.split(Aq_lb, self.RV_num_slices ** self.random_var_num, dim=0)
            new_bias_lb = torch.split(new_bias_lb, self.RV_num_slices ** self.random_var_num, dim=0)

            A_LB = None
            bias_LB = None
            for i in range(len(Aq_lb)):
                if i == 0:
                    A_LB = Aq_lb[0].sum(dim=0)
                    bias_LB = new_bias_lb[0].sum(dim=0)
                else:
                    A_LB = torch.cat((A_LB, Aq_lb[i].sum(dim=0)), dim=0)
                    bias_LB = torch.cat((bias_LB, new_bias_lb[i].sum(dim=0)), dim=0)

            result_dict["lb_A"] = torch.unsqueeze(A_LB, 1)
            result_dict["lb_bias"] = bias_LB

        if bounded_upper:
            bias_ub = torch.unsqueeze(bias_ub, -1)
            Aq_ub = ub_sets[0]
            Aw0_ub = ub_sets[2]
            Aw1_ub = ub_sets[3]

            Aq_ub[:, 0, :] = Aq_ub[:, 0, :] * rv_integrate_pdf_lst[0] * rv_integrate_pdf_lst[1]
            bias_ub[:, 0, :] = bias_ub[:, 0, :] * rv_integrate_pdf_lst[0] * rv_integrate_pdf_lst[1]
            Aw0_ub[:, 0, :] = Aw0_ub[:, 0, :] * rv_integrate_vMulpdf_lst[0] * rv_integrate_pdf_lst[1]
            Aw1_ub[:, 0, :] = Aw1_ub[:, 0, :] * rv_integrate_vMulpdf_lst[1] * rv_integrate_pdf_lst[0]
            new_bias_ub = bias_ub + Aw0_ub + Aw1_ub

            # u_center = (u_upper + u_lower) / 2.0
            # u_eps = (u_upper - u_lower) / 2.0

            Aq_ub = torch.split(Aq_ub, self.RV_num_slices ** self.random_var_num, dim=0)
            new_bias_ub = torch.split(new_bias_ub, self.RV_num_slices ** self.random_var_num, dim=0)

            A_UB = None
            bias_UB = None
            for i in range(len(Aq_ub)):
                if i == 0:
                    A_UB = Aq_ub[0].sum(dim=0)
                    bias_UB = new_bias_ub[0].sum(dim=0)
                else:
                    A_UB = torch.cat((A_UB, Aq_ub[i].sum(dim=0)), dim=0)
                    bias_UB = torch.cat((bias_UB, new_bias_ub[i].sum(dim=0)), dim=0)

            result_dict["ub_A"] = torch.unsqueeze(A_UB, 1)
            result_dict["ub_bias"] = bias_UB

        return result_dict


    #================================== k_induction ==================================

    def getLinearBoundOf_kIMECon(self, x_partitions, bounded_lower=True, bounded_upper=True):
        RV_partitions = get_RV_partitions(self.config)

        kI_fx_lb, kI_fx_ub, data_lb, data_ub = self.dynamic.dynamics_kI_IntervalPropagation_for_verify(
            x_partitions=x_partitions,
            v_partitions=RV_partitions,
            bounded_Col_model=self.bounded_Col_model
        )

        result = self.kI_approximate_process_via_partitions(
            kI_fx_lb=kI_fx_lb,
            kI_fx_ub=kI_fx_ub,
            bounded_CBC_model=self.bounded_CBC_model,
            bounded_lower=bounded_lower,
            bounded_upper=bounded_upper
        )

        result_dict_Ebx = self.kI_ME_processing_process(
            lb_sets=result["lb_sets"],
            ub_sets=result["ub_sets"],
            data_lb=data_lb,
            data_ub=data_ub,
            bounded_lower=bounded_lower,
            bounded_upper=bounded_upper
        )

        result_dict_Bx = crown(self.bounded_CBC_model, x_partitions.center, x_partitions.diff,
                               bound_lower=bounded_lower, bound_upper=bounded_upper, needed_A=True)
        result_dict = {}

        if bounded_lower:
            Ebx_A_lb = result_dict_Ebx["lb_A"]
            Ebx_bias_lb = result_dict_Ebx["lb_bias"]

            Bx_A_ub = result_dict_Bx["ub_A"]
            Bx_bias_ub = result_dict_Bx["ub_bias"]
            Affine_lower = Affine(Ebx_A_lb - Bx_A_ub, Ebx_bias_lb - Bx_bias_ub, x_partitions.lower, x_partitions.upper)
            result_dict["Affine_lower"] = Affine_lower

        if bounded_upper:
            Ebx_A_ub = result_dict_Ebx["ub_A"]
            Ebx_bias_ub = result_dict_Ebx["ub_bias"]

            Bx_A_lb = result_dict_Bx["lb_A"]
            Bx_bias_lb = result_dict_Bx["lb_bias"]
            Affine_upper = Affine(Ebx_A_ub - Bx_A_lb, Ebx_bias_ub - Bx_bias_lb, x_partitions.lower, x_partitions.upper)
            result_dict["Affine_upper"] = Affine_upper

        return result_dict

    def kI_approximate_process_via_partitions(self, kI_fx_lb, kI_fx_ub, bounded_CBC_model, bounded_lower=True, bounded_upper=True):
        """
            近似过程
            已有f^(k)(x,u,v)的上下界
            可以获取
            A * y + b <= B(y) <= A' * y + b'
            由于k归纳运算存在，所以很难直接转换为 Aq * x + Au * u + Aw * v + b
            但是还是得转换
            Aq * x + Au0 * u0 + Au1 * u1 + Aw0 * v0 + Aw1 * v1 + Aw2 * v2 + Aw3 * v3 + b
            转换由transform()完成
            返回 A' b'
        :param kI_fx_lb:
        :param kI_fx_ub:
        :param bounded_CBC_model:
        :param bounded_lower:
        :param bounded_upper:
        :return: result_dict
        """

        def transform(A_y, b_y):
            bias = b_y.clone().detach().to(torch.float32).to(self.device)
            bias = torch.unsqueeze(bias, -1)

            W1 = A_y[:, 0, 0]
            W2 = A_y[:, 0, 1]

            Aq = torch.empty_like(A_y)
            Aq[:, 0, 0] = W1
            Aq[:, 0, 1] = 0.0855 * W1 + 0.81 * W2

            # Au0 = W1.clone().detach().to(torch.float32).to(self.device)
            Au0 = 0.4725 * W1 + 0.45 * W2
            Au0 = torch.unsqueeze(Au0, -1)
            Au0 = torch.unsqueeze(Au0, -1)

            # Au1 = W1.clone().detach().to(torch.float32).to(self.device)
            Au1 = 0.45 * W1 + 0.5 * W2
            Au1 = torch.unsqueeze(Au1, -1)
            Au1 = torch.unsqueeze(Au1, -1)

            Aw0 = 0.1 * W1
            Aw0 = torch.unsqueeze(Aw0, -1)
            Aw0 = torch.unsqueeze(Aw0, -1)

            Aw1 = 0.000225 * W1 + 0.0045 * W2
            Aw1 = torch.unsqueeze(Aw1, -1)
            Aw1 = torch.unsqueeze(Aw1, -1)

            Aw2 = 0.1 * W1
            Aw2 = torch.unsqueeze(Aw2, -1)
            Aw2 = torch.unsqueeze(Aw2, -1)

            Aw3 = 0.005 * W2
            Aw3 = torch.unsqueeze(Aw3, -1)
            Aw3 = torch.unsqueeze(Aw3, -1)

            return Aq, Au0, Au1, Aw0, Aw1, Aw2, Aw3, bias

        rect = HyperRectangle(kI_fx_lb, kI_fx_ub)
        # print((kI_fx_ub - kI_fx_lb).min())
        # print(kI_fx_lb.shape[0])
        result_dict = crown_new(bounded_CBC_model, rect.center, rect.diff,
                            bound_lower=bounded_lower, bound_upper=bounded_upper, needed_A=True)
        lb_A = result_dict["lb_A"]
        lb_bias = result_dict["lb_bias"]
        ub_A = result_dict["ub_A"]
        ub_bias = result_dict["ub_bias"]

        result_dict = {}
        if bounded_lower:
            result_dict["lb_sets"] = transform(lb_A, lb_bias)
        if bounded_upper:
            result_dict["ub_sets"] = transform(ub_A, ub_bias)

        return result_dict

    def kI_ME_processing_process(self, lb_sets, ub_sets, data_lb, data_ub, bounded_lower = True, bounded_upper = True):
        """
            期望处理过程
            前提: 已有 B(f^(k)(x,u,v)) <= Aq * x + Au0 * u0 + Au1 * u1 + Aw0 * v0 + Aw1 * v1 + Aw2 * v2 + Aw3 * v3 + b
            先将 Au0 * u0 + Au1 * u1 + b 的上下界求出
            再处理随机变量
            求 E(B(f(x,u,v))) <= A * x + b

        :param lb_sets:
        :param ub_sets:
        :param data_lb:
        :param data_ub:
        :param bounded_lower:
        :param bounded_upper:
        :return:
        """

        # 先处理与u相关的线性乘积项
        A_u_lower_lb = None
        A_u_upper_ub = None
        for i in range(self.k_induction):
            u_xi_lb = data_lb[:, self.x_var_num + self.col_var_num * i : self.x_var_num + self.col_var_num * (i + 1)]
            u_xi_ub = data_ub[:, self.x_var_num + self.col_var_num * i : self.x_var_num + self.col_var_num * (i + 1)]

            if bounded_lower:
                # 与下界相关
                if i == 0:
                    A_u_lower_lb = Affine(lb_sets[i + 1], torch.squeeze(torch.zeros_like(lb_sets[-1]), -1), u_xi_lb, u_xi_ub).lower_Ax_b()
                else:
                    A_u_lower_lb = A_u_lower_lb + Affine(lb_sets[i + 1], torch.squeeze(torch.zeros_like(lb_sets[-1]), -1), u_xi_lb, u_xi_ub).lower_Ax_b()
            if bounded_upper:
                # 与上界相关
                if i == 0:
                    A_u_upper_ub = Affine(ub_sets[i + 1], torch.squeeze(torch.zeros_like(ub_sets[-1]), -1), u_xi_lb, u_xi_ub).upper_Ax_b()
                else:
                    A_u_upper_ub = A_u_upper_ub + Affine(ub_sets[i + 1], torch.squeeze(torch.zeros_like(ub_sets[-1]), -1), u_xi_lb, u_xi_ub).upper_Ax_b()

        if A_u_lower_lb != None:
            A_u_lower_lb = A_u_lower_lb + torch.squeeze(lb_sets[-1], -1)
        if A_u_upper_ub != None:
            A_u_upper_ub = A_u_upper_ub + torch.squeeze(ub_sets[-1], -1)

        bias_lb = A_u_lower_lb
        bias_ub = A_u_upper_ub

        # 处理随机
        rv_pdf_mulpdf_lst = []
        for i in range(self.k_induction * self.random_var_num):
            v_i_lb = data_lb[:, self.x_var_num + self.col_var_num * self.k_induction + i :
                                self.x_var_num + self.col_var_num * self.k_induction + (i + 1)]
            v_i_ub = data_ub[:, self.x_var_num + self.col_var_num * self.k_induction + i :
                                self.x_var_num + self.col_var_num * self.k_induction + (i + 1)]

            v_lower = v_i_lb.cpu().numpy()
            v_upper = v_i_ub.cpu().numpy()
            v_lower = np.squeeze(v_lower, -1)
            v_upper = np.squeeze(v_upper, -1)

            rv_integrate_pdf = torch.tensor(self.distribution.cdf(v_lower, v_upper)).to(torch.float32).to(self.device)
            rv_integrate_vMulpdf = torch.tensor(self.distribution.v_mul_px_plus(v_lower, v_upper)).to(torch.float32).to(self.device)
            rv_pdf_mulpdf_lst.append((torch.unsqueeze(rv_integrate_pdf, -1),
                                      torch.unsqueeze(rv_integrate_vMulpdf, -1)))

        '''
        
        v_upper = v_upper.cpu().numpy()
        rv_integrate_pdf_mul = torch.ones((v_lower.shape[0] ,1)).to(torch.float32).to(self.device)
        for i in range(self.k_induction):
            rv_integrate_pdf = torch.tensor(self.rv.integrate_pdf(v_lower[:, i], v_upper[:, i])).to(torch.float32).to(self.device)
            rv_integrate_pdf = torch.unsqueeze(rv_integrate_pdf, -1)
            rv_integrate_pdf_mul = rv_integrate_pdf_mul * rv_integrate_pdf
            # rv_integrate_vMulpdf = torch.tensor(self.rv.integrate_vMulpdf(v_lower, v_upper)).to(torch.float32).to(self.device)
        '''
        result_dict = {}

        if bounded_lower:
            Aq_lb = lb_sets[0]
            Aw0_lb = lb_sets[3]
            Aw1_lb = lb_sets[4]
            Aw2_lb = lb_sets[5]
            Aw3_lb = lb_sets[6]


            bias_lb = torch.unsqueeze(bias_lb, -1)
            Aq_lb[:, 0, :] = Aq_lb[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]
            bias_lb[:, 0, :] = bias_lb[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]

            Aw0_lb[:, 0, :] = Aw0_lb[:, 0, :] * rv_pdf_mulpdf_lst[0][1] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]
            Aw1_lb[:, 0, :] = Aw1_lb[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][1] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]
            Aw2_lb[:, 0, :] = Aw2_lb[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][1] * rv_pdf_mulpdf_lst[3][0]
            Aw3_lb[:, 0, :] = Aw3_lb[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][1]


            new_bias_lb = bias_lb + Aw0_lb + Aw1_lb + Aw2_lb + Aw3_lb

            Aq_lb = torch.split(Aq_lb, self.RV_num_slices ** (self.k_induction * self.random_var_num), dim=0)
            new_bias_lb = torch.split(new_bias_lb, self.RV_num_slices ** (self.k_induction * self.random_var_num), dim=0)

            A_LB = None
            bias_LB = None
            for i in range(len(Aq_lb)):
                if i == 0:
                    A_LB = Aq_lb[0].sum(dim=0)
                    bias_LB = new_bias_lb[0].sum(dim=0)
                else:
                    A_LB = torch.cat((A_LB, Aq_lb[i].sum(dim=0)), dim=0)
                    bias_LB = torch.cat((bias_LB, new_bias_lb[i].sum(dim=0)), dim=0)

            result_dict["lb_A"] = torch.unsqueeze(A_LB, 1)
            result_dict["lb_bias"] = bias_LB

        if bounded_upper:
            Aq_ub = ub_sets[0]
            Aw0_ub = ub_sets[3]
            Aw1_ub = ub_sets[4]
            Aw2_ub = ub_sets[4]
            Aw3_ub = ub_sets[4]


            bias_ub = torch.unsqueeze(bias_ub, -1)
            Aq_ub[:, 0, :] = Aq_ub[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]
            bias_ub[:, 0, :] = bias_ub[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]

            Aw0_ub[:, 0, :] = Aw0_ub[:, 0, :] * rv_pdf_mulpdf_lst[0][1] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]
            Aw1_ub[:, 0, :] = Aw1_ub[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][1] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][0]
            Aw2_ub[:, 0, :] = Aw2_ub[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][1] * rv_pdf_mulpdf_lst[3][0]
            Aw3_ub[:, 0, :] = Aw3_ub[:, 0, :] * rv_pdf_mulpdf_lst[0][0] * rv_pdf_mulpdf_lst[1][0] * rv_pdf_mulpdf_lst[2][0] * rv_pdf_mulpdf_lst[3][1]

            new_bias_ub = bias_ub + Aw0_ub + Aw1_ub + Aw2_ub + Aw3_ub

            Aq_ub = torch.split(Aq_ub, self.RV_num_slices ** (self.k_induction * self.random_var_num), dim=0)
            new_bias_ub = torch.split(new_bias_ub, self.RV_num_slices ** (self.k_induction * self.random_var_num), dim=0)

            A_UB = None
            bias_UB = None
            for i in range(len(Aq_ub)):
                if i == 0:
                    A_UB = Aq_ub[0].sum(dim=0)
                    bias_UB = new_bias_ub[0].sum(dim=0)
                else:
                    A_UB = torch.cat((A_UB, Aq_ub[i].sum(dim=0)), dim=0)
                    bias_UB = torch.cat((bias_UB, new_bias_ub[i].sum(dim=0)), dim=0)

            result_dict["ub_A"] = torch.unsqueeze(A_UB, 1)
            result_dict["ub_bias"] = bias_UB

        return result_dict