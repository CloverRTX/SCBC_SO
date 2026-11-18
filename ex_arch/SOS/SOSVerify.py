from SumOfSquares import SOSProblem
from utils.polynomial import Polynomial
import sympy as sp


class SOSVerifier():
    def __init__(self, P_x, config, dynamic):

        # 目标多项式 P(x); (多项式形式的SBC, class Polynomial)
        self.P_x = P_x

        # P(x)中的x的sympy表示形式
        self.x_sym = P_x.var_sym

        # P(x)中x的维度
        self.x_dim = P_x.var_dim

        # 多项式中定义系数参数的index
        self.para_index = P_x.para_index

        # 要验证的域
        # 初始域，不安全域，全域 （笛卡尔积形式）
        self.X0 = config["verify"]["X0"]
        self.Xu = config["verify"]["Xu"]
        self.X = config["verify"]["X"]

        self.epsilon = config["verify"]["epsilon"]
        self.fx_expr = dynamic.fx_expr

        # 扰动变量的维度
        self.random_var_num = dynamic.random_var_num
        # 扰动变量sympy形式
        self.random_var = dynamic.fxHelper.random_var

        # 验证过程中SOS多项式的次数 参考参数
        self.deg = config["verify"]["deg"]

        self.E_P_f_x_miner_P_x = None


    def SolveNonNeg(self, deg):
        '''
            con1条件验证, X中 B >= 0

        :param deg: SOS多项式次数
        :return:
        '''

        prob_nonneg = SOSProblem()
        B_I = self.P_x.RealExpr

        for i in range(self.x_dim):
            # Pi, parameters, terms = self.polynomial(deg)
            newPoly = Polynomial(self.x_dim, deg, self.para_index)

            # 更新多项式系数参数的index
            self.para_index = newPoly.para_index

            Pi, parameters, terms = newPoly.poly, newPoly.parameters, newPoly.terms
            prob_nonneg.add_sos_constraint(Pi, self.x_sym)
            B_I = B_I + Pi * (self.x_sym[i] - self.X[i][0]) * (self.x_sym[i] - self.X[i][1])
        B_I = sp.expand(B_I)
        prob_nonneg.add_sos_constraint(B_I, self.x_sym)
        try:
            prob_nonneg.solve(solver='mosek')
            return True
        except:
            return False

    def SolveInit(self, deg):
        '''
            con2条件验证, X0中 B <= epsilon

        :param deg: SOS多项式次数
        :return:
        '''

        prob_init = SOSProblem()
        B_I = -self.P_x.RealExpr + self.epsilon

        for i in range(self.x_dim):
            # Pi, parameters, terms = self.polynomial(deg)
            newPoly = Polynomial(self.x_dim, deg, self.para_index)

            # 更新多项式系数参数的index
            self.para_index = newPoly.para_index

            Pi, parameters, terms = newPoly.poly, newPoly.parameters, newPoly.terms
            prob_init.add_sos_constraint(Pi, self.x_sym)
            B_I = B_I + Pi * (self.x_sym[i] - self.X0[i][0]) * (self.x_sym[i] - self.X0[i][1])
        B_I = sp.expand(B_I)
        prob_init.add_sos_constraint(B_I, self.x_sym)
        try:
            prob_init.solve(solver='mosek')
            return True
        except:
            return False

    def SolveUnsafe(self, deg):
        '''
            con3条件验证, Xu中 B >= 1.

        :param deg: SOS多项式次数
        :return:
        '''

        prob_unsafe = SOSProblem()
        B_I = self.P_x.RealExpr - 1.

        for i in range(self.x_dim):
            # Pi, parameters, terms = self.polynomial(deg)
            newPoly = Polynomial(self.x_dim, deg, self.para_index)

            # 更新多项式系数参数的index
            self.para_index = newPoly.para_index

            Pi, parameters, terms = newPoly.poly, newPoly.parameters, newPoly.terms
            prob_unsafe.add_sos_constraint(Pi, self.x_sym)
            B_I = B_I + Pi * (self.x_sym[i] - self.Xu[i][0]) * (self.x_sym[i] - self.Xu[i][1])
        B_I = sp.expand(B_I)
        prob_unsafe.add_sos_constraint(B_I, self.x_sym)
        try:
            prob_unsafe.solve(solver='mosek')
            return True
        except:
            return False

    def SolveMECon(self, deg):
        '''
            con4条件验证, X中 E - B <= 0.

        :param deg: SOS多项式次数
        :return:
        '''

        prob_MECon = SOSProblem()

        # 数学期望计算, 选取的扰动的数学期望都为0
        dic = {}
        for i in range(self.x_dim):
            dic[self.x_sym[i]] = self.fx_expr[i]

        P_f_x = self.P_x.RealExpr.xreplace(dic)

        rv_mean = 0.
        dic = {}
        for i in range(self.random_var_num):
            dic[self.random_var[i]] = rv_mean
        E_P_f_x = P_f_x.subs(dic)

        B_I = sp.expand(self.P_x.RealExpr - E_P_f_x)

        # 记录E_P_f_x - P_x
        self.E_P_f_x_miner_P_x = sp.expand(E_P_f_x - self.P_x.RealExpr)

        for i in range(self.x_dim):
            # Pi, parameters, terms = self.polynomial(deg)
            newPoly = Polynomial(self.x_dim, deg, self.para_index)

            # 更新多项式系数参数的index
            self.para_index = newPoly.para_index

            Pi, parameters, terms = newPoly.poly, newPoly.parameters, newPoly.terms
            prob_MECon.add_sos_constraint(Pi, self.x_sym)
            B_I = B_I + Pi * (self.x_sym[i] - self.X[i][0]) * (self.x_sym[i] - self.X[i][1])
        B_I = sp.expand(B_I)
        prob_MECon.add_sos_constraint(B_I, self.x_sym)
        try:
            prob_MECon.solve(solver='mosek')
            return True
        except:
            return False


    def verify(self, deg):
        assert len(self.deg) == 4

        NonNeg = self.SolveNonNeg(deg[0])
        if not NonNeg:
            print('The NonNeg con is not satisfied.')
        else:
            print('The NonNeg con is satisfied.')

        Init = self.SolveInit(deg[1])
        if not Init:
            print('The init con is not satisfied.')
        else:
            print('The init con is satisfied.')

        Unsafe = self.SolveUnsafe(deg[2])
        if not Unsafe:
            print('The unsafe con is not satisfied.')
        else:
            print('The unsafe con is satisfied.')

        MECon = self.SolveMECon(deg[3])
        if not MECon:
            print('The ME con is not satisfied.')
        else:
            print('The ME con is satisfied.')

        return NonNeg, Init, Unsafe, MECon

if __name__ == '__main__':
    pass
