import numpy as np

# wait_and_judge 场景优化方法, 置信度计算


# 遍历所有的support constraint
def wait_and_judge_prob(d, N, bet):
    '''

    :param d: 优化问题中决策变量的个数
    :param N: 场景采样个数
    :param bet: 超参数
    :return: epsilon(k)
    '''

    # 用来存储结果
    out = np.zeros(d + 1)

    # 根据不同的support constraint个数
    # 一般只要求某个固定的support constraint k=? 即可
    for k in range(1, d + 1):
        #print(k)
        # m = [k:1:N] 即从k到N的整数序列
        m = np.arange(k, N + 1)
        # m的长度
        L = N - k + 1
        m_matrix = np.repeat([m], L, axis=0)

        # 取对数
        log_m_matrix = np.log(m_matrix)

        # 仅提取对角线上面的元素
        triu_log_m = np.triu(log_m_matrix, k=1)

        # 每行求和
        aux1 = np.sum(triu_log_m, axis=1)


        # ================================
        m_minus_k = m - k

        m_minus_k_matrix = np.repeat([m_minus_k], L, axis=0)

        # 取对数
        log_m_minus_k_matrix = np.log(m_minus_k_matrix)

        # 仅提取对角线上面的元素
        triu_log_m_minus_k = np.triu(log_m_minus_k_matrix, k=1)

        aux2 = np.sum(triu_log_m_minus_k, axis=1)

        coeffs = aux2 - aux1

        # 二分法求解
        t1 = 0.0
        t2 = 1.0

        while t2 - t1 > 1e-10:
            t = (t1 + t2) / 2
            exponent = coeffs - (N - m) * np.log(t)
            val = 1 - bet / (N + 1) * np.sum(np.exp(exponent))
            if val >= 0:
                t2 = t
            else:
                t1 = t

        out[k] = 1 - t1
    return out

# 只求特定的support constraint
def wait_and_judge_prob_at_k(N, bet, k):
    '''

    :param N: 场景采样个数
    :param bet: 超参数
    :param k: support constraint个数 (可以直接取值d)
    :return: epsilon(k)
    '''

    # 根据不同的support constraint个数
    # 一般只要求某个固定的support constraint k=? 即可

    # print(k)
    # m = [k:1:N] 即从k到N的整数序列
    m = np.arange(k, N + 1)
    # m的长度
    L = N - k + 1
    m_matrix = np.repeat([m], L, axis=0)

    # 取对数
    with np.errstate(divide='ignore'):
        log_m_matrix = np.log(m_matrix)

    # 仅提取对角线上面的元素
    triu_log_m = np.triu(log_m_matrix, k=1)

    # 每行求和
    aux1 = np.sum(triu_log_m, axis=1)

    # ================================
    m_minus_k = m - k

    m_minus_k_matrix = np.repeat([m_minus_k], L, axis=0)

    # 取对数
    with np.errstate(divide='ignore'):
        log_m_minus_k_matrix = np.log(m_minus_k_matrix)

    # 仅提取对角线上面的元素
    triu_log_m_minus_k = np.triu(log_m_minus_k_matrix, k=1)

    aux2 = np.sum(triu_log_m_minus_k, axis=1)

    coeffs = aux2 - aux1

    # 二分法求解
    t1 = 0.0
    t2 = 1.0

    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2
        exponent = coeffs - (N - m) * np.log(t)
        val = 1 - bet / (N + 1) * np.sum(np.exp(exponent))
        if val >= 0:
            t2 = t
        else:
            t1 = t

    epsilon_k = 1 - t1

    return epsilon_k


def Wait_and_Judge_Confidence(N, bet, k):
    print(f"==========[start_SO_alpha_beta_result]==============")
    epsilon_k = wait_and_judge_prob_at_k(N, bet, k)
    #sixelmath(r"\mathcal{P}^{N}\{\mathcal{P}\{x^{*} sat ori prob\} \geq 0\}")
    print("Pᴺ{{ P{{x* sat ori prob}} ≥ {alpha} }} ≥ {beta}".format(alpha=1 - epsilon_k, beta=1 - bet))
    print(f"==========[end_SO_alpha_beta_result]==============")


if __name__ == "__main__":
    N = 1000
    d = 101
    bet = 1e-6
    Wait_and_Judge_Confidence(N, bet, d)